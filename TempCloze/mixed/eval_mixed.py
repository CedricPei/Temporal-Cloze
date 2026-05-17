"""Temp-Mixed 评测脚本

题型设计：每道题含 4 个候选项 —— GT（正确答案）+ 1 个 S 类干扰项 + 1 个 A 类干扰项
+ 1 个 C/P 类干扰项。三种维度的干扰项同时出现在同一题中，具体干扰项由
mixed_ids.json 固定记录。

用法:
  python eval_mixed.py                          # 评测所有 4 个模型
  python eval_mixed.py --model gemini-2.5-pro   # 只评测单个模型
  python eval_mixed.py --seed 42                # 指定选项排列随机种子
"""

import argparse
import base64
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
from dotenv import dotenv_values
from openai import OpenAI
from tqdm import tqdm

# ==================== 路径 ====================

MIXED_DIR = Path(__file__).parent
ROOT = MIXED_DIR.parent                        # TempCloze/
PROJECT_ROOT = ROOT.parent                     # Temporal-Cloze/
CHOICES_DIR = ROOT / "choices"
MIXED_IDS_PATH = MIXED_DIR / "mixed_ids.json"
RESULTS_DIR = MIXED_DIR / "eval_results"

# ==================== 参数 ====================

MODELS = [
    "doubao-seed-1-6-251015",
    "gemini-2.5-pro",
    "qwen3.5-397b-a17b",
    "qwen3-vl-8b-instruct",
]

NUM_FRAMES = 16
MAX_HEIGHT = 360
MAX_RETRIES = 3
NUM_WORKERS = 16
SEED = 42

EVAL_PROMPT = """You are given:
- **BEGINNING**: the first part of a video (frames in temporal order).
- **END**: the last part of the same video (frames in temporal order).
The middle segment between BEGINNING and END was removed.

You will then see four candidate middle segments, labeled A, B, C, D. Each candidate is a short clip; exactly one is the true middle that connects BEGINNING to END. The others are wrong.

Task: Which candidate is the correct middle? Choose one of A, B, C, D.

**Output JSON only** : {"answer": "<A, B, C, or D>", "reason": "<one or two sentences>"}"""


# ==================== 环境变量 ====================

def _is_blank(v):
    return v is None or not str(v).strip()


def _load_env():
    keys = ("OPENAI_API_KEY", "OPENAI_BASE_URL")
    if all(not _is_blank(os.getenv(k)) for k in keys):
        return
    for env_path in (MIXED_DIR / ".env", ROOT / ".env", PROJECT_ROOT / ".env"):
        if not env_path.exists():
            continue
        env_map = dotenv_values(env_path)
        for k in keys:
            if _is_blank(os.getenv(k)):
                v = env_map.get(k)
                if v and str(v).strip():
                    os.environ[k] = str(v).strip()
        if all(not _is_blank(os.getenv(k)) for k in keys):
            return


_load_env()


# ==================== 帧采样与编码 ====================

def sample_and_encode(video_path: Path) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 2:
        cap.release()
        return []
    n = min(NUM_FRAMES, total - 2)
    indices = [1 + int((i + 0.5) * (total - 2) / n) for i in range(n)]
    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            if h > MAX_HEIGHT:
                scale = MAX_HEIGHT / h
                frame = cv2.resize(frame, (int(w * scale), MAX_HEIGHT))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return out


# ==================== 响应解析 ====================

def parse_response(raw: str, letters: list[str]) -> tuple[str | None, str]:
    raw = (raw or "").strip()
    if not raw:
        return None, ""

    import re
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        try:
            p = json.loads(m.group(1).strip())
            ans = (p.get("answer") or "").strip().upper()
            if ans in letters:
                return ans, p.get("reason") or ""
        except (json.JSONDecodeError, AttributeError):
            pass

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            p = json.loads(raw[start:end + 1])
            ans = (p.get("answer") or "").strip().upper()
            if ans in letters:
                return ans, p.get("reason") or ""
        except (json.JSONDecodeError, AttributeError):
            pass

    return None, raw


# ==================== 单题评测 ====================

def build_content(stem: str, options: list[tuple[str, bool]]) -> tuple[list[dict], str, dict]:
    """构建 API content，返回 (content, correct_letter, option_map)。"""
    base = CHOICES_DIR / stem
    letters = [chr(65 + i) for i in range(len(options))]
    correct_letter = next(letters[i] for i, (_, is_gt) in enumerate(options) if is_gt)
    option_map = {letters[i]: Path(rel).stem for i, (rel, _) in enumerate(options)}

    content: list[dict] = [{"type": "text", "text": EVAL_PROMPT}]
    content.append({"type": "text", "text": "[BEGINNING]"})
    for b in sample_and_encode(base / "before.mp4"):
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    content.append({"type": "text", "text": "[END]"})
    for b in sample_and_encode(base / "after.mp4"):
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    for i, (rel_path, _) in enumerate(options):
        content.append({"type": "text", "text": f"[Candidate {letters[i]}]"})
        for b in sample_and_encode(base / rel_path):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

    return content, correct_letter, option_map


def eval_one(client: OpenAI, model: str, item: dict, seed: int) -> dict:
    """对一道 mixed 题做 1-of-4 评测。"""
    stem = item["stem"]
    s_dist = item["S"]
    a_dist = item["A"]
    c_dist = item["C"]

    # 4 个选项：GT + S干扰 + A干扰 + C干扰，打乱顺序
    raw_options = [
        ("GT.mp4", True),
        (s_dist, False),
        (a_dist, False),
        (c_dist, False),
    ]
    rng = random.Random(f"{seed}:{stem}")
    rng.shuffle(raw_options)
    letters = [chr(65 + i) for i in range(4)]

    # option_dim_map: 字母 → 干扰维度类型
    dim_label = {"GT.mp4": "GT", **{s_dist: "S", a_dist: "A", c_dist: "C"}}

    content, correct_letter, option_map = build_content(stem, raw_options)
    option_dim = {letters[i]: dim_label[rel] for i, (rel, _) in enumerate(raw_options)}

    for attempt in range(MAX_RETRIES):
        try:
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=8192,
                temperature=0,
            )
            if not resp.choices or not resp.choices[0].message.content:
                raise ValueError("Empty response from API")
            raw = resp.choices[0].message.content.strip()
            answer, reason = parse_response(raw, letters)
            if answer is None:
                raise ValueError(f"Cannot parse answer: {raw[:200]}")
            return {
                "stem": stem,
                "correct": answer == correct_letter,
                "answer": answer,
                "expected": correct_letter,
                "reason": reason,
                "option_map": option_map,
                "option_dim": option_dim,
                "error_dim": option_dim.get(answer) if answer != correct_letter else None,
                "raw": raw,
                "timestamp_utc": timestamp_utc,
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 5)
            else:
                full_msg = str(e)
                if hasattr(e, "response") and e.response is not None:
                    r = e.response
                    body = getattr(r, "text", getattr(r, "content", repr(r)))
                    full_msg += f"\nFull response: {body}"
                print(f"[eval] API error on {stem}: {full_msg}", flush=True)
                return None


# ==================== 单模型评测流程 ====================

def run_model(model: str, items: list[dict], seed: int):
    model_tag = model.split("/")[-1]
    results_path = RESULTS_DIR / f"{model_tag}.json"

    if results_path.exists():
        all_results = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        all_results = {}

    tasks = [item for item in items if item["stem"] not in all_results]
    done = len(all_results)

    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Tasks: {len(tasks)}, already done: {done}", flush=True)

    if not tasks:
        print("No new tasks.", flush=True)
        _summarize(model_tag, all_results)
        return

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if not api_key or not base_url:
        print("[eval] Missing OPENAI_API_KEY or OPENAI_BASE_URL.", flush=True)
        return
    client = OpenAI(api_key=api_key, base_url=base_url)

    saved = 0
    with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, max(1, os.cpu_count()))) as executor:
        futures = {
            executor.submit(eval_one, client, model, item, seed): item["stem"]
            for item in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"[{model_tag}]"):
            stem = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"[eval] Unexpected error for {stem}: {e}", flush=True)
                continue

            if result is None:
                continue

            entry = {k: v for k, v in result.items() if k != "stem"}
            all_results[stem] = entry
            saved += 1

            tag = "✓" if result["correct"] else "✗"
            err = f" (chose {result['error_dim']})" if result.get("error_dim") else ""
            tqdm.write(f"  {tag} {stem}  ans={result['answer']} exp={result['expected']}{err}")

            if saved % 20 == 0:
                results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"Saved {saved} new → {results_path}", flush=True)
    _summarize(model_tag, all_results)


def _summarize(model_tag: str, results: dict):
    total = len(results)
    if total == 0:
        print(f"[{model_tag}] No results.", flush=True)
        return
    correct = sum(1 for v in results.values() if v.get("correct"))
    acc = correct / total
    print(f"\n--- [{model_tag}] Accuracy: {correct}/{total} = {acc:.1%} ---", flush=True)

    # 错误来源分析
    errors = [v for v in results.values() if not v.get("correct") and v.get("error_dim")]
    if errors:
        dim_counts = {}
        for v in errors:
            d = v["error_dim"]
            dim_counts[d] = dim_counts.get(d, 0) + 1
        print(f"  Error source: { {d: f'{c}/{len(errors)} ({c/len(errors):.1%})' for d, c in sorted(dim_counts.items())} }",
              flush=True)


# ==================== Main ====================

def run(model_filter: str | None = None, seed: int = SEED):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    items = json.loads(MIXED_IDS_PATH.read_text(encoding="utf-8"))
    print(f"Temp-Mixed: {len(items)} videos, seed={seed}")

    models = [m for m in MODELS if model_filter is None or model_filter in m] if model_filter else MODELS
    if not models:
        print(f"No model matches filter: {model_filter}")
        return

    for model in models:
        run_model(model, items, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temp-Mixed evaluator: GT+S+A+C mixed-distractor 4-way MCQ")
    parser.add_argument("--model", type=str, default=None, help="Filter to specific model (substring match)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for option shuffling (default: 42)")
    args = parser.parse_args()
    run(args.model, args.seed)
