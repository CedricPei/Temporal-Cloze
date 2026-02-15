"""Temporal Cloze 评测脚本

用法:
  python eval.py                                    # 评测全部题目
  python eval.py xMVtFHXrMR4.15_3                   # 评测指定题目
  python eval.py xMVtFHXrMR4.15_3 2Tz4_p9U56w.13_1  # 评测多个题目
"""

import base64
import json
import logging
import re
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# 路径
CHOICES_DIR = ROOT / "choices"
RESULTS_DIR = ROOT / "eval_results"
LOG_DIR = ROOT / "logs"

# 配置
NUM_FRAMES = 16
EVAL_MODEL = "google/gemini-2.5-pro"
# EVAL_MODEL = "bytedance-seed/seed-1.6"
NUM_WORKERS = 8

# 模型名：只用模型名，不含厂商（如 gemini-3-pro-preview, seed-1.6）
MODEL_TAG = EVAL_MODEL.split("/")[-1]

# S / A / C 三个维度的选项
DIMENSIONS = {
    "S": ["S/Rand1.mp4", "S/Rand2.mp4", "S/Rand3.mp4"],
    "A": ["A/Early.mp4", "A/Late.mp4", "A/Wide.mp4"],
    "C": ["C/Reverse.mp4", "C/Shuffle.mp4", "C/Loop.mp4"],
}

# Logging: 仅输出到日志文件，不输出到控制台
LOG_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
_fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", datefmt="%H:%M:%S")
_fh = logging.FileHandler(LOG_DIR / f"eval_{MODEL_TAG}.log", encoding="utf-8")
_fh.setFormatter(_fmt)
log.addHandler(_fh)


# ==================== 帧采样与编码 ====================

def sample_and_encode(video_path: Path) -> list[str]:
    """均匀采样 NUM_FRAMES 帧并编码为 base64 JPEG"""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    n = min(NUM_FRAMES, total)
    indices = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]

    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return out


# ==================== Prompt ====================

def build_prompt() -> str:
    return """You are given:
- **BEGINNING**: the first part of a video (frames in temporal order).
- **END**: the last part of the same video (frames in temporal order).
The middle segment between BEGINNING and END was removed.

You will then see four candidate middle segments, labeled A, B, C, D. Each candidate is a short clip; exactly one is the true middle that connects BEGINNING to END. The others are wrong.

Task: Which candidate is the correct middle? Choose one of A, B, C, D.

Output JSON only: {"answer": "<A, B, C, or D>", "reason": "<one or two sentences>"}"""


def parse_eval_response(raw: str, letters: list[str]) -> tuple[str | None, str]:
    """从模型返回文本中尽量解析出 answer 和 reason。letters 为可选字母如 ['A','B','C','D']。"""
    raw = (raw or "").strip()
    if not raw:
        return None, ""

    # 1) 去掉 markdown 代码块包裹
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # 2) 尝试直接解析 JSON
    try:
        parsed = json.loads(raw)
        answer = (parsed.get("answer") or "").strip().upper()
        reason = parsed.get("reason") or ""
        if answer in letters:
            return answer, reason
    except json.JSONDecodeError:
        pass

    # 3) 尝试从首尾截取 {...} 再解析（前后可能有说明文字）
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            answer = (parsed.get("answer") or "").strip().upper()
            reason = parsed.get("reason") or ""
            if answer in letters:
                return answer, reason
        except json.JSONDecodeError:
            pass

    # 4) 解析失败：从 raw 中抽取 answer（第一个出现的可选字母）和 reason 字段值
    answer = None
    for c in raw.upper():
        if c in letters:
            answer = c
            break
    reason = raw
    m = re.search(r'"reason"\s*:\s*"', raw)
    if m:
        start = m.end()
        i = start
        while i < len(raw):
            if raw[i] == "\\" and i + 1 < len(raw):
                i += 2
                continue
            if raw[i] == '"':
                reason = raw[start:i].replace("\\n", "\n").replace('\\"', '"')
                break
            i += 1
        else:
            reason = raw[start:].replace("\\n", "\n").replace('\\"', '"')
    return answer, reason


# ==================== 单题评测 ====================

def eval_one(client: OpenAI, stem: str, dim: str, distractors: list[str]) -> dict:
    """对一个视频的一个维度做 1-of-4 评测"""
    base = CHOICES_DIR / stem

    before_b64 = sample_and_encode(base / "before.mp4")
    after_b64 = sample_and_encode(base / "after.mp4")


    # GT + 3个干扰，打乱
    options = [("GT.mp4", True)] + [(d, False) for d in distractors]
    random.shuffle(options)
    letters = [chr(65 + i) for i in range(len(options))]
    correct_letter = next(letters[i] for i, (_, is_gt) in enumerate(options) if is_gt)

    # 构建 API content
    content = [{"type": "text", "text": build_prompt()}]

    content.append({"type": "text", "text": "[BEGINNING]"})
    for b in before_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

    content.append({"type": "text", "text": "[END]"})
    for b in after_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

    for i, (rel_path, _) in enumerate(options):
        full_path = base / rel_path

        opt_b64 = sample_and_encode(full_path)

        content.append({"type": "text", "text": f"[Candidate {letters[i]}]"})
        for b in opt_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

    # 调用 API，最多重试 3 次
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
                temperature=0,
            )
            if not resp.choices or not resp.choices[0].message.content:
                raise ValueError("Empty response from API")

            raw = resp.choices[0].message.content.strip()
            answer, reason = parse_eval_response(raw, letters)
            correct = answer == correct_letter
            return {"stem": stem, "dim": dim, "correct": correct,
                    "answer": answer, "expected": correct_letter, "reason": reason}
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 5
                time.sleep(wait)
            else:
                full_msg = str(e)
                if hasattr(e, "response") and e.response is not None:
                    r = e.response
                    body = getattr(r, "text", getattr(r, "content", repr(r)))
                    full_msg += f"\nFull response: {body}"
                log.error(f"API error on {stem} {dim}:\n{full_msg}")
                return {"stem": stem, "dim": dim, "correct": False,
                        "answer": None, "expected": correct_letter, "reason": None, "error": str(e)[:200]}


# ==================== Main ====================

def run(targets: list[str] = None):
    """targets: 指定题目stem列表, None则全部评测"""
    # 从 choices 目录中直接获取已有题目（存在 GT.mp4 的子目录）
    if not CHOICES_DIR.exists():
        log.error(f"Choices directory not found: {CHOICES_DIR}")
        return

    all_stems = sorted(
        p.name for p in CHOICES_DIR.iterdir()
        if p.is_dir() and (p / "GT.mp4").exists()
    )

    if targets:
        stems = [s for s in targets if s in all_stems]
    else:
        stems = all_stems

    log.info(f"Model: {EVAL_MODEL}")
    log.info(f"Num videos: {len(stems)}, total tasks ≈ {len(stems) * 3}")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{MODEL_TAG}.json"

    # 加载已有结果（支持断点续做）
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # 构建任务
    tasks = []
    for stem in stems:
        if not (CHOICES_DIR / stem / "GT.mp4").exists():
            continue
        for dim, distractors in DIMENSIONS.items():
            if stem in all_results and dim in all_results[stem]:
                continue
            tasks.append((stem, dim, distractors))

    log.info(f"Tasks to run: {len(tasks)}, already done: {sum(len(v) for v in all_results.values())}")

    if not tasks:
        log.info("No tasks to run.")
    else:
        max_workers = min(NUM_WORKERS, max(1, os.cpu_count() or 1))
        future_to_task: dict = {}
        saved_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for stem, dim, distractors in tasks:
                fut = executor.submit(eval_one, client, stem, dim, distractors)
                future_to_task[fut] = (stem, dim)

            for fut in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Evaluating"):
                stem, dim = future_to_task[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    # 单个题目 worker 出错，只跳过该题，继续其它题目
                    log.error(f"Unexpected error in worker for {stem} {dim}: {e}")
                    continue

                # error 则不写入该题结果，但不中止整体评测
                if "error" in result:
                    # 打印完整 error message，便于排查 provider 返回的原始信息
                    err_msg = result["error"]
                    log.error(f"API error on {stem} {dim}: {err_msg}")
                    print(f"[eval] ERROR: {stem} {dim} API error: {err_msg}")
                    continue

                # 去掉 stem/dim 字段，放进嵌套结构
                entry = {k: v for k, v in result.items() if k not in ("stem", "dim")}
                if stem not in all_results:
                    all_results[stem] = {}
                all_results[stem][dim] = entry
                saved_count += 1

                tag = "✓" if result["correct"] else "✗"
                tqdm.write(
                    f"  {tag} {stem} {dim}  ans={result['answer']} "
                    f"exp={result['expected']}  {(result.get('reason') or '')[:60]}"
                )

                # 每题写入一次
                try:
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    log.error(f"Failed to save results to {results_path}: {e}")

        log.info(f"Saved {saved_count}/{len(tasks)} tasks to {results_path}")

    # 汇总
    for dim in DIMENSIONS:
        entries = [v[dim] for v in all_results.values() if dim in v]
        c = sum(1 for e in entries if e["correct"])
        t = len(entries)
        log.info(f"{dim}: {c}/{t} = {c/t:.2%}" if t else f"{dim}: no data")

    all_entries = [e for v in all_results.values() for e in v.values()]
    total = len(all_entries)
    correct = sum(1 for e in all_entries if e["correct"])


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    run(targets)
