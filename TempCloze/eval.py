"""Temporal Cloze 评测脚本

用法:
  python eval.py video-cloze                 # 默认评测全部视频
  python eval.py subset 200 --seed 42        # 指定样本数与随机种子
"""

import argparse
import base64
from datetime import datetime, timezone
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from dotenv import dotenv_values
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).parent

def _is_blank_env(value: str | None) -> bool:
    return value is None or not str(value).strip()

def _load_api_env_from_files() -> None:
    keys = ("OPENAI_API_KEY", "OPENAI_BASE_URL")

    # Prefer current shell env vars; only fill missing/blank values from .env.
    if all(not _is_blank_env(os.getenv(k)) for k in keys):
        return

    for env_path in (ROOT / ".env", ROOT.parent / ".env"):
        if not env_path.exists():
            continue
        env_map = dotenv_values(env_path)
        for k in keys:
            if _is_blank_env(os.getenv(k)):
                v = env_map.get(k)
                if v is not None and str(v).strip():
                    os.environ[k] = str(v).strip()
        if all(not _is_blank_env(os.getenv(k)) for k in keys):
            return

_load_api_env_from_files()

def _mask_key(value: str) -> str:
    if not value:
        return "<EMPTY>"
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:6]}...{value[-4:]}"

DEFAULT_PRESET = "video-cloze"
CHOICES_SOURCE = {
    "video-cloze": ROOT  / "choices",
    "subset": ROOT / "subset" / "choices",
}

PRESET = DEFAULT_PRESET
PRESET_DIR = ROOT / PRESET
CHOICES_DIR = CHOICES_SOURCE.get(PRESET, PRESET_DIR / "choices")
RESULTS_DIR = PRESET_DIR / "eval_results_extra"

NUM_FRAMES = 16
MAX_HEIGHT = 360
EVAL_MODEL = "doubao-seed-1-8-251228-thinking"
NUM_WORKERS = 16
MAX_RETRIES = 3

MODEL_TAG = EVAL_MODEL.split("/")[-1]

DIMENSIONS = {
    "S": ["S/Rand1.mp4", "S/Rand2.mp4", "S/Rand3.mp4"],
    "A": ["A/Early.mp4", "A/Late.mp4", "A/Wide.mp4"],
    "C": ["C/Reverse.mp4", "C/Shuffle.mp4", "C/Loop.mp4"],
}

EVAL_PROMPT = """You are given:
- **BEGINNING**: the first part of a video (frames in temporal order).
- **END**: the last part of the same video (frames in temporal order).
The middle segment between BEGINNING and END was removed.

You will then see four candidate middle segments, labeled A, B, C, D. Each candidate is a short clip; exactly one is the true middle that connects BEGINNING to END. The others are wrong.

Task: Which candidate is the correct middle? Choose one of A, B, C, D.

**Output JSON only** : {"answer": "<A, B, C, or D>", "reason": "<one or two sentences>"}"""


# ==================== 帧采样与编码 ====================

def sample_and_encode(video_path: Path) -> list[str]:
    """均匀采样 NUM_FRAMES 帧并编码为 base64 JPEG"""
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

def _normalize_raw_json(answer: str, reason: str) -> str:
    return json.dumps({"answer": answer, "reason": reason}, ensure_ascii=False)


def parse_eval_response(raw: str, letters: list[str]) -> tuple[str | None, str, str | None]:
    """从模型返回文本中解析 answer、reason，并返回规范化 raw。"""
    raw = (raw or "").strip()
    if not raw:
        return None, "", None

    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
        answer = (parsed.get("answer") or "").strip().upper()
        reason = parsed.get("reason") or ""
        if answer in letters:
            return answer, reason, _normalize_raw_json(answer, reason)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            answer = (parsed.get("answer") or "").strip().upper()
            reason = parsed.get("reason") or ""
            if answer in letters:
                return answer, reason, _normalize_raw_json(answer, reason)
        except json.JSONDecodeError:
            pass

    return None, "", None


# ==================== 单题评测 ====================

def eval_one(client: OpenAI, stem: str, dim: str, distractors: list[str], seed: int) -> dict:
    """对一个视频的一个维度做 1-of-4 评测"""
    base = CHOICES_DIR / stem

    before_b64 = sample_and_encode(base / "before.mp4")
    after_b64 = sample_and_encode(base / "after.mp4")

    options = [("GT.mp4", True)] + [(d, False) for d in distractors]
    rng = random.Random(f"{seed}:{stem}:{dim}")
    rng.shuffle(options)
    letters = [chr(65 + i) for i in range(len(options))]
    correct_letter = next(letters[i] for i, (_, is_gt) in enumerate(options) if is_gt)
    option_map = {letters[i]: Path(rel_path).stem for i, (rel_path, _) in enumerate(options)}

    content: list[dict] = [{"type": "text", "text": EVAL_PROMPT}]

    content.append({"type": "text", "text": "[BEGINNING]"})
    for b in before_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

    content.append({"type": "text", "text": "[END]"})
    for b in after_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

    for i, (rel_path, _) in enumerate(options):
        opt_b64 = sample_and_encode(base / rel_path)
        content.append({"type": "text", "text": f"[Candidate {letters[i]}]"})
        for b in opt_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

    last_raw_original: str | None = None
    last_raw_normalized: str | None = None

    for attempt in range(MAX_RETRIES):
        try:
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            resp = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=8192,
                temperature=0,
            )
            if not resp.choices or not resp.choices[0].message.content:
                raise ValueError("Empty response from API")
            raw = resp.choices[0].message.content.strip()
            last_raw_original = raw
            answer, reason, normalized_raw = parse_eval_response(raw, letters)
            last_raw_normalized = normalized_raw
            if answer is None:
                raise ValueError(f"Cannot parse answer from response: {raw[:200]}")
            return {"stem": stem, "dim": dim, "correct": answer == correct_letter,
                    "answer": answer, "expected": correct_letter, "reason": reason,
                    "raw_original": raw, "raw": normalized_raw, "option_map": option_map,
                    "timestamp_utc": timestamp_utc}
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 5)
            else:
                timestamp_utc = datetime.now(timezone.utc).isoformat()
                full_msg = str(e)
                if hasattr(e, "response") and e.response is not None:
                    r = e.response
                    body = getattr(r, "text", getattr(r, "content", repr(r)))
                    full_msg += f"\nFull response: {body}"
                print(f"[eval] API error on {stem} {dim}:\n  {full_msg}", flush=True)
                return {
                    "stem": stem,
                    "dim": dim,
                    "correct": False,
                    "answer": None,
                    "expected": correct_letter,
                    "reason": None,
                    "option_map": option_map,
                    "error": str(e)[:200],
                    "raw_original": last_raw_original,
                    "raw": last_raw_normalized,
                    "timestamp_utc": timestamp_utc,
                }


# ==================== Main ====================

def run(preset: str = DEFAULT_PRESET, sample_size: int | None = None, seed: int = 42):
    global PRESET, PRESET_DIR, CHOICES_DIR, RESULTS_DIR

    if preset not in CHOICES_SOURCE:
        print(f"Unknown preset: {preset}, use 'video-cloze' or 'subset'")
        return

    if sample_size is not None and sample_size <= 0:
        print(f"Invalid sample_size: {sample_size}. Use a positive integer.")
        return

    PRESET = preset
    PRESET_DIR = ROOT / PRESET
    CHOICES_DIR = CHOICES_SOURCE.get(PRESET, PRESET_DIR / "choices")
    RESULTS_DIR = PRESET_DIR / "eval_results_extra"

    if not CHOICES_DIR.exists():
        print(f"Choices directory not found: {CHOICES_DIR}")
        return

    all_stems = sorted(
        p.name for p in CHOICES_DIR.iterdir()
        if p.is_dir() and (p / "GT.mp4").exists()
    )

    if not all_stems:
        print(f"No valid stems found in: {CHOICES_DIR}")
        return

    if sample_size is not None and sample_size < len(all_stems):
        sample_rng = random.Random(seed)
        stems = sorted(sample_rng.sample(all_stems, sample_size))
    else:
        stems = all_stems

    print(f"Preset: {PRESET}")
    print(f"Model: {EVAL_MODEL}")
    if sample_size is None:
        print(f"Sampling videos: {len(stems)} (full dataset, seed={seed})")
    else:
        print(f"Sampling videos: {len(stems)} (sample_size={sample_size}, seed={seed})")
    print(f"Num videos: {len(stems)}, total tasks ≈ {len(stems) * 3}")

    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    shown_key = _mask_key(api_key)
    print(f"Base URL: {base_url or '<EMPTY>'}")
    print(f"API Key: {shown_key}")

    if not base_url or not api_key:
        print("[eval] Missing OPENAI_BASE_URL or OPENAI_API_KEY after env resolution.")
        return
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{MODEL_TAG}.json"

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    tasks = []
    for stem in stems:
        for dim, distractors in DIMENSIONS.items():
            if stem in all_results and dim in all_results[stem]:
                continue
            tasks.append((stem, dim, distractors))

    print(f"Tasks to run: {len(tasks)}, already done: {sum(len(v) for v in all_results.values())}")

    if not tasks:
        print("No tasks to run.")
    else:
        max_workers = min(NUM_WORKERS, max(1, os.cpu_count()))
        saved_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(eval_one, client, stem, dim, distractors, seed): (stem, dim)
                for stem, dim, distractors in tasks
            }

            for fut in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Evaluating"):
                stem, dim = future_to_task[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"[eval] Unexpected error for {stem} {dim}: {e}", flush=True)
                    continue

                if "error" in result:
                    print(f"[eval] ERROR: {stem} {dim} — {result['error']}")
                    all_results.setdefault(stem, {})[dim] = {
                        "correct": False,
                        "answer": result.get("answer"),
                        "expected": result.get("expected"),
                        "reason": result.get("reason"),
                        "option_map": result.get("option_map"),
                        "error": result["error"],
                        "raw_original": result.get("raw_original"),
                        "raw": result.get("raw"),
                        "timestamp_utc": result.get("timestamp_utc"),
                    }
                    try:
                        with open(results_path, "w", encoding="utf-8") as f:
                            json.dump(all_results, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"[eval] Failed to save results: {e}", flush=True)
                    continue

                entry = {k: v for k, v in result.items() if k not in ("stem", "dim")}
                all_results.setdefault(stem, {})[dim] = entry
                saved_count += 1

                tag = "✓" if result["correct"] else "✗"
                tqdm.write(
                    f"  {tag} {stem} {dim}  ans={result['answer']} "
                    f"exp={result['expected']}  {(result.get('reason') or '')[:60]}"
                )

                try:
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[eval] Failed to save results: {e}", flush=True)

        print(f"Saved {saved_count}/{len(tasks)} tasks to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Cloze evaluator (sampled + reproducible)")
    parser.add_argument("preset", nargs="?", default=DEFAULT_PRESET, choices=sorted(CHOICES_SOURCE.keys()))
    parser.add_argument("sample_size", nargs="?", type=int, default=None, help="How many videos to sample (default: use all videos)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling and option shuffle")
    args = parser.parse_args()

    run(args.preset, args.sample_size, args.seed)
