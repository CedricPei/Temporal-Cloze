"""Temporal Cloze 评测脚本

用法:
  python eval.py                                    # 评测全部题目
  python eval.py xMVtFHXrMR4.15_3                   # 评测指定题目
  python eval.py xMVtFHXrMR4.15_3 2Tz4_p9U56w.13_1  # 评测多个题目
"""

import base64
import json
import logging
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

CHOICES_DIR = ROOT / "choices"
RESULTS_DIR = ROOT / "eval_results"
LOG_DIR = ROOT / "logs"

NUM_FRAMES = 16
MAX_HEIGHT = 360
EVAL_MODEL = "google/gemini-2.5-pro"
# EVAL_MODEL = "bytedance-seed/seed-1.6"
NUM_WORKERS = 8
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

Output JSON only: {"answer": "<A, B, C, or D>", "reason": "<one or two sentences>"}"""

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
            h, w = frame.shape[:2]
            if h > MAX_HEIGHT:
                scale = MAX_HEIGHT / h
                frame = cv2.resize(frame, (int(w * scale), MAX_HEIGHT))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return out


# ==================== 响应解析 ====================

def parse_eval_response(raw: str, letters: list[str]) -> tuple[str | None, str]:
    """从模型返回文本中解析 answer 和 reason。"""
    raw = (raw or "").strip()
    if not raw:
        return None, ""

    # 去掉 markdown 代码块包裹
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # 尝试直接解析 JSON
    try:
        parsed = json.loads(raw)
        answer = (parsed.get("answer") or "").strip().upper()
        reason = parsed.get("reason") or ""
        if answer in letters:
            return answer, reason
    except json.JSONDecodeError:
        pass

    # 尝试从文本中截取 {...} 再解析
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

    return None, raw


# ==================== 单题评测 ====================

def eval_one(client: OpenAI, stem: str, dim: str, distractors: list[str]) -> dict:
    """对一个视频的一个维度做 1-of-4 评测"""
    base = CHOICES_DIR / stem

    before_b64 = sample_and_encode(base / "before.mp4")
    after_b64 = sample_and_encode(base / "after.mp4")

    options = [("GT.mp4", True)] + [(d, False) for d in distractors]
    random.shuffle(options)
    letters = [chr(65 + i) for i in range(len(options))]
    correct_letter = next(letters[i] for i, (_, is_gt) in enumerate(options) if is_gt)

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

    for attempt in range(MAX_RETRIES):
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
            if answer is None:
                raise ValueError(f"Cannot parse answer from response: {raw[:200]}")
            option_map = {letters[i]: Path(rel_path).stem for i, (rel_path, _) in enumerate(options)}
            return {"stem": stem, "dim": dim, "correct": answer == correct_letter,
                    "answer": answer, "expected": correct_letter, "reason": reason,
                    "option_map": option_map}
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 5)
            else:
                full_msg = str(e)
                if hasattr(e, "response") and e.response is not None:
                    r = e.response
                    body = getattr(r, "text", getattr(r, "content", repr(r)))
                    full_msg += f"\nFull response: {body}"
                log.error(f"API error on {stem} {dim}:\n{full_msg}")
                return {"stem": stem, "dim": dim, "error": str(e)[:200]}


# ==================== Main ====================

def run(targets: list[str] | None = None):
    if not CHOICES_DIR.exists():
        log.error(f"Choices directory not found: {CHOICES_DIR}")
        return

    all_stems = sorted(
        p.name for p in CHOICES_DIR.iterdir()
        if p.is_dir() and (p / "GT.mp4").exists()
    )

    stems = [s for s in targets if s in all_stems] if targets else all_stems

    log.info(f"Model: {EVAL_MODEL}")
    log.info(f"Num videos: {len(stems)}, total tasks ≈ {len(stems) * 3}")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
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

    log.info(f"Tasks to run: {len(tasks)}, already done: {sum(len(v) for v in all_results.values())}")

    if not tasks:
        log.info("No tasks to run.")
    else:
        max_workers = min(NUM_WORKERS, max(1, os.cpu_count()))
        saved_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(eval_one, client, stem, dim, distractors): (stem, dim)
                for stem, dim, distractors in tasks
            }

            for fut in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Evaluating"):
                stem, dim = future_to_task[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    log.error(f"Unexpected error in worker for {stem} {dim}: {e}")
                    continue

                if "error" in result:
                    log.error(f"API error on {stem} {dim}: {result['error']}")
                    print(f"[eval] ERROR: {stem} {dim} — {result['error']}")
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
                    log.error(f"Failed to save results: {e}")

        log.info(f"Saved {saved_count}/{len(tasks)} tasks to {results_path}")



if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    run(targets)
