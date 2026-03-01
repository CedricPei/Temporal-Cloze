"""从 hdvg_300k_720p.csv 下载视频片段：o3 预筛 → yt-dlp 下载 → 质量检查。

Pipeline:  CSV → duration 过滤 → o3 caption 预筛 → 下载 → 质量检查 → 保留/删除
输出:      output/lvd/{llm_filter,rejected}.json
"""

import ast
import csv
import json
import os
import random
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ==================== 路径 & 配置 ====================

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
CSV_PATH = ROOT / "hdvg_300k_720p.csv"
OUT_DIR = ROOT / "downloaded"
CHOICES_DIR = PROJECT_ROOT / "choices"
COOKIES_PATH = ROOT / "cookies.txt"
OUTPUT_DIR = PROJECT_ROOT / "output" / "lvd"

NUM_VIDEOS = 50
MIN_DURATION = 12.0
MAX_DURATION = 90.0
MAX_HEIGHT = 360
NUM_WORKERS = 8

load_dotenv(PROJECT_ROOT / ".env")

# ==================== JSON 持久化 ====================


def _load(path: Path) -> dict:
    return json.load(open(path, encoding="utf-8")) if path.exists() else {}


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ==================== 质量检查 ====================


def check_quality(path: Path) -> str:
    """检查视频质量，返回 'LOW' / 'MEDIUM' / 'HIGH'。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return "LOW"
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    bitrate = path.stat().st_size * 8 / duration / 1000 if duration > 0 else 0
    blur_scores = []
    for _ in range(min(10, frames)):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    cap.release()
    sharpness = sum(blur_scores) / len(blur_scores) if blur_scores else 0
    if bitrate < 200 or sharpness < 30:
        return "LOW"
    if bitrate >= 1000 and sharpness >= 100:
        return "HIGH"
    return "MEDIUM"


# ==================== CSV 解析 ====================


def _parse_span(s: str) -> tuple[str, str] | None:
    try:
        val = ast.literal_eval(s or "")
    except (SyntaxError, ValueError):
        return None
    if isinstance(val, (list, tuple)) and len(val) == 2 and all(isinstance(v, str) for v in val):
        return val[0], val[1]
    return None


def _ts(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_candidates() -> list[dict]:
    rows: list[dict] = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r.get("key") or "").strip()
            url = (r.get("url") or "").strip()
            span = _parse_span(r.get("orig_span", ""))
            caption = (r.get("refined_caption") or "").strip()
            if key and url and span:
                rows.append({"key": key, "url": url, "start": span[0], "end": span[1], "caption": caption})
    return rows


# ==================== yt-dlp 下载 ====================


def download_one(key: str, url: str, start: str, end: str) -> bool:
    out_path = OUT_DIR / f"{key}.mp4"
    if out_path.exists():
        return True
    cmd = [
        "yt-dlp", "--js-runtimes", "node", "--remote-components", "ejs:github",
        "--merge-output-format", "mp4",
        "-f", f"bv*[height<={MAX_HEIGHT}]/bv[height<={MAX_HEIGHT}]",
        "-o", str(out_path),
        "--download-sections", f"*{start}-{end}",
    ]
    if COOKIES_PATH.exists():
        cmd += ["--cookies", str(COOKIES_PATH)]
    cmd.append(url)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


# ==================== o3 过滤 ====================

O3_PROMPT = """[Temporal Cloze Task Suitability]
Can this video be used for a "temporal cloze" task?
(Given the beginning and end of a video, predict what happens in the middle gap)

✓ PASS: Video has causal/temporal continuity where the middle can be inferred from before & after
✗ REJECT: Middle segment cannot be meaningfully predicted

Caption: {caption}

JSON: {{"pass": true/false, "reason": "one or two brief sentences"}}
"""


def o3_filter(client: OpenAI, caption: str) -> tuple[bool, str]:
    if not (caption or "").strip():
        return False, "EMPTY"
    resp = client.chat.completions.create(
        model="openai/o3",
        messages=[
            {"role": "system", "content": "You are a video content classifier. Output valid JSON only."},
            {"role": "user", "content": O3_PROMPT.format(caption=caption)},
        ],
        max_tokens=500,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        return bool(data.get("pass")), data.get("reason", "")
    except json.JSONDecodeError:
        return False, raw


# ==================== Main ====================


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    llm_results = _load(OUTPUT_DIR / "llm_filter.json")
    rejected = _load(OUTPUT_DIR / "rejected.json")
    meta = _load(OUTPUT_DIR / "meta.json")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

    pool: list[dict] = []
    for c in load_candidates():
        name = f"{c['key']}.mp4"
        if name in rejected or name in meta or (CHOICES_DIR / c["key"]).exists() or (OUT_DIR / name).exists():
            continue
        try:
            dur = _ts(c["end"]) - _ts(c["start"])
        except Exception:
            rejected[name] = "DURATION_PARSE_ERROR"
            continue
        if not (MIN_DURATION <= dur <= MAX_DURATION):
            rejected[name] = f"DURATION_OUT_OF_RANGE({dur:.2f}s)"
            continue
        pool.append(c)

    random.shuffle(pool)

    lock_llm, lock_rej, lock_cnt = (threading.Lock() for _ in range(3))
    count = 0

    def process(c: dict) -> None:
        nonlocal count
        name = f"{c['key']}.mp4"
        video = OUT_DIR / name

        with lock_cnt:
            if count >= NUM_VIDEOS:
                return

        # 1) o3 过滤
        ok, reason = o3_filter(client, c["caption"])
        with lock_llm:
            llm_results[name] = {"pass": ok, "reason": reason}
            _save(llm_results, OUTPUT_DIR / "llm_filter.json")
        if not ok:
            with lock_rej:
                rejected[name] = f"LLM_REJECT: {reason}"
                _save(rejected, OUTPUT_DIR / "rejected.json")
            return

        with lock_cnt:
            if count >= NUM_VIDEOS:
                return

        # 2) 下载
        if not download_one(c["key"], c["url"], c["start"], c["end"]):
            return

        # 3) 质量检查
        if check_quality(video) == "LOW":
            with lock_rej:
                rejected[name] = "QUALITY_LOW"
                _save(rejected, OUTPUT_DIR / "rejected.json")
            video.unlink(missing_ok=True)
            return

        with lock_cnt:
            if count < NUM_VIDEOS:
                count += 1

    if not pool:
        print("No valid candidates to process.")
        return

    workers = min(NUM_WORKERS, max(1, os.cpu_count() or 1))
    print(f"Processing {len(pool)} candidates with up to {workers} workers ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in tqdm(ex.map(process, pool), total=min(NUM_VIDEOS, len(pool)), desc="Filter+Download"):
            with lock_cnt:
                if count >= NUM_VIDEOS:
                    break

    print(f"Downloaded {count}/{NUM_VIDEOS} videos.")


if __name__ == "__main__":
    main()
