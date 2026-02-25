"""Download random clips from hdvg_300k_720p.csv with LLM pre-filtering.

Requirements:
- yt-dlp (pip install yt-dlp)
- ffmpeg (in PATH)
- openai + dotenv (for LLM pre-filter)

Behavior:
- 从 CSV 读取候选 clip（由 key / video_id / orig_span 指定）。
- 先用 LLM 对 refined_caption 做一次过滤，只保留“适合 Temporal Cloze”的样本。
- 从通过 LLM 的样本中随机采样 NUM_VIDEOS 个，调用 yt-dlp 下载对应时间段到 src/downloaded/<key>.mp4。
- LLM 过滤结果会写入 output/lvd_llm_filter.json。
"""

import ast
import csv
import json
import logging
import os
import random
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
CSV_PATH = ROOT / "hdvg_300k_720p.csv"
OUT_DIR = ROOT / "downloaded"
CHOICES_DIR = PROJECT_ROOT / "choices"
COOKIES_PATH = ROOT / "cookies.txt"
LLM_LOG_PATH = PROJECT_ROOT / "output" / "lvd_llm_filter.json"
REJECT_LOG_PATH = PROJECT_ROOT / "output" / "lvd_rejected.json"
META_JSON = PROJECT_ROOT / "output" / "lvd_meta.json"

# 下载数量 & 时长 / 分辨率控制
NUM_VIDEOS = 50              # 每次希望下载的视频数量
MIN_DURATION = 12.0          # 片段至少 12 秒
MAX_DURATION = 90.0          # 片段不超过 90 秒（从源头控制 T_RANGE）
MAX_HEIGHT = 360             # 优先使用 <=360p 的流（文件更小）
NUM_WORKERS = 8              # 多线程 worker 数量

load_dotenv(PROJECT_ROOT / ".env")
log = logging.getLogger(__name__)


def _parse_span(span_str: str) -> tuple[str, str] | None:
    """Parse orig_span like \"['00:01:35.750', '00:02:58.750']\" -> (start, end)."""
    if not span_str:
        return None
    try:
        val = ast.literal_eval(span_str)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(val, (list, tuple)) or len(val) != 2:
        return None
    start, end = val
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    return start, end


def _time_to_seconds(t: str) -> float:
    """Convert 'HH:MM:SS.xxx' to seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_candidates() -> list[dict]:
    """
    CSV schema (given by user):
      - key:        id of the video clip
      - video_id:   id of the YouTube video
      - url:        url of the video that the clip comes from
      - orig_span:  ['HH:MM:SS.xxx', 'HH:MM:SS.xxx'] for the clip in the original video
      - refined_caption: caption text for LLM filtering
    """
    candidates: list[dict] = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("key") or "").strip()
            vid = row.get("video_id")
            url = row.get("url")
            span_str = row.get("orig_span")
            refined_caption = (row.get("refined_caption") or "").strip()
            if not key or not vid or not url or not span_str:
                continue

            span = _parse_span(span_str)
            if not span:
                continue
            start_str, end_str = span
            candidates.append(
                {
                    "key": key,
                    "video_id": vid,
                    "url": url,
                    "start": start_str,
                    "end": end_str,
                    "refined_caption": refined_caption,
                }
            )
    return candidates


def download_one(key: str, url: str, start: str, end: str) -> tuple[str, bool]:
    """Download one clip [start, end] and save as <key>.mp4 using yt-dlp.
    
    Returns: (key, success)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{key}.mp4"

    cmd = [
        "yt-dlp",
        "--js-runtimes",
        "node",
        "--remote-components",
        "ejs:github",
        "--merge-output-format",
        "mp4",
        "-f",
        f"bv*[height<={MAX_HEIGHT}]/bv[height<={MAX_HEIGHT}]",
        "-o",
        str(out_path),
        "--download-sections",
        f"*{start}-{end}",
    ]
    if COOKIES_PATH.exists():
        cmd += ["--cookies", str(COOKIES_PATH)]
    cmd.append(url)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return key, True
    except subprocess.CalledProcessError as e:
        return key, False


def main() -> None:
    candidates = load_candidates()

    # ---------- LLM 预过滤 ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LLM_LOG_PATH.exists():
        with open(LLM_LOG_PATH, "r", encoding="utf-8") as f:
            llm_results: dict = json.load(f)
    else:
        llm_results = {}

    if REJECT_LOG_PATH.exists():
        with open(REJECT_LOG_PATH, "r", encoding="utf-8") as f:
            rejected: dict = json.load(f)
    else:
        rejected = {}

    kept_names: set[str] = set()
    if META_JSON.exists():
        with open(META_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for name, info in meta.items():
            try:
                if info.get("keep"):
                    # meta 的 key 是原始文件名，例如 '<key>.mp4'
                    kept_names.add(name)
            except AttributeError:
                continue

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    LLM_PROMPT = """[Temporal Cloze Task Suitability]
Can this video be used for a \"temporal cloze\" task?
(Given the beginning and end of a video, predict what happens in the middle gap)

✓ PASS: Video has causal/temporal continuity where the middle can be inferred from before & after
- Cooking process (ingredients → [gap] → finished dish)
- Object transformation (intact → [gap] → broken/changed)
- Goal-directed actions (start state → [gap] → end state)

✗ REJECT: Middle segment cannot be meaningfully predicted
- Random/unrelated scenes
- Repetitive loops (same action throughout)
- Static scenes with no progression

Caption: {caption}

JSON: {{"pass": true/false, "reason": "one or two brief sentences"}}
"""

    def llm_filter_one(caption: str) -> tuple[bool, str]:
        caption = (caption or "").strip()
        if not caption:
            return False, "EMPTY"
        resp = client.chat.completions.create(
            model="openai/o3",
            messages=[
                {"role": "system", "content": "You are a video content classifier. Output valid JSON only."},
                {"role": "user", "content": LLM_PROMPT.format(caption=caption)},
            ],
            max_tokens=500,
        )
        content = resp.choices[0].message.content
        if not content:
            return False, "NO_RESPONSE"
        try:
            data = json.loads(content.strip())
            return bool(data.get("pass")), data.get("reason", "")
        except json.JSONDecodeError:
            return False, content


    pool: list[dict] = []
    for c in candidates:
        name = f"{c['key']}.mp4"
        if name in rejected:
            continue
        if (CHOICES_DIR / c["key"]).exists():
            continue
        if (OUT_DIR / name).exists():
            continue
        if name in kept_names:
            continue
        try:
            dur = _time_to_seconds(c["end"]) - _time_to_seconds(c["start"])
        except Exception:
            rejected[name] = "DURATION_PARSE_ERROR"
            continue
        if not (MIN_DURATION <= dur <= MAX_DURATION):
            rejected[name] = f"DURATION_OUT_OF_RANGE({dur:.2f}s)"
            continue
        pool.append(c)

    # 随机打乱，保证整体还是随机采样
    random.shuffle(pool)

    # 多线程：每个线程内部串行执行「LLM 过滤 → （通过则）下载」
    llm_lock = threading.Lock()
    rej_lock = threading.Lock()
    counter_lock = threading.Lock()
    downloaded_count = 0

    def process_candidate(c: dict) -> None:
        nonlocal downloaded_count

        name = f"{c['key']}.mp4"

        # 如果已经下载够了，就直接退出
        with counter_lock:
            if downloaded_count >= NUM_VIDEOS:
                return

        # 1) LLM 过滤
        ok, reason = llm_filter_one(c.get("refined_caption", ""))
        with llm_lock:
            llm_results[name] = {"pass": ok, "reason": reason}
            with open(LLM_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(llm_results, f, indent=2, ensure_ascii=False)
        if not ok:
            with rej_lock:
                rejected[name] = f"LLM_REJECT: {reason}"
                with open(REJECT_LOG_PATH, "w", encoding="utf-8") as f:
                    json.dump(rejected, f, indent=2, ensure_ascii=False)
            return

        # 2) 通过过滤后，再检查数量是否已满
        with counter_lock:
            if downloaded_count >= NUM_VIDEOS:
                return

        # 3) 下载
        key, success = download_one(c["key"], c["url"], c["start"], c["end"])
        if success:
            with counter_lock:
                if downloaded_count < NUM_VIDEOS:
                    downloaded_count += 1

    max_workers = min(NUM_WORKERS, max(1, os.cpu_count() or 1))
    if not pool:
        print("No valid candidates to process.")
        return

    print(f"Processing candidates with up to {max_workers} workers...")
    # 进度条按目标下载数量 NUM_VIDEOS 显示（而不是候选总数），更符合直觉
    target_total = min(NUM_VIDEOS, len(pool))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(
            executor.map(process_candidate, pool),
            total=target_total,
            desc="Filter+Download",
        ):
            with counter_lock:
                if downloaded_count >= NUM_VIDEOS:
                    break

    print(f"Downloaded {downloaded_count}/{NUM_VIDEOS} videos.")


if __name__ == "__main__":
    main()

