"""从 MiraData 下载视频片段：o3 caption 预筛 → yt-dlp 下载 → 质量检查。

数据源: miradata.csv（已完成 duration 过滤 + video_id 去重 + shuffle）
CSV 字段: video_id, video_url, seconds, timestamp, dense_caption

Pipeline:  CSV → o3 dense_caption 预筛 → 下载 → 质量检查 → 保留/删除
输出:      output/mira/{llm_filter,rejected}.json
"""

import ast
import csv
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ==================== 路径 & 配置 ====================

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent.parent
CSV_PATH = ROOT / "miradata.csv"
OUT_DIR = ROOT.parent / "downloaded"
CHOICES_DIR = PROJECT_ROOT / "choices"
COOKIES_PATH = ROOT.parent / "cookies.txt"
OUTPUT_DIR = PROJECT_ROOT / "output" / "mira"

NUM_VIDEOS = 50
MAX_HEIGHT = 360
NUM_WORKERS = 8

SKIP_ERRORS = (
    "Video unavailable",
    "Private video",
    "HTTP Error 500",
    "not made this video available in your country",
    "This video has been removed",
    "Join this channel to get access",
    "This live event will begin",
    "Premieres in",
)

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


def _parse_timestamp(s: str) -> tuple[str, str] | None:
    """解析 timestamp 字段，如 "['00:06:21.300', '00:07:42.133']"。"""
    try:
        val = ast.literal_eval(s or "")
    except (SyntaxError, ValueError):
        return None
    if isinstance(val, (list, tuple)) and len(val) == 2 and all(isinstance(v, str) for v in val):
        return val[0], val[1]
    return None


def load_candidates() -> list[dict]:
    rows: list[dict] = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            video_id = (r.get("video_id") or "").strip()
            video_url = (r.get("video_url") or "").strip()
            timestamp = (r.get("timestamp") or "").strip()
            caption = (r.get("dense_caption") or "").strip()

            if not (video_id and video_url and timestamp):
                continue

            span = _parse_timestamp(timestamp)
            if span is None:
                continue

            rows.append({
                "key": video_id,
                "video_url": video_url,
                "start": span[0],
                "end": span[1],
                "caption": caption,
            })
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
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode(errors="replace").strip()
        print(f"[DOWNLOAD FAILED] {key} | {url}\n  {msg}", flush=True)
        if any(s in msg for s in SKIP_ERRORS):
            return False
        sys.exit(1)


# ==================== o3 过滤 ====================

O3_PROMPT = """[Temporal Cloze Task Suitability]

You are judging whether the video described by the caption is suitable for a temporal cloze benchmark. In this benchmark, a video is split into three ordered visual segments:
- BEGINNING: the context shown before the missing middle.
- ANSWER: the missing middle segment that a model should choose or predict.
- ENDING: the context shown after the missing middle.

A good example lets a viewer infer a plausible ANSWER from BEGINNING and ENDING through visual temporal and causal continuity. Judge only what can be supported by visible actions or events in the caption. Ignore audio-only information, speech or dialogue content, subtitles, narration, background music, creator commentary, and any purely emotional or aesthetic judgment unless it is visually grounded.

Please consider these quality aspects:
- Scene transition: whether BEGINNING, ANSWER, and ENDING appear to belong to a coherent visual scene or a justifiable scene transition.
- Subject consistency: whether the main person, object, or environment remains consistent across the three segments.
- Action consistency: whether the motion or activity in ANSWER reasonably continues from BEGINNING and leads toward ENDING.
- Logical consistency: whether the full sequence makes temporal and causal sense as a middle-completion example.

PASS only if:
- The caption describes a concrete visual process, action, interaction, transformation, or event with ordered steps.
- The likely ANSWER segment would be visually distinguishable and meaningfully constrained by BEGINNING and ENDING.
- BEGINNING, ANSWER, and ENDING would form one coherent temporal sequence with stable subjects and a reasonable causal or physical progression.

REJECT if any of these apply:
- The caption is mostly static description, scenery, appearance, atmosphere, or a montage without a clear ordered visual action.
- The event depends mainly on dialogue, audio, text, subtitles, narration, or hidden intent rather than visible motion.
- The middle could be almost anything, is too ambiguous, or cannot be inferred from before/after visual context.
- There are abrupt unrelated scene cuts, inconsistent subjects or environments, or disconnected actions that would make BEGINNING, ANSWER, and ENDING incoherent.
- The video only shows repeated motion with no meaningful progress, or a single state with little temporal change.

Caption:
{caption}

Return valid JSON only, using one of these shapes:
{{"pass": true, "reason": "one or two concise sentences naming the decisive quality aspects"}}
{{"pass": false, "reason": "one or two concise sentences naming the decisive quality aspects"}}
"""


def o3_filter(client: OpenAI, caption: str, name: str) -> tuple[bool, str]:
    if not (caption or "").strip():
        return False, "EMPTY"
    try:
        resp = client.chat.completions.create(
            model="openai/o3",
            messages=[
                {"role": "system", "content": "You are a video content classifier. Output valid JSON only."},
                {"role": "user", "content": O3_PROMPT.format(caption=caption)},
            ],
            max_tokens=500,
        )
    except Exception as e:
        print(f"[O3 ERROR] {name} | {e}", flush=True)
        sys.exit(1)
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        return bool(data.get("pass")), data.get("reason", "")
    except json.JSONDecodeError:
        return False, raw


# ==================== Main ====================


def main() -> None:
    print("=== MiraData Download ===")
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
        pool.append(c)

    print(f"Candidates: {len(pool)}")

    if not pool:
        print("No valid candidates to process.")
        return

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
        cached = llm_results.get(name)
        if cached:
            ok, reason = cached.get("pass", False), cached.get("reason", "")
        else:
            ok, reason = o3_filter(client, c["caption"], name)
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

        # 2) yt-dlp 下载
        if not download_one(c["key"], c["video_url"], c["start"], c["end"]):
            with lock_rej:
                rejected[name] = "DOWNLOAD_FAILED"
                _save(rejected, OUTPUT_DIR / "rejected.json")
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

    workers = min(NUM_WORKERS, max(1, os.cpu_count() or 1))
    print(f"Processing with up to {workers} workers ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in tqdm(ex.map(process, pool), total=min(NUM_VIDEOS, len(pool)),
                      desc="MiraData Filter+Download"):
            with lock_cnt:
                if count >= NUM_VIDEOS:
                    break

    print(f"Downloaded {count}/{NUM_VIDEOS} videos.")


if __name__ == "__main__":
    main()
