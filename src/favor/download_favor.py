"""从 FAVOR-Bench 下载视频：本地 CSV 元数据 → o3 预筛 → HF 下载 → 质量检查。

元数据: src/favor_video_perspective.csv (1,776 rows, 从 HF dataset subset 导出)
  video_name / caption / camera_motion / subject_attributes / motion_list / questions

Pipeline:  CSV → o3 caption 预筛 → HF 视频下载 → 时长过滤 → 质量检查
输出:      output/favor/{llm_filter.json, rejected.json}
"""

import csv
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import requests
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ==================== 路径 & 配置 ====================

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
CSV_PATH = ROOT / "favor_video_perspective.csv"
OUT_DIR = ROOT / "downloaded"
CHOICES_DIR = PROJECT_ROOT / "choices"
OUTPUT_DIR = PROJECT_ROOT / "output" / "favor"

HF_VIDEO_URL = "https://huggingface.co/datasets/zl2048/FAVOR/resolve/main/videos/FAVOR-Bench"

NUM_VIDEOS = 500
MIN_DURATION = 12.0
MAX_DURATION = 90.0
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


# ==================== CSV 加载 ====================


def load_candidates() -> list[dict]:
    rows: list[dict] = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = (r.get("video_name") or "").strip()
            caption = (r.get("caption") or "").strip()
            if name:
                rows.append({"video_name": name, "caption": caption})
    return rows


# ==================== 视频下载 ====================


def download_video(name: str) -> bool:
    """从 HuggingFace 直接下载视频文件。"""
    out_path = OUT_DIR / name
    if out_path.exists():
        return True
    url = f"{HF_VIDEO_URL}/{requests.utils.quote(name)}"
    try:
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"[HF DOWNLOAD FAILED] {name} | {e}", flush=True)
        out_path.unlink(missing_ok=True)
        return False


def get_video_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


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
    print("=== FAVOR-Bench Download ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 从本地 CSV 读取元数据
    all_entries = load_candidates()
    print(f"Loaded {len(all_entries)} videos from {CSV_PATH.name}")

    # 2) 加载已有状态
    llm_results = _load(OUTPUT_DIR / "llm_filter.json")
    rejected = _load(OUTPUT_DIR / "rejected.json")
    meta = _load(OUTPUT_DIR / "meta.json")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

    # 3) 构建候选池：跳过已处理 / 已下载 / 已生成 choices 的
    pool = []
    for entry in all_entries:
        name = entry["video_name"]
        if name in rejected or name in meta or (CHOICES_DIR / Path(name).stem).exists() or (OUT_DIR / name).exists():
            continue
        pool.append(entry)

    if not pool:
        print("No valid candidates to process.")
        return

    lock_llm, lock_rej, lock_cnt = (threading.Lock() for _ in range(3))
    count = 0

    def process(entry: dict) -> None:
        nonlocal count
        name = entry["video_name"]
        caption = entry.get("caption", "")
        video = OUT_DIR / name

        with lock_cnt:
            if count >= NUM_VIDEOS:
                return

        # 1) o3 过滤（用 CSV 里的 caption）
        cached = llm_results.get(name)
        if cached:
            ok, reason = cached.get("pass", False), cached.get("reason", "")
        else:
            ok, reason = o3_filter(client, caption)
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

        # 2) 从 HF 下载视频
        if not video.exists() and not download_video(name):
            with lock_rej:
                rejected[name] = "DOWNLOAD_FAILED"
                _save(rejected, OUTPUT_DIR / "rejected.json")
            return

        # 3) 时长检查
        duration = get_video_duration(video)
        if not (MIN_DURATION <= duration <= MAX_DURATION):
            with lock_rej:
                rejected[name] = f"DURATION_OUT_OF_RANGE({duration:.2f}s)"
                _save(rejected, OUTPUT_DIR / "rejected.json")
            video.unlink(missing_ok=True)
            return

        # 4) 质量检查
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
    print(f"Processing {len(pool)} candidates with up to {workers} workers ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in tqdm(ex.map(process, pool), total=min(NUM_VIDEOS, len(pool)),
                      desc="FAVOR Filter+Download"):
            with lock_cnt:
                if count >= NUM_VIDEOS:
                    break

    print(f"Downloaded {count}/{NUM_VIDEOS} videos.")


if __name__ == "__main__":
    main()
