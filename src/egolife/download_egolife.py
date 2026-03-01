"""EgoLife 视频筛选与下载：o3 caption 预筛 → HF 下载 → 时长过滤 → 质量检查 → 360p 转码。

数据来源: EgoLife_Caption.json（来自 lmms-lab/EgoLife, 每条 30s 片段）
视频下载: HuggingFace datasets CDN

Pipeline:
  1. 从 EgoLife_Caption.json 加载英文 caption（忽略语音/声音相关内容）
  2. o3 对 caption 做 temporal cloze 适用性判断
  3. 通过的视频从 HuggingFace 下载
  4. 时长过滤（12–90s）
  5. 质量检查（bitrate + sharpness）
  6. 转码为 360p
  7. 全部通过 → 保留在 src/downloaded/

输出:
  output/egolife/{llm_filter.json, rejected.json}

Usage:
  python src/egolife/download_egolife.py
"""

import json
import os
import re
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

ROOT = Path(__file__).parent                # src/egolife
SRC = ROOT.parent                           # src
PROJECT_ROOT = SRC.parent                   # T-Cloze

CAPTION_PATH = ROOT / "EgoLife_Caption.json"
DOWNLOAD_CACHE = ROOT / "cache"             # 下载缓存（质量检查前）
DOWNLOADED_DIR = SRC / "downloaded"          # 最终通过的视频
CHOICES_DIR = PROJECT_ROOT / "choices"
OUTPUT_DIR = PROJECT_ROOT / "output" / "egolife"

HF_BASE_URL = "https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main"

NUM_VIDEOS = 400
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


# ==================== Caption 加载 ====================


def _strip_audio_info(text: str) -> str:
    """移除 caption 中与声音/语音/对话相关的描述，只保留视觉动作信息。"""
    lines = text.split(". ")
    visual_lines = []
    audio_keywords = re.compile(
        r"\b(said|says|asked|replied|responded|whispered|shouted|laughed|"
        r"heard|voice|sound|music|song|singing|speech|spoke|told|mentioned|"
        r"exclaimed|murmured|chuckled|giggled)\b",
        re.IGNORECASE,
    )
    for line in lines:
        if not audio_keywords.search(line):
            visual_lines.append(line)
    return ". ".join(visual_lines).strip()


def load_candidates() -> list[dict]:
    """从 EgoLife_Caption.json 读取英文 caption 条目。"""
    with open(CAPTION_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    entries: list[dict] = []
    seen_videos: set[str] = set()
    for item in raw:
        vid_id = item.get("id", "")
        video_path = item.get("video", "")
        if not video_path or video_path in seen_videos:
            continue
        seen_videos.add(video_path)

        caption = ""
        for conv in item.get("conversations", []):
            if conv.get("from") == "gpt":
                caption = conv.get("value", "")
                break

        caption_visual = _strip_audio_info(caption)

        video_name = Path(video_path).name  # e.g. DAY1_A6_SHURE_14000000.mp4
        entries.append({
            "id": vid_id,
            "video_name": video_name,
            "video_hf_path": video_path,  # data/EgoLife/A6_SHURE/DAY1/...
            "caption": caption_visual,
        })

    return entries


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


# ==================== 时长 & 下载 & 转码 ====================


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


def download_video(hf_path: str, out_path: Path) -> bool:
    """从 HuggingFace 下载视频到 out_path。"""
    if out_path.exists():
        return True
    real_path = hf_path.removeprefix("data/EgoLife/")
    url = f"{HF_BASE_URL}/{requests.utils.quote(real_path, safe='/')}"
    try:
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"[HF DOWNLOAD FAILED] {hf_path} | {e}", flush=True)
        out_path.unlink(missing_ok=True)
        return False


CIRCLE_MASK = (
    "geq="
    "lum='if(gt(sqrt((X-179)*(X-179)+(Y-180)*(Y-180)),178),16,lum(X,Y))':"
    "cb='if(gt(sqrt((X/2-89.5)*(X/2-89.5)+(Y/2-90)*(Y/2-90)),89),128,cb(X,Y))':"
    "cr='if(gt(sqrt((X/2-89.5)*(X/2-89.5)+(Y/2-90)*(Y/2-90)),89),128,cr(X,Y))'"
)


def transcode_to_360p(src: Path, dst: Path) -> bool:
    """转码为 360p 并用圆形遮罩覆盖鱼眼圆外的时间戳水印。"""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"scale=-2:360,{CIRCLE_MASK}",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-an",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# ==================== o3 过滤 ====================

O3_PROMPT = """[Temporal Cloze Task Suitability]
Can this video be used for a "temporal cloze" task?
(Given the beginning and end of a video, predict what happens in the middle gap)

IMPORTANT: Ignore any audio/speech/dialogue information. Judge ONLY based on visual actions described.

✓ PASS: Video has clear causal/temporal continuity in visual actions where the middle can be inferred from before & after
✗ REJECT: Visual actions lack temporal progression, or the middle segment cannot be meaningfully predicted from visual context alone

Caption (visual actions only): {caption}

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
    print("=== EgoLife Download & Filter ===")

    DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    DOWNLOADED_DIR.mkdir(parents=True, exist_ok=True)

    all_entries = load_candidates()
    print(f"Loaded {len(all_entries)} unique video entries from {CAPTION_PATH.name}")

    llm_results = _load(OUTPUT_DIR / "llm_filter.json")
    rejected = _load(OUTPUT_DIR / "rejected.json")
    meta = _load(OUTPUT_DIR / "meta.json")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 构建候选池：跳过已处理 / 已下载 / 已生成 choices 的
    pool: list[dict] = []
    for entry in all_entries:
        name = entry["video_name"]
        stem = Path(name).stem
        if name in rejected or name in meta or (CHOICES_DIR / stem).exists() or (DOWNLOADED_DIR / name).exists():
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

        with lock_cnt:
            if count >= NUM_VIDEOS:
                return

        # 1) o3 过滤（基于去除声音信息的 caption）
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

        # 2) 从 HuggingFace 下载
        cache_path = DOWNLOAD_CACHE / name
        if not cache_path.exists() and not download_video(entry["video_hf_path"], cache_path):
            with lock_rej:
                rejected[name] = "DOWNLOAD_FAILED"
                _save(rejected, OUTPUT_DIR / "rejected.json")
            return

        # 3) 时长检查
        duration = get_video_duration(cache_path)
        if not (MIN_DURATION <= duration <= MAX_DURATION):
            with lock_rej:
                rejected[name] = f"DURATION_OUT_OF_RANGE({duration:.2f}s)"
                _save(rejected, OUTPUT_DIR / "rejected.json")
            cache_path.unlink(missing_ok=True)
            return

        # 4) 质量检查
        if check_quality(cache_path) == "LOW":
            with lock_rej:
                rejected[name] = "QUALITY_LOW"
                _save(rejected, OUTPUT_DIR / "rejected.json")
            cache_path.unlink(missing_ok=True)
            return

        # 5) 转码为 360p 并放到 downloaded
        dst = DOWNLOADED_DIR / name
        if not transcode_to_360p(cache_path, dst):
            with lock_rej:
                rejected[name] = "TRANSCODE_FAILED"
                _save(rejected, OUTPUT_DIR / "rejected.json")
            cache_path.unlink(missing_ok=True)
            return

        cache_path.unlink(missing_ok=True)

        with lock_cnt:
            if count < NUM_VIDEOS:
                count += 1

    workers = min(NUM_WORKERS, max(1, os.cpu_count() or 1))
    print(f"Processing {len(pool)} candidates with up to {workers} workers ...")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in tqdm(ex.map(process, pool), total=min(NUM_VIDEOS, len(pool)),
                      desc="EgoLife Filter+Download"):
            with lock_cnt:
                if count >= NUM_VIDEOS:
                    break

    print(f"Done. Passed: {count}/{NUM_VIDEOS}, Rejected: {len(rejected)}, LLM entries: {len(llm_results)}")


if __name__ == "__main__":
    main()
