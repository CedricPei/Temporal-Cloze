"""DailyOmni 本地视频筛选：时长过滤 → 质量检查 → o3 预筛。

视频已经预先下载在项目内的 `src/Videos` 目录中（每个视频一个子目录，
包含 `<video_id>_video.mp4` 和 `video_consistent_captions.txt`）。

本脚本不做下载，只对本地视频做过滤，并写入：
  output/dailyomni/{llm_filter.json, rejected.json}

Usage:
  python src/download_dailyomni.py
"""

import json
import os
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

# DailyOmni 的视频目录（固定为项目内的 src/Videos）
VIDEO_ROOT = ROOT / "Videos"
CHOICES_DIR = PROJECT_ROOT / "choices"

OUTPUT_DIR = PROJECT_ROOT / "output" / "dailyomni"

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


# ==================== 时长 ====================


def get_video_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ==================== 数据加载（从本地 Videos 目录） ====================


def load_candidates() -> list[dict]:
    """扫描 VIDEO_ROOT 下的子目录，收集 (video_path, caption)。"""
    entries: list[dict] = []
    if not VIDEO_ROOT.exists():
        print(f"[WARN] VIDEO_ROOT 不存在: {VIDEO_ROOT}（期望为项目内的 src/Videos）")
        return entries

    for d in sorted(VIDEO_ROOT.iterdir()):
        if not d.is_dir():
            continue

        # 约定: 目录内的 *_video.mp4 为主视频；若没有则取第一个 mp4
        mp4s = sorted(p for p in d.glob("*.mp4") if p.is_file())
        if not mp4s:
            continue

        video = None
        for p in mp4s:
            if p.name.endswith("_video.mp4"):
                video = p
                break
        if video is None:
            video = mp4s[0]

        cap_path = d / "video_consistent_captions.txt"
        caption = ""
        if cap_path.exists():
            try:
                caption = cap_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                caption = ""

        entries.append(
            {
                "id": d.name,
                "video_name": video.name,   # 作为 JSON 中的 key
                "video_path": str(video),
                "caption": caption,
            }
        )

    return entries


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


def o3_filter(client: OpenAI, caption: str) -> tuple[bool, str]:
    if not (caption or "").strip():
        return False, "EMPTY"
    resp = client.chat.completions.create(
        model="openai/o3",
        messages=[
            {
                "role": "system",
                "content": "You are a video content classifier. Output valid JSON only.",
            },
            {
                "role": "user",
                "content": O3_PROMPT.format(caption=caption),
            },
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
    print("=== DailyOmni Local Filter ===")

    all_entries = load_candidates()
    print(f"Found {len(all_entries)} video entries under {VIDEO_ROOT}")

    llm_results = _load(OUTPUT_DIR / "llm_filter.json")
    rejected = _load(OUTPUT_DIR / "rejected.json")
    meta = _load(OUTPUT_DIR / "meta.json")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 构建候选池：跳过已处理 / 已生成 choices 的
    pool: list[dict] = []
    for entry in all_entries:
        name = entry["video_name"]
        stem = Path(name).stem
        if name in rejected or name in meta or (CHOICES_DIR / stem).exists():
            continue
        pool.append(entry)

    if not pool:
        print("No valid candidates to process.")
        return

    lock_llm, lock_rej = (threading.Lock() for _ in range(2))

    def process(entry: dict) -> None:
        name = entry["video_name"]
        caption = entry.get("caption", "")
        video = Path(entry["video_path"])

        # 1) 文件存在性检查
        if not video.exists():
            with lock_rej:
                rejected[name] = "VIDEO_NOT_FOUND"
                _save(rejected, OUTPUT_DIR / "rejected.json")
            return

        # 2) 时长检查
        duration = get_video_duration(video)
        if not (MIN_DURATION <= duration <= MAX_DURATION):
            with lock_rej:
                rejected[name] = f"DURATION_OUT_OF_RANGE({duration:.2f}s)"
                _save(rejected, OUTPUT_DIR / "rejected.json")
            return

        # 3) 质量检查
        if check_quality(video) == "LOW":
            with lock_rej:
                rejected[name] = "QUALITY_LOW"
                _save(rejected, OUTPUT_DIR / "rejected.json")
            return

        # 4) o3 过滤（按 caption）
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

    workers = min(NUM_WORKERS, max(1, os.cpu_count() or 1))
    print(f"Processing {len(pool)} candidates with up to {workers} workers ...")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in tqdm(ex.map(process, pool), total=len(pool), desc="DailyOmni Filter"):
            pass

    print(f"Done. Rejected: {len(rejected)}, LLM entries: {len(llm_results)}")


if __name__ == "__main__":
    main()
