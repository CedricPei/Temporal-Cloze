"""Gap 检测 + 生成 Temporal Cloze Benchmark: 题干 (before/GT/after) + 干扰项 (S/A/C)

Pipeline:
1. 对 src/downloaded/ 中的视频做 Gap Detection（光流）
2. 通过 → 写入 meta.json (gap_start, gap_end)
3. 失败 → 写入 rejected.json，删除视频
4. 对 meta.json 中有 gap 且尚未生成的条目生成 choices

输出结构:
choices/{stem}/
├── before.mp4 / GT.mp4 / after.mp4
├── C/  Reverse / Shuffle / Loop
├── A/  Early / Late / Wide
└── S/  Rand1 / Rand2 / Rand3

Usage:
  python generate.py              # 默认处理 lvd
  python generate.py tt           # 处理 video-tt
  python generate.py favor        # 处理 FAVOR-Bench
  python generate.py care         # 处理 CareBench
  python generate.py dailyomni    # 处理 DailyOmni
  python generate.py egolife      # 处理 EgoLife
  python generate.py mira         # 处理 MiraData
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
SRC = Path(__file__).parent

PRESET = sys.argv[1] if len(sys.argv) > 1 else "lvd"
VIDEO_DIR = SRC / "downloaded"
OUTPUT_DIR = ROOT / "choices"
META_PATH = ROOT / "output" / PRESET / "meta.json"
REJECTED_PATH = ROOT / "output" / PRESET / "rejected.json"
LLM_FILTER_PATH = ROOT / "output" / PRESET / "llm_filter.json"

# Gap 参数
GAP_LEN_RATIO_MIN = 0.2
GAP_LEN_RATIO_MAX = 0.4
MIDDLE_MARGIN = 0.25
MAG_THRESHOLD = 1.0
GAP_MAX_TRIES = 3
RESIZE_HEIGHT = 256


# ==================== JSON 持久化 ====================

def _load(path: Path) -> dict:
    return json.load(open(path, encoding="utf-8")) if path.exists() else {}


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ==================== Gap Detection ====================

def detect_gap(path: Path) -> dict | None:
    """在视频中间 50% 区域随机取一段 (20–40%)，用光流判断是否有足够运动。
    返回 {"gap_start": float, "gap_end": float} 或 None。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    if fps <= 0 or duration <= 0:
        cap.release()
        return None

    t_lo = MIDDLE_MARGIN * duration
    t_hi = (1 - MIDDLE_MARGIN) * duration
    if t_hi - t_lo < duration * GAP_LEN_RATIO_MIN:
        cap.release()
        return None

    for _ in range(GAP_MAX_TRIES):
        gap_len = random.uniform(
            duration * GAP_LEN_RATIO_MIN,
            min(duration * GAP_LEN_RATIO_MAX, t_hi - t_lo),
        )
        start = random.uniform(t_lo, t_hi - gap_len)
        end = start + gap_len

        start_frame = int(start * fps)
        end_frame = int(end * fps)
        if end_frame <= start_frame + 1:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        prev = None
        magnitudes = []
        resize_wh = None
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if resize_wh is None:
                new_w = int(round(RESIZE_HEIGHT * w / h))
                resize_wh = (new_w, RESIZE_HEIGHT)
            gray = cv2.resize(gray, resize_wh)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                magnitudes.append(float(mag.mean()))
            prev = gray

        if magnitudes and sum(magnitudes) / len(magnitudes) > MAG_THRESHOLD:
            cap.release()
            return {"gap_start": round(start, 2), "gap_end": round(end, 2)}

    cap.release()
    return None


# ==================== ffmpeg 工具 ====================

def ffmpeg(args: list):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args,
                   check=True, stdin=subprocess.DEVNULL)


def cut(src: Path, dst: Path, start: float, end: float):
    ffmpeg([
        "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(src),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an", str(dst)
    ])


def get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    return float(result.stdout.strip())


# ==================== 题干 ====================

def make_stems(src: Path, out: Path, gs: float, ge: float):
    cut(src, out / "before.mp4", 0, gs)
    cut(src, out / "GT.mp4", gs, ge)
    cut(src, out / "after.mp4", ge, 9999)


# ==================== C: Causality ====================

def make_c(src: Path, out: Path, gs: float, ge: float, total: float):
    out.mkdir(exist_ok=True)
    D = ge - gs

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    cut(src, Path(tmp_path), gs, ge)
    ffmpeg(["-i", tmp_path, "-vf", "reverse",
            "-c:v", "libx264", "-crf", "18", "-an", str(out / "Reverse.mp4")])
    Path(tmp_path).unlink()

    seg = D / 3
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a, b, c = td / "a.mp4", td / "b.mp4", td / "c.mp4"
        cut(src, a, gs, gs + seg)
        cut(src, b, gs + seg, gs + 2 * seg)
        cut(src, c, gs + 2 * seg, ge)
        lst = td / "list.txt"
        lst.write_text(f"file '{b}'\nfile '{c}'\nfile '{a}'\n")
        ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst),
                "-c:v", "libx264", "-crf", "18", "-an", str(out / "Shuffle.mp4")])

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mid = td / "mid.mp4"
        cut(src, mid, gs + D / 3, gs + 2 * D / 3)
        lst = td / "list.txt"
        lst.write_text(f"file '{mid}'\nfile '{mid}'\nfile '{mid}'\n")
        ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst),
                "-c:v", "libx264", "-crf", "18", "-an", str(out / "Loop.mp4")])


# ==================== A: Alignment ====================

def make_a(src: Path, out: Path, gs: float, ge: float, total: float):
    out.mkdir(exist_ok=True)
    D = ge - gs
    cut(src, out / "Early.mp4", max(0, gs - 0.5 * D), gs + 0.5 * D)
    cut(src, out / "Late.mp4", ge - 0.5 * D, min(total, ge + 0.5 * D))
    cut(src, out / "Wide.mp4", max(0, gs - 0.5 * D), min(total, ge + 0.5 * D))


# ==================== S: Semantic ====================

def make_s(src: Path, out: Path, gs: float, ge: float, total: float):
    out.mkdir(exist_ok=True)
    min_len = 0.2 * total
    regions = [(0.0, gs), (ge, total)]
    for idx in range(1, 4):
        r_start, r_end = random.choice(regions)
        max_len = r_end - r_start
        length = random.uniform(min_len, max_len)
        start = random.uniform(r_start, r_end - length)
        cut(src, out / f"Rand{idx}.mp4", start, start + length)


# ==================== Main ====================

NUM_WORKERS = min(6, max(1, os.cpu_count() or 1))


def run():
    meta = _load(META_PATH)
    rejected = _load(REJECTED_PATH)
    llm_filter = _load(LLM_FILTER_PATH)
    lock_meta = threading.Lock()
    lock_rej = threading.Lock()

    # 只处理当前 preset 拥有的视频（llm_filter 中 pass=true 或已在 meta 中）
    owned_names = {k for k, v in llm_filter.items()
                   if isinstance(v, dict) and v.get("pass")}
    owned_names |= set(meta.keys())
    videos = sorted([f for f in VIDEO_DIR.iterdir()
                     if f.suffix.lower() == ".mp4" and f.name in owned_names])

    # ---- Phase 1: Gap Detection（多线程） ----
    need_gap = [v for v in videos if v.name not in meta and v.name not in rejected]
    if need_gap:
        def _detect(video: Path) -> None:
            gap = detect_gap(video)
            if gap:
                with lock_meta:
                    meta[video.name] = gap
                    _save(meta, META_PATH)
            else:
                with lock_rej:
                    rejected[video.name] = "GAP_REJECT"
                    _save(rejected, REJECTED_PATH)
                video.unlink(missing_ok=True)

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futs = {ex.submit(_detect, v): v for v in need_gap}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Gap detection"):
                fut.result()

    # ---- Phase 2: Generate choices（meta 中有视频且未生成的） ----
    todo = []
    for name, info in meta.items():
        gs, ge = info.get("gap_start"), info.get("gap_end")
        if gs is None or ge is None:
            continue
        if not (VIDEO_DIR / name).exists():
            continue
        stem = Path(name).stem
        if (OUTPUT_DIR / stem / "GT.mp4").exists():
            continue
        todo.append((name, info))

    if todo:
        def _generate(name: str, info: dict) -> None:
            src = VIDEO_DIR / name
            stem = src.stem
            gs, ge = info["gap_start"], info["gap_end"]
            base = OUTPUT_DIR / stem
            if (base / "GT.mp4").exists():
                return
            base.mkdir(parents=True, exist_ok=True)
            try:
                total = get_duration(src)
                make_stems(src, base, gs, ge)
                make_c(src, base / "C", gs, ge, total)
                make_a(src, base / "A", gs, ge, total)
                make_s(src, base / "S", gs, ge, total)
                src.unlink(missing_ok=True)
            except Exception:
                if base.exists():
                    shutil.rmtree(base)
                raise

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futs = {ex.submit(_generate, n, i): n for n, i in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Generating choices"):
                fut.result()


if __name__ == "__main__":
    run()
