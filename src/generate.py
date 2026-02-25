"""生成 Temporal Cloze Benchmark: 题干 (before/GT/after) + 干扰项 (S/A/C)

输出结构:
choices/{stem}/
├── before.mp4
├── GT.mp4
├── after.mp4
├── C/  Reverse / Shuffle / Loop    (Causality)
├── A/  Early / Late / Wide         (Alignment)
└── S/  Rand1 / Rand2 / Rand3      (Semantic)

Usage:
  python generate.py          # 默认处理 lvd
  python generate.py tt       # 处理 video-tt
"""

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).parent.parent
SRC = Path(__file__).parent

PRESET = sys.argv[1] if len(sys.argv) > 1 else "lvd"
VIDEO_DIR = SRC / "downloaded"
GAP_JSON = ROOT / "output" / f"{PRESET}_meta.json"
OUTPUT_DIR = ROOT / "choices"


def ffmpeg(args: list):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, check=True)


def cut(src: Path, dst: Path, start: float, end: float):
    ffmpeg([
        "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(src),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an", str(dst)
    ])


def get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


# ==================== 题干 ====================

def make_stems(src: Path, out: Path, gs: float, ge: float):
    """生成 before / GT / after"""
    cut(src, out / "before.mp4", 0, gs)
    cut(src, out / "GT.mp4", gs, ge)
    cut(src, out / "after.mp4", ge, 9999)


# ==================== C: Causality ====================

def make_c(src: Path, out: Path, gs: float, ge: float, total: float):
    out.mkdir(exist_ok=True)
    D = ge - gs

    # Reverse
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    cut(src, Path(tmp_path), gs, ge)
    ffmpeg(["-i", tmp_path, "-vf", "reverse",
            "-c:v", "libx264", "-crf", "18", "-an", str(out / "Reverse.mp4")])
    Path(tmp_path).unlink()

    # Shuffle (B-C-A)
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

    # Loop (middle 33% × 3)
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

def run():
    with open(GAP_JSON, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    kept = {k: v for k, v in metadata.items() if v.get("keep")}

    # 只对：keep、有 gap、原视频存在、且尚未生成 GT.mp4 的条目进行处理
    todo_items = []
    for name, info in kept.items():
        gs, ge = info.get("gap_start"), info.get("gap_end")
        if gs is None or ge is None:
            continue
        if not (VIDEO_DIR / name).exists():
            continue
        stem = Path(name).stem
        if (OUTPUT_DIR / stem / "GT.mp4").exists():
            continue
        todo_items.append((name, info))

    for name, info in tqdm(todo_items, desc="Generating choices", total=len(todo_items)):
        src = VIDEO_DIR / name
        stem = src.stem
        gs, ge = info["gap_start"], info["gap_end"]
        base = OUTPUT_DIR / stem
        if (base / "GT.mp4").exists():
            continue
        base.mkdir(parents=True, exist_ok=True)
        try:
            total = get_duration(src)
            make_stems(src, base, gs, ge)
            make_c(src, base / "C", gs, ge, total)
            make_a(src, base / "A", gs, ge, total)
            make_s(src, base / "S", gs, ge, total)
            if src.exists():
                src.unlink()
        except Exception:
            if base.exists():
                shutil.rmtree(base)
            raise
    

if __name__ == "__main__":
    run()
