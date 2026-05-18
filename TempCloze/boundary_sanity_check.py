"""Boundary sanity check for the main Temporal Cloze set.

Checks the main-eval 16-frame setting with boundary windows k=1,2,3:
- last k sampled frames of before.mp4 vs first k sampled frames of each candidate
- last k sampled frames of each candidate vs first k sampled frames of after.mp4

For each window, exact=true if any frame in one boundary window exactly
matches any frame in the opposite boundary window after evaluator preprocessing.

The output is one CSV table with rows: S, A, P, Overall.

Usage:
  python TempCloze/boundary_sanity_check.py
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).parent
CHOICES_DIR = ROOT / "choices"
DEFAULT_OUT = ROOT / "boundary_sanity_results" / "main_boundary_sanity_table.csv"

NUM_FRAMES = 16
MAX_HEIGHT = 360
JPEG_QUALITY = 85
WINDOWS = [1, 2, 3]

# P is the progression dimension; files are stored under C/ in the current data.
DIMENSIONS = {
    "S": ["S/Rand1.mp4", "S/Rand2.mp4", "S/Rand3.mp4"],
    "A": ["A/Early.mp4", "A/Late.mp4", "A/Wide.mp4"],
    "P": ["C/Reverse.mp4", "C/Shuffle.mp4", "C/Loop.mp4"],
}


@dataclass(frozen=True)
class BoundaryFrames:
    first: list[np.ndarray]
    last: list[np.ndarray]
    total_frames: int
    sampled_frames: int
    first_indices: list[int]
    last_indices: list[int]


def sample_indices(total_frames: int, num_frames: int) -> list[int]:
    """Match eval.py sample_and_encode exactly."""
    if total_frames <= 2:
        return []
    n = min(num_frames, total_frames - 2)
    return [1 + int((i + 0.5) * (total_frames - 2) / n) for i in range(n)]


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Match evaluator resize and JPEG round-trip before comparing pixels."""
    h, w = frame.shape[:2]
    if h > MAX_HEIGHT:
        scale = MAX_HEIGHT / h
        frame = cv2.resize(frame, (int(w * scale), MAX_HEIGHT))

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return frame
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else frame


def read_boundary_frames(video_path: Path, num_frames: int, max_window: int) -> BoundaryFrames:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_indices(total, num_frames)
    if not indices:
        cap.release()
        return BoundaryFrames([], [], total, 0, [], [])

    first_indices = indices[:max_window]
    last_indices = indices[-max_window:]

    def read_frames(frame_indices: list[int]) -> list[np.ndarray]:
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(normalize_frame(frame))
        return frames

    first = read_frames(first_indices)
    last = read_frames(last_indices)
    cap.release()
    return BoundaryFrames(first, last, total, len(indices), first_indices, last_indices)


def any_exact_match(left: list[np.ndarray], right: list[np.ndarray]) -> bool:
    for left_frame in left:
        for right_frame in right:
            if left_frame.shape == right_frame.shape and np.array_equal(left_frame, right_frame):
                return True
    return False


def load_main_stems() -> list[str]:
    return sorted(
        p.name for p in CHOICES_DIR.iterdir()
        if p.is_dir() and (p / "GT.mp4").exists()
    )


def check_stem(stem: str, num_frames: int) -> list[dict]:
    base = CHOICES_DIR / stem
    max_window = max(WINDOWS)
    before = read_boundary_frames(base / "before.mp4", num_frames, max_window)
    after = read_boundary_frames(base / "after.mp4", num_frames, max_window)

    rows = []
    for dim, distractors in DIMENSIONS.items():
        for candidate_path in ["GT.mp4", *distractors]:
            candidate = read_boundary_frames(base / candidate_path, num_frames, max_window)
            row = {"dimension": dim}
            for k in WINDOWS:
                row[f"start_{k}f_exact"] = any_exact_match(before.last[-k:], candidate.first[:k])
                row[f"end_{k}f_exact"] = any_exact_match(candidate.last[-k:], after.first[:k])
            rows.append(row)
    return rows


def count_rate(count: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{count}/{total} ({100.0 * count / total:.2f}%)"


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["dimension"]].append(row)
        groups["Overall"].append(row)

    ordered_dims = ["S", "A", "P", "Overall"]
    table = []
    for dim in ordered_dims:
        entries = groups[dim]
        total = len(entries)
        row = {
            "dimension": dim,
            "candidate_checks": total,
        }
        for k in WINDOWS:
            start_count = sum(bool(e[f"start_{k}f_exact"]) for e in entries)
            end_count = sum(bool(e[f"end_{k}f_exact"]) for e in entries)
            row[f"start_{k}f_exact"] = count_rate(start_count, total)
            row[f"end_{k}f_exact"] = count_rate(end_count, total)
        table.append(row)
    return table


def write_csv(path: Path, table: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)


def print_table(table: list[dict]) -> None:
    headers = list(table[0].keys())
    widths = {
        h: max(len(h), *(len(str(row[h])) for row in table))
        for h in headers
    }
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in table:
        print(" | ".join(str(row[h]).ljust(widths[h]) for h in headers))


def run(args: argparse.Namespace) -> None:
    stems = load_main_stems()
    print(f"Main set videos: {len(stems)}")
    print(f"Frames per clip: {args.num_frames}")

    all_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(check_stem, stem, args.num_frames): stem for stem in stems}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Checking main set"):
            all_rows.extend(fut.result())

    table = aggregate(all_rows)
    write_csv(args.out, table)
    print()
    print_table(table)
    print(f"\nWrote: {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main-set boundary sanity check.")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
