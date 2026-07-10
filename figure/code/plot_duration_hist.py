"""统计 video-cloze/choices 下每个 stem 的 before+GT+after 总时长，生成直方图。

输出：figure/pics/fig_duration_hist.pdf
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CHOICES_DIR = REPO_ROOT / "TempCloze" / "choices"
OUT_DIR = REPO_ROOT / "figure" / "pics"

LABEL_FS = 15
TICK_FS = 14
TEXT_FS = 15
TEXT_WEIGHT = "semibold"


def get_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0.0


def collect_durations() -> list[float]:
    durations = []
    for stem_dir in sorted(CHOICES_DIR.iterdir()):
        if not stem_dir.is_dir():
            continue
        total = 0.0
        valid = True
        for name in ("before.mp4", "GT.mp4", "after.mp4"):
            p = stem_dir / name
            if not p.exists():
                valid = False
                break
            total += get_duration(p)
        if valid and total > 0:
            durations.append(total)
    return durations


def main():
    print("Collecting durations...")
    durations = collect_durations()
    durations = np.array(durations)
    print(f"  n={len(durations)},  mean={durations.mean():.1f}s,  "
          f"median={np.median(durations):.1f}s,  max={durations.max():.1f}s")

    fig, ax = plt.subplots(figsize=(6, 3.8))

    bins = np.arange(0, durations.max() + 5, 5)
    counts, edges, patches = ax.hist(durations, bins=bins,
                                     color="#5B8DB8", edgecolor="white",
                                     linewidth=0.6, alpha=0.92)

    # 按 bin 高度渐变色：低→浅蓝，高→深蓝
    max_count = counts.max()
    cmap = plt.cm.Blues
    for patch, cnt in zip(patches, counts):
        patch.set_facecolor(cmap(0.35 + 0.55 * cnt / max_count))

    ax.set_xlabel("Duration (s)", fontsize=LABEL_FS, style="italic", fontweight=TEXT_WEIGHT)
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.set_xlim(10, 65)
    ax.set_ylim(0, max_count * 1.18)

    # 加粗刻度
    ax.tick_params(axis="both", labelsize=TICK_FS, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(TEXT_WEIGHT)

    # Mean & Median 文字标注
    mean_val   = durations.mean()
    median_val = np.median(durations)
    ax.text(0.97, 0.95,
            f"Mean = {mean_val:.1f} s\nMedian = {median_val:.1f} s",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=TEXT_FS, fontweight=TEXT_WEIGHT, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig_duration_hist.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
