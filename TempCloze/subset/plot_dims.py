"""按「2×2 子图」画维度对比图。

子图布局（固定）：
  Gemini 2.5 Pro  |  Seed-1.6
  Qwen3.5-397B    |  Qwen3-VL-8B

输出：plots/
  fig_frames_dims.png          帧数 vs 准确率
  fig_passk_dims.png           pass@k
  perm_consistency_table.csv   排列一致性统计表

Usage:
  python plot_dims.py
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = Path(__file__).parent / "eval_results"
PLOTS_DIR   = Path(__file__).parent / "plots"
DIMS = ["S", "A", "C"]

# ── 模型列表 & 子图顺序 ───────────────────────────────────────────────────
FRAMES_TAGS = [
    "doubao-seed-1-6-251015",
    "gemini-2.5-pro",
    "qwen3.5-397b-a17b",
    "qwen3-vl-8b-instruct",
]
PASSK_TAGS = FRAMES_TAGS
PERM_TAGS  = FRAMES_TAGS

# 子图布局顺序：左上→右上→左下→右下
PLOT_ORDER = [
    "gemini-2.5-pro",
    "doubao-seed-1-6-251015",
    "qwen3.5-397b-a17b",
    "qwen3-vl-8b-instruct",
]

MODEL_LABEL = {
    "doubao-seed-1-6-251015": "Seed1.6",
    "gemini-2.5-pro":         "Gemini2.5-Pro",
    "qwen3.5-397b-a17b":      "Qwen3.5-397B-A17B",
    "qwen3-vl-8b-instruct":   "Qwen3VL-8B-I",
}

# ── 维度编码：统一颜色 + 统一虚线，用标记区分 ────────────────────────────
DIM_COLOR  = {"S": "#4A90D9", "A": "#E07B39", "C": "#56A76B"}
DIM_MARKER = {"S": "o",       "A": "s",        "C": "^"}
DIM_LABEL  = {"S": "Semantic", "A": "Alignment", "C": "Progression"}
ALL_PERMS  = ["A", "B", "C", "D"]


# ── 通用工具 ──────────────────────────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n or n == 0 or c == 0:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pct_fmt(y, _):
    return f"{y*100:.0f}"


def _auto_ylim(ax, pad: float = 0.12, bottom_min: float = 0.0):
    """在已绘制数据基础上，上下各留 pad 比例的余量，底部不低于 bottom_min。"""
    ax.autoscale(axis="y", tight=True)
    ymin, ymax = ax.get_ylim()
    rng = ymax - ymin if ymax > ymin else 0.1
    ax.set_ylim(max(bottom_min, ymin - rng * pad), ymax + rng * pad)


LEGEND_FS   = 16   # 图例字号
LABEL_FS    = 15   # 轴标签字号
TITLE_FS    = 18   # 子图标题字号
TICK_FS     = 14   # 刻度字号
TEXT_WEIGHT = "semibold"
TITLE_WEIGHT = "bold"


def _build_legend_handles(include_overall: bool = False):
    handles = [
        mlines.Line2D([], [], color=DIM_COLOR[d], linewidth=2.2,
                      marker=DIM_MARKER[d], linestyle="--",
                      markersize=8, label=DIM_LABEL[d])
        for d in DIMS
    ]
    if include_overall:
        handles.append(
            mlines.Line2D([], [], color="red", linewidth=2.4,
                          linestyle="-", marker="D", markersize=6.5, label="Overall")
        )
    return handles


def _attach_legend(fig, xlabel, include_overall=False):
    handles = _build_legend_handles(include_overall)
    legend = fig.legend(handles=handles, loc="upper center", ncol=4,
                        bbox_to_anchor=(0.5, 1.01), fontsize=LEGEND_FS,
                        framealpha=0.95, handlelength=2.9, columnspacing=1.8,
                        handletextpad=0.7, borderpad=0.6)
    for txt in legend.get_texts():
        txt.set_fontweight(TEXT_WEIGHT)
    for ax in fig.axes:
        ax.set_xlabel(xlabel, fontsize=LABEL_FS, style="italic", fontweight=TEXT_WEIGHT)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight(TEXT_WEIGHT)
    fig.tight_layout()
    fig.subplots_adjust(top=0.87, hspace=0.38, wspace=0.18)


def _draw_subplots(axes_flat, plot_order, all_data, xs, x_fn, ylabel):
    """在 4 个子图里各画一个模型的 S/A/C 三条虚线（统一颜色）。"""
    for ax, tag in zip(axes_flat, plot_order):
        if tag not in all_data:
            ax.set_visible(False)
            continue
        data = all_data[tag]
        for dim in DIMS:
            ys, x_plot = [], []
            for x in xs:
                entries = x_fn(data, dim, x)
                if not entries:
                    continue
                acc = sum(1 for e in entries if e["correct"]) / len(entries)
                x_plot.append(x)
                ys.append(acc)
            if x_plot:
                ax.plot(x_plot, ys,
                        color=DIM_COLOR[dim],
                        marker=DIM_MARKER[dim],
                        linestyle="--",
                        linewidth=1.8, markersize=6)
        ax.set_title(MODEL_LABEL[tag], fontsize=TITLE_FS, pad=6, fontweight=TITLE_WEIGHT)
        ax.set_ylabel(ylabel, fontsize=LABEL_FS, style="italic", fontweight=TEXT_WEIGHT)
        ax.set_xticks(xs)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
        ax.grid(True, linestyle=":", alpha=0.5)


# ==================== Frames ====================

def plot_frames():
    all_data, frame_counts = {}, set()
    for tag in FRAMES_TAGS:
        p = RESULTS_DIR / f"frames_{tag}.json"
        if not p.exists():
            print(f"[skip] {p.name} not found")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        all_data[tag] = data
        frame_counts.update(v["num_frames"] for v in data.values())

    if not all_data:
        print("No frames data.")
        return
    xs = sorted(frame_counts)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes_flat = axes.flatten()

    def x_fn(data, dim, nf):
        return [v for k, v in data.items()
                if k.rsplit("|", 2)[1] == dim and v["num_frames"] == nf]

    _draw_subplots(axes_flat, PLOT_ORDER, all_data, xs, x_fn, ylabel="Accuracy (%)")

    # Overall 折线（红色实线）
    for ax, tag in zip(axes_flat, PLOT_ORDER):
        if tag not in all_data:
            continue
        data = all_data[tag]
        x_ov, y_ov = [], []
        for nf in xs:
            entries = [v for v in data.values() if v["num_frames"] == nf]
            if entries:
                x_ov.append(nf)
                y_ov.append(sum(1 for e in entries if e["correct"]) / len(entries))
        if x_ov:
            ax.plot(x_ov, y_ov, color="red", linewidth=2.2,
                    linestyle="-", marker="D", markersize=5, zorder=5)
        _auto_ylim(ax)

    for ax in axes_flat[len(PLOT_ORDER):]:
        ax.set_visible(False)

    _attach_legend(fig, xlabel=r"Number of Frames", include_overall=True)

    out = PLOTS_DIR / "frame_count.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ==================== Pass@k ====================

def plot_passk():
    all_data, max_k = {}, 0
    for tag in PASSK_TAGS:
        p = RESULTS_DIR / f"passk_{tag}.json"
        if not p.exists():
            print(f"[skip] {p.name} not found")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        all_data[tag] = data
        groups = defaultdict(list)
        for key, v in data.items():
            groups[key.rsplit("|", 1)[0]].append(v)
        if groups:
            max_k = max(max_k, max(len(g) for g in groups.values()))

    if not all_data or max_k == 0:
        print("No pass@k data.")
        return
    xs = list(range(1, max_k + 1))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes_flat = axes.flatten()

    for ax, tag in zip(axes_flat, PLOT_ORDER):
        if tag not in all_data:
            ax.set_visible(False)
            continue
        data = all_data[tag]

        # S / A / C 虚线
        for dim in DIMS:
            ys, x_plot = [], []
            for k in xs:
                groups = defaultdict(list)
                for key, v in data.items():
                    if key.rsplit("|", 2)[1] == dim:
                        groups[key.rsplit("|", 1)[0]].append(v)
                scores = [pass_at_k(len(g), sum(e["correct"] for e in g), k)
                          for g in groups.values() if len(g) >= k]
                if not scores:
                    continue
                x_plot.append(k)
                ys.append(sum(scores) / len(scores))
            if x_plot:
                ax.plot(x_plot, ys,
                        color=DIM_COLOR[dim], marker=DIM_MARKER[dim],
                        linestyle="--", linewidth=1.8, markersize=6)

        # Overall 折线（红色实线）
        x_ov, y_ov = [], []
        for k in xs:
            groups = defaultdict(list)
            for key, v in data.items():
                groups[key.rsplit("|", 1)[0]].append(v)
            scores = [pass_at_k(len(g), sum(e["correct"] for e in g), k)
                      for g in groups.values() if len(g) >= k]
            if scores:
                x_ov.append(k)
                y_ov.append(sum(scores) / len(scores))
        if x_ov:
            ax.plot(x_ov, y_ov, color="red", linewidth=2.2,
                    linestyle="-", marker="D", markersize=5, zorder=5)

        ax.set_title(MODEL_LABEL[tag], fontsize=TITLE_FS, pad=6, fontweight=TITLE_WEIGHT)
        ax.set_ylabel("Pass@k (%)", fontsize=LABEL_FS, style="italic", fontweight=TEXT_WEIGHT)
        ax.set_xticks(xs)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
        ax.grid(True, linestyle=":", alpha=0.5)
        _auto_ylim(ax)

    for ax in axes_flat[len(PLOT_ORDER):]:
        ax.set_visible(False)

    _attach_legend(fig, xlabel=r"Number of Attempts ($k$)", include_overall=True)

    out = PLOTS_DIR / "passk.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ==================== Permutation Table ====================

def _load_perm_data(tag: str) -> dict:
    """返回 {stem|dim -> {perm_label -> entry}}，全部 A/B/C/D 均来自 permutation json。"""
    perm_path = RESULTS_DIR / f"permutation_{tag}.json"
    if not perm_path.exists():
        return {}
    perm_data = json.loads(perm_path.read_text())

    by_sd: dict[str, dict] = defaultdict(dict)
    for k, v in perm_data.items():
        sd, perm_key = k.rsplit("|", 1)
        perm_label = perm_key.replace("perm", "")
        by_sd[sd][perm_label] = v

    return {sd: entries for sd, entries in by_sd.items()
            if all(p in entries for p in ALL_PERMS)}


def _get_clip(entry: dict) -> str | None:
    answer = entry.get("answer")
    option_map = entry.get("option_map", {})
    return option_map.get(answer, answer) if answer else None


def _perm_stats(quads: dict, dim: str | None):
    """计算指定 dim 的统计量，dim=None 表示 Overall（全部）。返回 (mean, range, std, clip_flip, correct_flip) 或 None。"""
    if dim is None:
        sub = quads
    else:
        sub = {sd: e for sd, e in quads.items() if sd.endswith(f"|{dim}")}
    if not sub:
        return None

    pairs = [("A","B"),("A","C"),("A","D"),("B","C"),("B","D"),("C","D")]
    n = len(sub)

    perm_accs = [sum(1 for e in sub.values() if e[p].get("correct")) / n
                 for p in ALL_PERMS]
    mean_acc = sum(perm_accs) / 4
    range_   = max(perm_accs) - min(perm_accs)
    std      = (sum((a - mean_acc) ** 2 for a in perm_accs) / 4) ** 0.5

    clip_flips = [
        sum(1 for e in sub.values()
            if _get_clip(e[p1]) != _get_clip(e[p2])) / n
        for p1, p2 in pairs
    ]
    correct_flips = [
        sum(1 for e in sub.values()
            if e[p1].get("correct") != e[p2].get("correct")) / n
        for p1, p2 in pairs
    ]
    return mean_acc, range_, std, sum(clip_flips)/len(clip_flips), sum(correct_flips)/len(correct_flips)


def print_perm_table():
    all_quads = {}
    for tag in PERM_TAGS:
        quads = _load_perm_data(tag)
        if quads:
            all_quads[tag] = quads
        else:
            print(f"[skip] permutation data for {tag} not complete")

    tags = [t for t in PERM_TAGS if t in all_quads]
    if not tags:
        print("No permutation data.")
        return

    header = ["Model", "Dim", "Mean Acc", "Range", "Std", "Clip Flip Rate", "Correctness Flip Rate"]
    rows = []

    for tag in tags:
        quads = all_quads[tag]
        for dim in DIMS:
            stats = _perm_stats(quads, dim)
            if stats is None:
                continue
            mean_acc, range_, std, clip_flip, correct_flip = stats
            rows.append([MODEL_LABEL[tag], dim,
                         f"{mean_acc:.1%}", f"{range_:.1%}", f"{std:.1%}",
                         f"{clip_flip:.1%}", f"{correct_flip:.1%}"])
        # Overall 行
        stats = _perm_stats(quads, None)
        if stats is not None:
            mean_acc, range_, std, clip_flip, correct_flip = stats
            rows.append([MODEL_LABEL[tag], "Overall",
                         f"{mean_acc:.1%}", f"{range_:.1%}", f"{std:.1%}",
                         f"{clip_flip:.1%}", f"{correct_flip:.1%}"])

    # 打印表格
    col_w = [max(len(header[i]), max(len(r[i]) for r in rows)) + 2
             for i in range(len(header))]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    def fmt_row(r):
        return "|" + "|".join(f" {r[i]:<{col_w[i]-1}}" for i in range(len(r))) + "|"

    print("\n=== Permutation Consistency Table ===")
    print(sep)
    print(fmt_row(header))
    print(sep)
    prev_model = None
    for row in rows:
        if prev_model and row[0] != prev_model:
            print(sep)
        print(fmt_row(row))
        prev_model = row[0]
    print(sep)

    out_csv = PLOTS_DIR / "perm_consistency_table.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_frames()
    plot_passk()
    print_perm_table()
