#!/usr/bin/env python3
import argparse
import json
import math
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d import proj3d
try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


# ---------------------------------------------------------
# Global style
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 160,
    "savefig.dpi": 240,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TEXT_WEIGHT = "semibold"

COLOR_S = "#4C78A8"
COLOR_A = "#F58518"
COLOR_C = "#54A24B"
COLOR_NEUTRAL = "#9D9D9D"

A_SUB_COLORS = {
    "Advanced": "#5B8FF9",
    "Deferred": "#61DDAA",
    "Expanded": "#F6BD16",
}
C_SUB_COLORS = {
    "Reverse": "#E8684A",
    "Reorder": "#6DC8EC",
    "Repeat": "#9270CA",
}
CONSIST_COLORS = {
    "3/3": "#3B82F6",
    "2/3": "#60A5FA",
    "1/3": "#93C5FD",
    "0/3": "#DBEAFE",
}

MODEL_BAR_PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_EVAL_ROOT = REPO_ROOT / "TempCloze" / "eval_results"
OUTPUT_ROOT = REPO_ROOT / "output"

DATASET_SOURCE_ORDER = ["care", "dailyomni", "egolife", "favor", "lvd", "mira", "tt"]
DATASET_SOURCE_LABELS = {
    "care": "CARE",
    "dailyomni": "DailyOmni",
    "egolife": "EgoLife",
    "favor": "FAVOR",
    "lvd": "LVD",
    "mira": "MIRA",
    "tt": "TT",
}

DATASET_SOURCE_MODEL_PATHS = [
    ("Seed1.8-T", MODEL_EVAL_ROOT / "closed" / "eval_results" / "seed-1-8-thinking.json"),
    ("Qwen3.5-397B-A17B", MODEL_EVAL_ROOT / "closed" / "eval_results" / "qwen3.5-397b-a17b.json"),
]

MODEL_NAME_OVERRIDES = {
    "Gemini3.Flash": "Gemini3-Flash",
}

MODEL_EVAL_PATHS = {
    "Seed1.8-T": MODEL_EVAL_ROOT / "closed" / "eval_results" / "seed-1-8-thinking.json",
    "Gemini2.5-Pro": MODEL_EVAL_ROOT / "closed" / "eval_results" / "gemini-2.5-pro.json",
    "Qwen3.5-Plus": MODEL_EVAL_ROOT / "closed" / "eval_results" / "qwen3.5-plus.json",
    "InternVL3.5-38B": MODEL_EVAL_ROOT / "open" / "eval_results" / "InternVL3_5-38B.json",
    "Qwen3VL-8B-I": MODEL_EVAL_ROOT / "open" / "eval_results" / "Qwen3-VL-8B-Instruct.json",
}


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wrap_label(s: str, width: int = 18) -> str:
    s = s.replace("vllm-", "vllm ")
    s = s.replace("OpenGVLab_", "OpenGVLab ")
    s = s.replace("_", " ")
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=True))


def save_fig(fig, out_dir: str, stem: str):
    p = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def maybe_adjust_text(ax, texts, x=None, y=None, objects=None, **extra_kwargs):
    if adjust_text is None or not texts:
        return

    kwargs = {
        "ax": ax,
        "arrowprops": dict(arrowstyle="-", color="#999999", lw=0.5, alpha=0.6),
        "expand": (1.12, 1.25),
        "force_text": (0.2, 0.3),
        "force_static": (0.45, 0.55),
        "force_pull": (0.02, 0.02),
        "ensure_inside_axes": True,
        "only_move": {"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
    }

    if x is not None and y is not None:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if mask.any():
            kwargs["x"] = x_arr[mask]
            kwargs["y"] = y_arr[mask]

    if objects:
        kwargs["objects"] = objects

    if extra_kwargs:
        kwargs.update(extra_kwargs)

    try:
        adjust_text(texts, **kwargs)
        return
    except Exception:
        pass

    kwargs.pop("objects", None)
    try:
        adjust_text(texts, **kwargs)
    except Exception:
        return


def safe_zscore(arr: np.ndarray, axis=1):
    mean = arr.mean(axis=axis, keepdims=True)
    std = arr.std(axis=axis, keepdims=True)
    std[std == 0] = 1.0
    return (arr - mean) / std


def derive_consistency_bins(model_stats: dict) -> dict[str, float]:
    """Return exclusive 3/3, 2/3, 1/3, 0/3 proportions from inclusive overlap stats."""
    direct_keys = ["3/3", "2/3", "1/3", "0/3"]
    if all(model_stats.get(k) is not None for k in direct_keys):
        return {k: float(model_stats[k]) for k in direct_keys}

    s_acc = model_stats.get("S_acc")
    a_acc = model_stats.get("A_acc")
    p_acc = model_stats.get("P_acc")
    sa = model_stats.get("S+A")
    sp = model_stats.get("S+P", model_stats.get("S+C"))
    ap = model_stats.get("A+P", model_stats.get("A+C"))
    sap = model_stats.get("S+A+P", model_stats.get("S+A+C"))

    required = [s_acc, a_acc, p_acc, sa, sp, ap, sap]
    if any(v is None for v in required):
        return {k: np.nan for k in direct_keys}

    x111 = float(sap)
    x110 = float(sa) - x111
    x101 = float(sp) - x111
    x011 = float(ap) - x111
    x100 = float(s_acc) - float(sa) - float(sp) + x111
    x010 = float(a_acc) - float(sa) - float(ap) + x111
    x001 = float(p_acc) - float(sp) - float(ap) + x111
    x000 = 1.0 - (x111 + x110 + x101 + x011 + x100 + x010 + x001)

    derived = {
        "3/3": x111,
        "2/3": x110 + x101 + x011,
        "1/3": x100 + x010 + x001,
        "0/3": x000,
    }

    # Numerical noise can produce tiny negatives or totals slightly off 1.
    derived = {k: float(np.clip(v, 0.0, 1.0)) for k, v in derived.items()}
    total = sum(derived.values())
    if total > 0:
        derived = {k: v / total for k, v in derived.items()}
    return derived


def load_question_dataset_sources() -> dict[str, str]:
    """Map each question stem in TempCloze/choices to its source dataset name."""
    lookup = {}
    for dataset_name in DATASET_SOURCE_ORDER:
        meta_path = OUTPUT_ROOT / dataset_name / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing dataset meta file: {meta_path}")
        meta = load_json(str(meta_path))
        for key in meta.keys():
            stem = Path(key).stem
            if stem in lookup:
                raise ValueError(f"Question stem {stem} appears in multiple dataset sources.")
            lookup[stem] = dataset_name
    return lookup


def parse_range(s: str) -> tuple[float, float]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid range: {s}. Expected format low,high")
    low = float(parts[0])
    high = float(parts[1])
    if low > high:
        raise ValueError(f"Invalid range: low ({low}) > high ({high})")
    return low, high


def filter_models_by_acc(df: pd.DataFrame, low: float, high: float, exclude_zero: bool = True) -> pd.DataFrame:
    keep = (df["acc"] < low) | (df["acc"] > high)
    if exclude_zero:
        keep = keep & (df["acc"] != 0)
    return df.loc[keep].copy().reset_index(drop=True)


def ternary_to_xy(a, b, c):
    """Map ternary coordinates (sum=1) to 2D Cartesian coordinates."""
    s = a + b + c
    if s == 0:
        return 0.5, math.sqrt(3) / 6
    a, b, c = a / s, b / s, c / s
    # vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    x = b + 0.5 * c
    y = (math.sqrt(3) / 2) * c
    return x, y


def draw_ternary_axes(ax, labels, title=None):
    h = math.sqrt(3) / 2
    tri = np.array([[0, 0], [1, 0], [0.5, h], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.2)
    # Grid lines
    for t in [0.2, 0.4, 0.6, 0.8]:
        x1, y1 = ternary_to_xy(t, 0, 1 - t)
        x2, y2 = ternary_to_xy(t, 1 - t, 0)
        ax.plot([x1, x2], [y1, y2], color="#dddddd", lw=0.7)
        x1, y1 = ternary_to_xy(0, t, 1 - t)
        x2, y2 = ternary_to_xy(1 - t, t, 0)
        ax.plot([x1, x2], [y1, y2], color="#dddddd", lw=0.7)
        x1, y1 = ternary_to_xy(0, 1 - t, t)
        x2, y2 = ternary_to_xy(1 - t, 0, t)
        ax.plot([x1, x2], [y1, y2], color="#dddddd", lw=0.7)

    ax.text(-0.06, -0.04, labels[0], ha="right", va="top", fontsize=12, weight="bold")
    ax.text(1.06, -0.04, labels[1], ha="left", va="top", fontsize=12, weight="bold")
    ax.text(0.5, h + 0.06, labels[2], ha="center", va="bottom", fontsize=12, weight="bold")
    if title:
        ax.set_title(title, pad=16)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, h + 0.12)
    ax.set_aspect("equal")
    ax.axis("off")


def add_point_labels(ax, xs, ys, labels, n_top=12, score=None, fontsize=8):
    if score is None:
        idx = np.arange(len(labels))
    else:
        idx = np.argsort(score)[-n_top:]
    texts = []
    for i in idx:
        texts.append(ax.text(xs[i], ys[i], labels[i], fontsize=fontsize, ha="left", va="bottom"))
    maybe_adjust_text(ax, texts, x=xs, y=ys)


# ---------------------------------------------------------
# Data preparation
# ---------------------------------------------------------
def build_dataframe(report: dict):
    rows = []
    total_questions = report.get("total_questions", None)
    for model_name, m in report["models"].items():
        model_display = MODEL_NAME_OVERRIDES.get(model_name, model_name)
        err = m.get("error_source", {})
        a = err.get("A", {})
        c = err.get("P", err.get("C", {}))
        consistency = derive_consistency_bins(m)

        row = {
            "model": model_display,
            "model_display": model_display,
            "total_questions_global": total_questions,
            "total_questions_model": m.get("total_questions", np.nan),
            "invalid_entries_skipped": m.get("invalid_entries_skipped", 0),
            "S_acc": m.get("S_acc", np.nan),
            "A_acc": m.get("A_acc", np.nan),
            "P_acc": m.get("P_acc", np.nan),
            "acc": m.get("acc", np.nan),
            "3/3": consistency["3/3"],
            "2/3": consistency["2/3"],
            "1/3": consistency["1/3"],
            "0/3": consistency["0/3"],
            "A_total_errors": a.get("total_errors", a.get("total", 0)),
            "A_Earlier_pct": a.get("Early", {}).get("pct", np.nan),
            "A_Later_pct": a.get("Late", {}).get("pct", np.nan),
            "A_Extended_pct": a.get("Wide", {}).get("pct", np.nan),
            "A_Earlier_count": a.get("Early", {}).get("count", 0),
            "A_Later_count": a.get("Late", {}).get("count", 0),
            "A_Extended_count": a.get("Wide", {}).get("count", 0),
            "C_total_errors": c.get("total_errors", c.get("total", 0)),
            "C_Reverse_pct": c.get("Reverse", {}).get("pct", np.nan),
            "C_Shuffle_pct": c.get("Shuffle", {}).get("pct", np.nan),
            "C_Loop_pct": c.get("Loop", {}).get("pct", np.nan),
            "C_Reverse_count": c.get("Reverse", {}).get("count", 0),
            "C_Shuffle_count": c.get("Shuffle", {}).get("count", 0),
            "C_Loop_count": c.get("Loop", {}).get("count", 0),
        }

        # Derived metrics
        row["error_S"] = 1 - row["S_acc"]
        row["error_A"] = 1 - row["A_acc"]
        row["error_P"] = 1 - row["P_acc"]
        row["A_bottleneck"] = ((row["S_acc"] + row["P_acc"]) / 2) - row["A_acc"]
        row["P_minus_S"] = row["P_acc"] - row["S_acc"]
        row["A_minus_S"] = row["A_acc"] - row["S_acc"]
        row["P_minus_A"] = row["P_acc"] - row["A_acc"]
        row["consistency_balance"] = row["3/3"] - row["0/3"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("acc", ascending=False).reset_index(drop=True)
    return df


def export_metrics_table(df: pd.DataFrame, out_dir: str):
    cols = [
        "model", "acc", "S_acc", "A_acc", "P_acc",
        "3/3", "2/3", "1/3", "0/3",
        "A_bottleneck", "P_minus_S", "A_Earlier_pct", "A_Later_pct", "A_Extended_pct",
        "C_Reverse_pct", "C_Shuffle_pct", "C_Loop_pct"
    ]
    export_df = df[cols].rename(columns={
        "C_Reverse_pct": "P_Reverse_pct",
        "C_Shuffle_pct": "P_Shuffle_pct",
        "C_Loop_pct": "P_Loop_pct",
    })
    export_df.to_csv(os.path.join(out_dir, "temporal_cloze_metrics_table_v2.csv"), index=False)


# ---------------------------------------------------------
# Figure 01: task overview
# ---------------------------------------------------------
def plot_task_overview(out_dir):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis("off")

    ax.text(0.02, 0.95, "Temporal Cloze: infer the unique missing middle clip", fontsize=20, weight="bold",
            ha="left", va="top", transform=ax.transAxes)

    # Top pipeline
    y = 0.68
    w = 0.16
    h = 0.16
    xs = [0.06, 0.28, 0.50, 0.72]
    labels = ["BEGINNING\nbefore.mp4", "MISSING\nGT.mp4", "ENDING\nafter.mp4", "OUTPUT JSON\n{\"answer\":\"B\", ...}"]
    fills = ["#DDECF9", "#FFF2CC", "#D9F2DD", "#F3E8FF"]
    for i, x in enumerate(xs):
        rect = Rectangle((x, y), w, h, facecolor=fills[i], edgecolor="black", lw=1.4, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, labels[i], ha="center", va="center", fontsize=12, weight="bold", transform=ax.transAxes)

    for i in range(3):
        arr = FancyArrowPatch((xs[i] + w, y + h/2), (xs[i+1], y + h/2), transform=ax.transAxes,
                              arrowstyle="-|>", mutation_scale=16, lw=1.5, color="black")
        ax.add_patch(arr)

    ax.text(0.28 + w/2, y - 0.05, "model sees before + after, then selects the correct middle", ha="center",
            va="top", fontsize=11, color="#555555", transform=ax.transAxes)

    # Bottom: taxonomy
    ax.text(0.02, 0.48, "Negative option taxonomy", fontsize=16, weight="bold", ha="left", transform=ax.transAxes)

    blocks = [
        ("S: random distractors", ["Rand1", "Rand2", "Rand3"], "#E8F1FB"),
        ("A: nearby temporal distractors", ["Advanced", "Deferred", "Expanded"], "#FFF1E8"),
        ("C: temporal corruption distractors", ["Reverse", "Reorder", "Repeat"], "#EDF8ED"),
    ]
    bx = [0.05, 0.37, 0.69]
    for i, (title, items, fill) in enumerate(blocks):
        x = bx[i]
        rect = Rectangle((x, 0.12), 0.25, 0.26, facecolor=fill, edgecolor="black", lw=1.2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.125, 0.34, title, ha="center", va="center", fontsize=13, weight="bold", transform=ax.transAxes)
        for j, it in enumerate(items):
            r = Rectangle((x + 0.03 + j * 0.073, 0.18), 0.06, 0.08, facecolor="white", edgecolor="#666666", lw=1.0,
                          transform=ax.transAxes)
            ax.add_patch(r)
            ax.text(x + 0.06 + j * 0.073, 0.22, it, ha="center", va="center", fontsize=10, transform=ax.transAxes)

    ax.text(0.175, 0.08, "semantic distractors", ha="center", va="center", fontsize=11, color="#4C78A8",
            transform=ax.transAxes)
    ax.text(0.495, 0.08, "hardest: temporally adjacent but plausible", ha="center", va="center", fontsize=11,
            color="#F58518", transform=ax.transAxes)
    ax.text(0.815, 0.08, "content reused but internal order broken", ha="center", va="center", fontsize=11,
            color="#54A24B", transform=ax.transAxes)

    save_fig(fig, out_dir, "figure_01_task_overview_v2")


# ---------------------------------------------------------
# Figure 02: capability map
# ---------------------------------------------------------
def plot_capability_map(df, out_dir):
    pairs = [
        ("A_acc", "P_acc", "A Accuracy", "P Accuracy", "A vs P", "figure_02_capability_map_A_vs_C_v2"),
        ("P_acc", "S_acc", "P Accuracy", "S Accuracy", "P vs S", "figure_02_capability_map_C_vs_S_v2"),
        ("S_acc", "A_acc", "S Accuracy", "A Accuracy", "S vs A", "figure_02_capability_map_S_vs_A_v2"),
    ]

    top13_df = df.sort_values("acc", ascending=False).head(13).copy()
    # Similar-by-accuracy downsampling: sort by acc and keep one in every adjacent pair.
    top13_keep_models = set(top13_df.iloc[np.arange(0, len(top13_df), 2)]["model"].tolist())
    marker_size = 170

    for x_col, y_col, x_lab, y_lab, title, stem in pairs:
        fig, ax = plt.subplots(figsize=(9.2, 8))

        scatter_artists = []
        d_top = df[df["model"].isin(top13_keep_models)].copy()
        d_rest = df[~df["model"].isin(top13_df["model"])].copy()

        d_top = d_top[np.isfinite(d_top[x_col]) & np.isfinite(d_top[y_col])].copy()
        d_rest = d_rest[np.isfinite(d_rest[x_col]) & np.isfinite(d_rest[y_col])].copy()
        d_plot = pd.concat([d_rest, d_top], axis=0, ignore_index=True)
        if d_plot.empty:
            plt.close(fig)
            continue

        if not d_rest.empty:
            sc_rest = ax.scatter(
                d_rest[x_col],
                d_rest[y_col],
                s=marker_size,
                c="#F58518",
                marker="o",
                alpha=0.95,
                edgecolor="black",
                linewidth=1.0,
                label="Open-Source",
            )
            scatter_artists.append(sc_rest)

        if not d_top.empty:
            sc_top = ax.scatter(
                d_top[x_col],
                d_top[y_col],
                s=marker_size,
                c="#54A24B",
                marker="^",
                alpha=0.95,
                edgecolor="black",
                linewidth=1.0,
                label="Proprietary",
            )
            scatter_artists.append(sc_top)

        texts = []
        for idx, (_, r) in enumerate(d_plot.iterrows()):
            dx = 0.006 if (idx % 2 == 0) else -0.006
            dy = 0.004 if ((idx // 2) % 2 == 0) else -0.004
            texts.append(ax.text(r[x_col] + dx, r[y_col] + dy, r["model"], fontsize=12, fontweight="bold"))

        maybe_adjust_text(
            ax,
            texts,
            x=d_plot[x_col].to_numpy(),
            y=d_plot[y_col].to_numpy(),
            objects=scatter_artists,
            expand=(1.2, 1.35),
            force_text=(0.35, 0.55),
            force_static=(0.8, 1.0),
            force_explode=(0.5, 0.9),
            max_move=(90, 90),
            pull_threshold=6,
            iter_lim=1800,
            min_arrow_len=2,
        )

        # Random baseline guides and requested slope=1 reference line.
        ax.axvline(0.25, color="#999999", ls="--", lw=1.1)
        ax.axhline(0.25, color="#999999", ls="--", lw=1.1)
        x_line = np.array([0.0, 1.0])
        ax.plot(x_line, x_line, color="#9E9E9E", ls="--", lw=1.2, zorder=0)

        x_min = float(d_plot[x_col].min())
        y_min = float(d_plot[y_col].min())
        ax.set_xlim(x_min - 0.1, 1.0)
        ax.set_ylim(y_min - 0.1, 1.0)
        ax.tick_params(axis="both", labelsize=14)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.set_xlabel(x_lab, fontsize=16, fontweight="bold")
        ax.set_ylabel(y_lab, fontsize=16, fontweight="bold")
        ax.set_title(f"{title}", fontsize=18, fontweight="bold")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, loc="lower right", prop={"size": 14, "weight": "bold"})
        save_fig(fig, out_dir, stem)


def plot_capability_map_3d(
    df,
    out_dir,
    label_topk=15,
    elev=22.0,
    azim=38.0,
):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    acc = df["acc"].to_numpy()
    acc_min, acc_max = float(acc.min()), float(acc.max())
    sizes = 60 + 900 * (acc - acc_min) / (acc_max - acc_min + 1e-9)
    norm = colors.Normalize(vmin=acc_min, vmax=acc_max)

    sc = ax.scatter(
        df["S_acc"],
        df["A_acc"],
        df["P_acc"],
        s=sizes,
        c=acc,
        cmap="viridis",
        norm=norm,
        alpha=0.9,
        edgecolor="none",
        linewidth=0,

        depthshade=True,
    )

    # Random baseline reference in 3D: S=A=C=0.25 planes + key dashed guide lines.
    baseline = 0.25
    grid = np.linspace(0.0, 1.0, 12)
    Y, Z = np.meshgrid(grid, grid)
    X = np.full_like(Y, baseline)
    ax.plot_surface(X, Y, Z, color="#4C78A8", alpha=0.06, linewidth=0, shade=False)

    X, Z = np.meshgrid(grid, grid)
    Y = np.full_like(X, baseline)
    ax.plot_surface(X, Y, Z, color="#F58518", alpha=0.06, linewidth=0, shade=False)

    X, Y = np.meshgrid(grid, grid)
    Z = np.full_like(X, baseline)
    ax.plot_surface(X, Y, Z, color="#54A24B", alpha=0.06, linewidth=0, shade=False)

    ax.plot([0, 1], [baseline, baseline], [baseline, baseline], ls="--", lw=1.1, color="#4C78A8")
    ax.plot([baseline, baseline], [0, 1], [baseline, baseline], ls="--", lw=1.1, color="#F58518")
    ax.plot([baseline, baseline], [baseline, baseline], [0, 1], ls="--", lw=1.1, color="#54A24B")
    ax.text(
        baseline + 0.01,
        baseline + 0.01,
        baseline + 0.01,
        "random baseline 0.25",
        fontsize=8,
        color="#666666",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_xlabel("S Accuracy")
    ax.set_ylabel("A Accuracy")
    ax.set_zlabel("C Accuracy")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("3D capability map: S/A/C axes, size+color = overall Accuracy (with random baseline 0.25)")

    # Draw labels as 2D projected annotations so they are never hidden by 3D markers.
    d = df.sort_values("acc", ascending=False).head(label_topk)
    fig.canvas.draw()
    for _, r in d.iterrows():
        x2, y2, _ = proj3d.proj_transform(r["S_acc"], r["A_acc"], r["P_acc"], ax.get_proj())
        ax.annotate(
            r["model"],
            xy=(x2, y2),
            xycoords="data",
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="bottom",
            annotation_clip=False,
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.8)
    cbar.set_label("Overall Accuracy")
    save_fig(fig, out_dir, "figure_02b_capability_map_3d_SAC_v2")


# ---------------------------------------------------------
# Figure 03: global error ternary
# ---------------------------------------------------------
def plot_error_ternary(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 8))
    draw_ternary_axes(ax, labels=["Error on S", "Error on A", "Error on C"],
                      title="Where does each model's error budget go?")
    xs, ys = [], []
    for _, r in df.iterrows():
        x, y = ternary_to_xy(r["error_S"], r["error_A"], r["error_P"])
        xs.append(x); ys.append(y)
    sizes = 100 + 900 * df["acc"].to_numpy()
    colors_arr = df["A_bottleneck"].to_numpy()
    sc = ax.scatter(xs, ys, s=sizes, c=colors_arr, cmap="coolwarm", edgecolor="black", linewidth=0.7, alpha=0.88)
    add_point_labels(ax, np.array(xs), np.array(ys), df["model"].tolist(), n_top=15, score=df["acc"].to_numpy())

    cbar = fig.colorbar(sc, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("A bottleneck score")
    save_fig(fig, out_dir, "figure_03_error_ternary_v2")


# ---------------------------------------------------------
# Figure 04: consistency stacked
# ---------------------------------------------------------
def plot_consistency_stacked(df, out_dir):
    d = df.sort_values("3/3", ascending=False).reset_index(drop=True)
    label_fs = 15
    tick_fs = 10
    legend_fs = 14
    text_weight = "semibold"

    fig, ax = plt.subplots(figsize=(16, 4.4))
    x = np.arange(len(d))
    bottom = np.zeros(len(d))
    for key in ["0/3", "1/3", "2/3", "3/3"]:
        ax.bar(x, d[key], bottom=bottom, label=key, color=CONSIST_COLORS[key], width=0.62, edgecolor="white")
        bottom += d[key].to_numpy()

    ax.set_xticks(x)
    ax.set_xticklabels(d["model_display"], rotation=45, ha="center", fontsize=tick_fs)
    ax.set_ylabel("Proportion", fontsize=label_fs, style="italic", fontweight=text_weight)
    ax.set_title("")
    ax.tick_params(axis="both", labelsize=tick_fs)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("normal")
    for tick in ax.get_yticklabels():
        tick.set_fontweight(text_weight)
    legend_keys = ["0/3", "1/3", "2/3", "3/3"]
    leg_ax = ax.inset_axes([1.015, 0.00, 0.08, 1.00], transform=ax.transAxes)
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    step = 1.0 / len(legend_keys)
    for i, key in enumerate(legend_keys):
        y0 = i * step
        leg_ax.add_patch(Rectangle((0.05, y0), 0.35, step,
                                   facecolor=CONSIST_COLORS[key], edgecolor="white", linewidth=1.0))
        leg_ax.text(0.48, y0 + step / 2, key, ha="left", va="center",
                    fontsize=legend_fs, fontweight=text_weight)
    leg_ax.axis("off")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.2)
    fig.subplots_adjust(right=0.80, top=0.96, bottom=0.22)

    save_fig(fig, out_dir, "figure_04_consistency_stacked_v2")


def build_dataset_consistency_distribution(model_path: Path, dataset_lookup: dict[str, str]) -> pd.DataFrame:
    """Aggregate 0/3..3/3 proportions by dataset source for one model eval json."""
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model eval file: {model_path}")

    raw = load_json(str(model_path))
    rows = []
    for qid, entries in raw.items():
        if not all(
            dim in entries
            and isinstance(entries[dim], dict)
            and isinstance(entries[dim].get("correct"), bool)
            for dim in ("S", "A", "C")
        ):
            continue
        dataset_name = dataset_lookup.get(qid)
        if dataset_name is None:
            raise KeyError(f"Question stem {qid} not found in dataset source lookup.")
        n_correct = sum(int(bool(entries[dim]["correct"])) for dim in ("S", "A", "C"))
        rows.append({"dataset": dataset_name, "n_correct": n_correct})

    if not rows:
        raise ValueError(f"No valid S/A/C triples found in {model_path}")

    df_rows = pd.DataFrame(rows)
    counts = (
        df_rows.groupby(["dataset", "n_correct"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DATASET_SOURCE_ORDER, fill_value=0)
        .reindex(columns=[0, 1, 2, 3], fill_value=0)
    )
    proportions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    out = proportions.rename(columns={0: "0/3", 1: "1/3", 2: "2/3", 3: "3/3"}).reset_index()
    out["dataset_display"] = out["dataset"].map(DATASET_SOURCE_LABELS)
    return out


def plot_dataset_consistency_stacked_for_model(
    model_display: str,
    model_path: Path,
    dataset_lookup: dict[str, str],
    out_dir: str,
):
    d = build_dataset_consistency_distribution(model_path, dataset_lookup)
    label_fs = 15
    tick_fs = 11
    legend_fs = 14
    text_weight = "semibold"

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    x = np.arange(len(d))
    bottom = np.zeros(len(d))
    for key in ["0/3", "1/3", "2/3", "3/3"]:
        ax.bar(x, d[key], bottom=bottom, label=key, color=CONSIST_COLORS[key], width=0.62, edgecolor="white")
        bottom += d[key].to_numpy()

    ax.set_xticks(x)
    ax.set_xticklabels(d["dataset_display"], rotation=20, ha="center", fontsize=tick_fs)
    ax.set_ylabel("Proportion", fontsize=label_fs, style="italic", fontweight=text_weight)
    ax.tick_params(axis="both", labelsize=tick_fs)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(text_weight)

    ax.text(
        0.015, 0.98, model_display,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, fontweight="bold",
    )

    legend_keys = ["0/3", "1/3", "2/3", "3/3"]
    leg_ax = ax.inset_axes([1.015, 0.00, 0.08, 1.00], transform=ax.transAxes)
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    step = 1.0 / len(legend_keys)
    for i, key in enumerate(legend_keys):
        y0 = i * step
        leg_ax.add_patch(Rectangle((0.05, y0), 0.35, step,
                                   facecolor=CONSIST_COLORS[key], edgecolor="white", linewidth=1.0))
        leg_ax.text(0.48, y0 + step / 2, key, ha="left", va="center",
                    fontsize=legend_fs, fontweight=text_weight)
    leg_ax.axis("off")

    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.2)
    fig.subplots_adjust(right=0.82, top=0.95, bottom=0.22)

    stem = f"figure_04_dataset_consistency_{model_display.lower().replace('.', '').replace('-', '_')}_v2"
    save_fig(fig, out_dir, stem)


def plot_dataset_consistency_for_selected_models(out_dir: str):
    dataset_lookup = load_question_dataset_sources()
    model_dists = {}
    for model_display, model_path in DATASET_SOURCE_MODEL_PATHS:
        model_dists[model_display] = build_dataset_consistency_distribution(model_path, dataset_lookup).set_index("dataset")

    difficulty_score = {}
    for dataset_name in DATASET_SOURCE_ORDER:
        per_model_scores = []
        for model_display in model_dists:
            row = model_dists[model_display].loc[dataset_name]
            score = (
                row["1/3"] * 1
                + row["2/3"] * 2
                + row["3/3"] * 3
            ) / 3.0
            per_model_scores.append(float(score))
        difficulty_score[dataset_name] = float(np.mean(per_model_scores))

    dataset_order = sorted(DATASET_SOURCE_ORDER, key=lambda name: difficulty_score[name])
    dataset_labels = [DATASET_SOURCE_LABELS[name] for name in dataset_order]

    label_fs = 15
    tick_fs = 11
    legend_fs = 13
    text_weight = "semibold"
    bar_width = 0.34
    x = np.arange(len(dataset_order))
    offsets = [-bar_width / 1.7, bar_width / 1.7]
    model_hatches = {
        DATASET_SOURCE_MODEL_PATHS[0][0]: "",
        DATASET_SOURCE_MODEL_PATHS[1][0]: "///",
    }

    fig, ax = plt.subplots(figsize=(10.8, 4.6))

    for offset, (model_display, _) in zip(offsets, DATASET_SOURCE_MODEL_PATHS):
        d = model_dists[model_display].loc[dataset_order].reset_index()
        bottom = np.zeros(len(d))
        for key in ["0/3", "1/3", "2/3", "3/3"]:
            ax.bar(
                x + offset,
                d[key],
                bottom=bottom,
                width=bar_width,
                color=CONSIST_COLORS[key],
                edgecolor="#333333",
                linewidth=0.7,
                hatch=model_hatches[model_display],
                zorder=3,
            )
            bottom += d[key].to_numpy()

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, rotation=18, ha="center", fontsize=tick_fs)
    ax.set_ylabel("Proportion", fontsize=label_fs, style="italic", fontweight=text_weight)
    ax.tick_params(axis="both", labelsize=tick_fs)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(text_weight)


    for anchor_x, (model_display, _) in zip([0.27, 0.70], DATASET_SOURCE_MODEL_PATHS):
        model_handle = Rectangle((0, 0), 1, 1, facecolor="#F3F4F6", edgecolor="#333333",
                                 hatch=model_hatches[model_display], linewidth=0.8)
        model_legend = fig.legend(
            [model_handle],
            [model_display],
            loc="upper center",
            bbox_to_anchor=(anchor_x, 0.985),
            bbox_transform=fig.transFigure,
            frameon=False,
            fontsize=legend_fs,
            handlelength=1.8,
            handletextpad=0.6,
            borderpad=0.2,
            prop={"size": legend_fs, "weight": text_weight},
        )
        fig.add_artist(model_legend)

    legend_keys = ["0/3", "1/3", "2/3", "3/3"]
    leg_ax = ax.inset_axes([1.015, 0.00, 0.08, 1.00], transform=ax.transAxes)
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    step = 1.0 / len(legend_keys)
    for i, key in enumerate(legend_keys):
        y0 = i * step
        leg_ax.add_patch(Rectangle((0.05, y0), 0.35, step,
                                   facecolor=CONSIST_COLORS[key], edgecolor="white", linewidth=1.0))
        leg_ax.text(0.48, y0 + step / 2, key, ha="left", va="center",
                    fontsize=legend_fs, fontweight=text_weight)
    leg_ax.axis("off")

    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    fig.subplots_adjust( top=0.92, bottom=0.22)

    save_fig(fig, out_dir, "figure_04_dataset_consistency_two_models_v2")


# ---------------------------------------------------------
# Figure 05-06: error source heatmaps
# ---------------------------------------------------------
def plot_error_source_heatmaps(df, out_dir):
    cols = ["A_Earlier_pct", "A_Later_pct", "A_Extended_pct", "C_Reverse_pct", "C_Shuffle_pct", "C_Loop_pct"]
    pretty = ["Advanced", "Deferred", "Expanded", "Reversed", "Reordered", "Repeated"]
    d = df.sort_values("acc", ascending=False).reset_index(drop=True)

    mat = d[cols].fillna(0).to_numpy()
    for stem, title, arr, cmap in [
        ("figure_05_error_source_heatmap_raw_v2", "Failure signature heatmap (raw proportions)", mat, "YlOrRd"),
    ]:
        is_raw = stem == "figure_05_error_source_heatmap_raw_v2"
        arr_plot = arr.T  # swap x/y: models on x-axis, subtypes on y-axis
        fig, ax = plt.subplots(figsize=(max(16, 0.42 * len(d) + 2), 3.8))
        im = ax.imshow(arr_plot, aspect="auto", cmap=cmap)

        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels(d["model_display"], rotation=30, ha="center")
        ax.set_yticks(np.arange(len(pretty)))
        ax.set_yticklabels(pretty)
        ax.tick_params(axis="both")
        for tick in ax.get_xticklabels():
            tick.set_fontweight("normal")
        for tick in ax.get_yticklabels():
            tick.set_fontweight(TEXT_WEIGHT)

        # Visually separate A-type and C-type failure subtypes.
        ax.axhline(2.5, color="white", lw=4.0, alpha=0.95, zorder=3)
        ax.axhline(2.5, color="#333333", lw=1.2, alpha=0.95, zorder=4)
        ax.text(-0.10, 0.79, "A-type", transform=ax.transAxes, ha="center", va="center", fontsize=10,
                fontweight="bold", rotation=90, clip_on=False)
        ax.text(-0.10, 0.21, "P-type", transform=ax.transAxes, ha="center", va="center", fontsize=10,
                fontweight="bold", rotation=90, clip_on=False)

        if not is_raw:
            ax.set_ylabel("Failure subtype", fontstyle="italic", fontweight=TEXT_WEIGHT)

        if "raw" in stem:
            for i in range(arr_plot.shape[0]):
                for j in range(arr_plot.shape[1]):
                    ax.text(j, i, f"{arr_plot[i, j]:.2f}", ha="center", va="center", fontsize=8.0,
                            fontweight=TEXT_WEIGHT)

        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        if not is_raw:
            cbar.set_label("Proportion", fontstyle="italic", fontweight=TEXT_WEIGHT)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight(TEXT_WEIGHT)
        save_fig(fig, out_dir, stem)


# ---------------------------------------------------------
# Figure 05b-05c: per-question SAC correctness patterns
# ---------------------------------------------------------
def load_model_sac_binary_df(model_name: str) -> pd.DataFrame:
    """Load per-question binary correctness for S/A/P display tasks.

    Returns one row per question with columns S, A, C containing 0/1 correctness.
    This is the shared representation for pattern-level analysis.
    """
    model_path = MODEL_EVAL_PATHS.get(model_name)
    if model_path is None:
        raise KeyError(f"No eval path configured for model: {model_name}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model eval file: {model_path}")

    raw = load_json(str(model_path))
    rows = []
    for qid, entries in raw.items():
        if not all(
            d in entries
            and isinstance(entries[d], dict)
            and isinstance(entries[d].get("correct"), bool)
            for d in ("S", "A", "C")
        ):
            continue
        rows.append({
            "question_id": qid,
            "S": int(bool(entries["S"]["correct"])),
            "A": int(bool(entries["A"]["correct"])),
            "C": int(bool(entries["C"]["correct"])),
        })

    if not rows:
        raise ValueError(f"No valid S/A/C triples found for {model_name} pattern analysis.")

    return pd.DataFrame(rows)[["question_id", "S", "A", "C"]]


def sac_pattern_distribution(model_name: str) -> pd.DataFrame:
    """Return the 8 SAP-display correctness patterns and their counts/proportions."""
    df = load_model_sac_binary_df(model_name)
    pattern_order = ["111", "110", "101", "011", "100", "010", "001", "000"]
    pattern_labels = {
        "111": "S+A+P",
        "110": "S+A",
        "101": "S+P",
        "011": "A+P",
        "100": "S only",
        "010": "A only",
        "001": "P only",
        "000": "None",
    }

    patterns = df[["S", "A", "C"]].astype(str).agg("".join, axis=1)
    counts = patterns.value_counts().reindex(pattern_order, fill_value=0)
    out = pd.DataFrame({
        "pattern": pattern_order,
        "label": [pattern_labels[p] for p in pattern_order],
        "count": counts.to_numpy(dtype=int),
    })
    out["proportion"] = out["count"] / max(1, int(out["count"].sum()))
    out["n_correct"] = out["pattern"].map(lambda p: p.count("1"))
    out["model"] = model_name
    return out


def export_sac_pattern_table(out_dir: str, model_names=None):
    """Export a CSV table used by Figure 05b/05c."""
    ensure_dir(out_dir)
    if model_names is None:
        model_names = list(MODEL_EVAL_PATHS.keys())
    table = pd.concat([sac_pattern_distribution(name) for name in model_names], ignore_index=True)
    table.to_csv(os.path.join(out_dir, "sac_pattern_distribution_v2.csv"), index=False)
    return table


def draw_sac_pattern_bar(ax, pattern_df: pd.DataFrame, title: str, show_count: bool = True):
    """Draw one SAC pattern bar chart on a supplied axis."""
    color_map = {
        3: "#3B82F6",  # all three correct
        2: "#60A5FA",  # two correct
        1: "#93C5FD",  # one correct
        0: "#DBEAFE",  # none correct
    }
    x = np.arange(len(pattern_df))
    vals = pattern_df["proportion"].to_numpy(dtype=float)
    bar_colors = [color_map[int(k)] for k in pattern_df["n_correct"]]
    bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", linewidth=1.1, width=0.72)

    ax.set_xticks(x)
    ax.set_xticklabels(pattern_df["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, max(0.05, vals.max() * 1.22))
    ax.set_ylabel("Proportion of questions", fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.22)

    for b, prop, cnt in zip(bars, vals, pattern_df["count"]):
        label = f"{prop:.2f}"
        if show_count:
            label += f"\n(n={int(cnt)})"
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(0.004, vals.max() * 0.018),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Add compact visual key for how many of S/A/P are correct.
    handles = [Rectangle((0, 0), 1, 1, facecolor=color_map[k], edgecolor="white") for k in [3, 2, 1, 0]]
    labels = ["3 correct", "2 correct", "1 correct", "0 correct"]
    ax.legend(handles, labels, frameon=False, fontsize=8, ncol=4, loc="upper right")


def plot_seed18t_sac_patterns(out_dir):
    """Figure 05b: SAP correctness pattern distribution for Seed1.8-T."""
    ensure_dir(out_dir)
    d = sac_pattern_distribution("Seed1.8-T")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    draw_sac_pattern_bar(ax, d, "Seed1.8-T: per-question SAP correctness patterns")
    fig.tight_layout()
    save_fig(fig, out_dir, "figure_05b_seed18t_sac_patterns_v2")


def plot_multi_model_sac_patterns(out_dir, model_names=None):
    """Figure 05c: compare SAP correctness pattern distributions across models."""
    ensure_dir(out_dir)
    if model_names is None:
        model_names = list(MODEL_EVAL_PATHS.keys())

    all_d = export_sac_pattern_table(out_dir, model_names=model_names)
    pattern_order = ["111", "110", "101", "011", "100", "010", "001", "000"]
    labels = ["S+A+P", "S+A", "S+P", "A+P", "S only", "A only", "P only", "None"]
    color_map = {
        "111": "#3B82F6",
        "110": "#60A5FA",
        "101": "#7DD3FC",
        "011": "#A78BFA",
        "100": "#93C5FD",
        "010": "#FDBA74",
        "001": "#86EFAC",
        "000": "#DBEAFE",
    }

    pivot = all_d.pivot(index="model", columns="pattern", values="proportion").reindex(model_names)[pattern_order]

    fig, ax = plt.subplots(figsize=(10.8, max(4.6, 0.52 * len(model_names) + 2)))
    y = np.arange(len(model_names))
    left = np.zeros(len(model_names))
    for p, label in zip(pattern_order, labels):
        vals = pivot[p].fillna(0).to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.68, color=color_map[p], edgecolor="white", linewidth=1.0, label=label)
        # Label sizeable segments only to avoid clutter.
        for i, v in enumerate(vals):
            if v >= 0.075:
                ax.text(left[i] + v / 2, y[i], f"{v:.2f}", ha="center", va="center", fontsize=8, color="#111111")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(model_names, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of questions", fontweight="bold")
    ax.set_title("Per-question SAP correctness pattern distribution across models", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(ncol=4, bbox_to_anchor=(0.5, -0.16), loc="upper center", frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save_fig(fig, out_dir, "figure_05c_multi_model_sac_patterns_v2")


# ---------------------------------------------------------
# New figures added for deeper dimensional interpretation
# ---------------------------------------------------------
def plot_dimension_slopegraph(df, out_dir, top_k=18):
    d = df.sort_values("acc", ascending=False).head(top_k).copy()
    d = d.sort_values("A_bottleneck", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(7, 0.42 * len(d) + 1)))
    x = [0, 1, 2]
    labels = ["S", "A", "C"]
    for i, r in d.iterrows():
        ys = [r["S_acc"], r["A_acc"], r["P_acc"]]
        ax.plot(x, ys, color="#999999", alpha=0.55, lw=1.7)
        ax.scatter(x, ys, s=36, color=[COLOR_S, COLOR_A, COLOR_C], zorder=3)
        ax.text(-0.06, ys[0], r["model"], ha="right", va="center", fontsize=8)
        ax.text(2.06, ys[2], f"{r['acc']:.3f}", ha="left", va="center", fontsize=8, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, weight="bold")
    ax.set_ylim(0.18, 1.02)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title("Per-model dimension profile: most strong models show an A bottleneck")
    ax.grid(axis="y", alpha=0.2)
    save_fig(fig, out_dir, "figure_08_dimension_slopegraph_v2")


def plot_bottleneck_quadrant(df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    x = df["A_bottleneck"]
    y = df["P_minus_S"]
    sizes = 120 + 1000 * df["3/3"]
    sc = ax.scatter(x, y, s=sizes, c=df["acc"], cmap="plasma", edgecolor="black", linewidth=0.7, alpha=0.88)
    texts = []
    for _, r in df.iterrows():
        texts.append(ax.text(r["A_bottleneck"] + 0.004, r["P_minus_S"] + 0.002, r["model"], fontsize=12))
    maybe_adjust_text(ax, texts, x=x.to_numpy(), y=y.to_numpy())

    ax.axvline(0, color="#888888", ls="--", lw=1.0)
    ax.axhline(0, color="#888888", ls="--", lw=1.0)
    ax.text(ax.get_xlim()[1] * 0.72 if ax.get_xlim()[1] > 0 else 0.02, ax.get_ylim()[1] * 0.92,
            "A harder than {S,C}", fontsize=10, color="#F58518")
    ax.text(ax.get_xlim()[0] * 0.8 if ax.get_xlim()[0] < 0 else -0.06, ax.get_ylim()[1] * 0.92,
            "A easier than {S,C}", fontsize=10, color="#4C78A8")
    ax.text(ax.get_xlim()[1] * 0.72 if ax.get_xlim()[1] > 0 else 0.02, ax.get_ylim()[0] * 0.88,
            "C weaker than S", fontsize=10, color="#777777")
    ax.text(ax.get_xlim()[1] * 0.72 if ax.get_xlim()[1] > 0 else 0.02, ax.get_ylim()[1] * 0.05,
            "C stronger than S", fontsize=10, color="#54A24B")

    ax.set_xlabel("A bottleneck score = (S + C)/2 - A")
    ax.set_ylabel("C - S")
    ax.set_title("Dimensional bottleneck map")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Overall Accuracy")
    ax.grid(alpha=0.22)
    save_fig(fig, out_dir, "figure_09_bottleneck_quadrant_v2")


def plot_subtype_ternary(df, out_dir, dim="A"):
    if dim == "A":
        cols = ["A_Earlier_pct", "A_Later_pct", "A_Extended_pct"]
        labels = ["Advanced", "Deferred", "Expanded"]
        title = "A-subtype error simplex"
        cmap = "YlOrBr"
        colorval = df["A_acc"].to_numpy()
        stem = "figure_10_A_subtype_ternary_v2"
    else:
        cols = ["C_Reverse_pct", "C_Shuffle_pct", "C_Loop_pct"]
        labels = ["Reverse", "Reorder", "Repeat"]
        title = "C-subtype error simplex"
        cmap = "PuBuGn"
        colorval = df["P_acc"].to_numpy()
        stem = "figure_11_C_subtype_ternary_v2"

    fig, ax = plt.subplots(figsize=(9, 8))
    draw_ternary_axes(ax, labels=labels, title=title)
    xs, ys = [], []
    for _, r in df.iterrows():
        x, y = ternary_to_xy(r[cols[0]], r[cols[1]], r[cols[2]])
        xs.append(x); ys.append(y)
    sizes = 120 + 900 * df["acc"]
    sc = ax.scatter(xs, ys, s=sizes, c=colorval, cmap=cmap, edgecolor="black", linewidth=0.7, alpha=0.9)
    add_point_labels(ax, np.array(xs), np.array(ys), df["model"].tolist(), n_top=15, score=df["acc"].to_numpy())
    cbar = fig.colorbar(sc, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label(f"{dim} Accuracy")
    save_fig(fig, out_dir, stem)


def plot_clustered_feature_heatmap(df, out_dir):
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import pdist

    cols = [
        "S_acc", "A_acc", "P_acc", "3/3", "2/3", "1/3", "0/3",
        "A_Earlier_pct", "A_Later_pct", "A_Extended_pct",
        "C_Reverse_pct", "C_Shuffle_pct", "C_Loop_pct"
    ]
    pretty = ["S", "A", "C", "3/3", "2/3", "1/3", "0/3",
              "Advanced", "Deferred", "Expanded", "Reverse", "Reorder", "Repeat"]

    X = df[cols].fillna(0).to_numpy()
    Xz = safe_zscore(X, axis=0)

    row_link = linkage(pdist(Xz, metric="euclidean"), method="ward")
    col_link = linkage(pdist(Xz.T, metric="euclidean"), method="ward")
    row_ord = leaves_list(row_link)
    col_ord = leaves_list(col_link)

    Xp = Xz[row_ord][:, col_ord]
    row_labels = df["model"].iloc[row_ord].tolist()
    col_labels = [pretty[i] for i in col_ord]

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 8], height_ratios=[1.2, 8], wspace=0.02, hspace=0.02)

    ax_col = fig.add_subplot(gs[0, 1])
    dendrogram(col_link, ax=ax_col, color_threshold=None, no_labels=True)
    ax_col.axis("off")

    ax_row = fig.add_subplot(gs[1, 0])
    dendrogram(row_link, ax=ax_row, orientation="right", color_threshold=None, no_labels=True)
    ax_row.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(Xp, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Clustered feature heatmap: temporal reasoning archetypes")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Standardized feature value")
    save_fig(fig, out_dir, "figure_12_clustered_feature_heatmap_v2")


def plot_pca_archetypes(df, out_dir):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cols = [
        "S_acc", "A_acc", "P_acc", "3/3", "2/3", "1/3", "0/3",
        "A_Earlier_pct", "A_Later_pct", "A_Extended_pct",
        "C_Reverse_pct", "C_Shuffle_pct", "C_Loop_pct"
    ]
    X = df[cols].fillna(0).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(10, 8))
    sizes = 120 + 1000 * df["3/3"].to_numpy()
    sc = ax.scatter(Z[:, 0], Z[:, 1], s=sizes, c=df["acc"], cmap="viridis", edgecolor="black", linewidth=0.7, alpha=0.88)
    texts = []
    for i, label in enumerate(df["model"]):
        texts.append(ax.text(Z[i, 0] + 0.03, Z[i, 1] + 0.02, label, fontsize=12))
    maybe_adjust_text(ax, texts, x=Z[:, 0], y=Z[:, 1])

    ax.axhline(0, color="#cccccc", lw=1)
    ax.axvline(0, color="#cccccc", lw=1)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("Model archetype map from temporal-cloze behavior features")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Overall Accuracy")
    ax.grid(alpha=0.18)
    save_fig(fig, out_dir, "figure_13_pca_model_archetypes_v2")


def plot_leaderboard(df, out_dir, top_k=20):
    d = df.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.36 * len(d) + 1)))
    bars = ax.barh(d["model"], d["acc"], color=cm.viridis(np.linspace(0.2, 0.9, len(d))))
    for b, v in zip(bars, d["acc"]):
        ax.text(v + 0.007, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.25, max(0.3, d["acc"].max() + 0.08))

    split_after_rank = 13
    if len(d) > split_after_rank:
        y_sep = len(d) - split_after_rank - 0.5
        ax.axhline(y_sep, ls="--", color="#666666", lw=1.2, alpha=0.9)
        ax.text(
            0.99,
            y_sep + 0.28,
            "Close / Open Source Model",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=12,
            color="#555555",
            clip_on=False,
        )

    ax.set_xlabel("Overall Accuracy")
    save_fig(fig, out_dir, "figure_14_leaderboard_top20_v2")


def _draw_group_label_line(ax, x0, x1, y, text, color, fontsize, line_width=2.0, gap_pad=0.02):
    center_x = (x0 + x1) / 2
    txt = ax.text(
        center_x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=TEXT_WEIGHT,
        color=color,
        clip_on=False,
    )
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bbox = txt.get_window_extent(renderer=renderer)
    x_gap0 = ax.transData.inverted().transform((bbox.x0, bbox.y0))[0]
    x_gap1 = ax.transData.inverted().transform((bbox.x1, bbox.y0))[0]
    left_stop = max(x0, x_gap0 - gap_pad)
    right_start = min(x1, x_gap1 + gap_pad)

    if left_stop > x0:
        ax.plot([x0, left_stop], [y, y], color=color, lw=line_width, solid_capstyle="round", clip_on=False)
    if right_start < x1:
        ax.plot([right_start, x1], [y, y], color=color, lw=line_width, solid_capstyle="round", clip_on=False)


def plot_top6_metric_bars(df, out_dir, top_k=10, font_scale=1.0, show_value_labels=False, stem="figure_15_top6_metric_bars_no_value_v2"):
    d_all = df.sort_values("acc", ascending=False).reset_index(drop=True)

    prop_models = [
        "Seed1.8-T",
        "Qwen3.5-Plus",
        "Seed1.8-I",
        "Gemini2.5-Pro",
        "Gemini2.5-Flash",
        "GPT5.4",
        "Claude4.6-Sonnet",
        "Claude4.6-Opus",
        "Gemini3-Flash",
        "Seed1.6",
        "Grok4.1",
    ]
    open_models = [
        "Qwen3.5-397B-A17B",
        "Qwen3.5-35B-A3B",
        "KimiK2.5",
        "Qwen3VL-32B-I",
        "InternVL3.5-38B",
    ]
    target_models = prop_models + open_models

    available_models = set(d_all["model"])
    missing = [name for name in target_models if name not in available_models]

    if missing:
        raise ValueError(f"Missing requested figure_15 models: {missing}")

    d_prop = d_all.set_index("model").loc[prop_models].reset_index()
    d_open = d_all.set_index("model").loc[open_models].reset_index()
    d = pd.concat([d_prop, d_open], axis=0).reset_index(drop=True)
    d["model_display"] = d["model"]

    if d.empty:
        return

    compact_scale = 0.72
    fs_bar_text = 8.5 * compact_scale * font_scale
    fs_tick = 12.0 * compact_scale * font_scale
    fs_group = 11.5 * compact_scale * font_scale
    fs_axis = 14.0 * compact_scale * font_scale
    fs_legend = 12.0 * compact_scale * font_scale

    metrics = [
        ("S_acc", "S", "#3B82F6"),
        ("A_acc", "A", "#F59E0B"),
        ("P_acc", "P", "#14B8A6"),
    ]

    labels = [wrap_label(x, width=10) for x in d["model_display"].tolist()]
    n_models = len(d)
    n_prop = len(d_prop)
    n_metrics = len(metrics)

    bar_width = 0.3 * compact_scale
    group_gap = 0.3 * compact_scale
    group_span = n_metrics * bar_width
    group_left = np.arange(n_models) * (group_span + group_gap)
    group_centers = group_left + (n_metrics - 1) * bar_width / 2

    fig, ax = plt.subplots(figsize=(10.45, 2.95))

    x_pad = bar_width * 0.8
    x_left = group_left[0] - group_gap / 2 - group_gap
    x_right = group_left[-1] + group_span + group_gap / 2
    boundary_x = None
    if 0 < n_prop < n_models:
        boundary_x = (group_centers[n_prop - 1] + group_centers[n_prop]) / 2

    for j, (col, title, color) in enumerate(metrics):
        vals = d[col].to_numpy()
        xpos = group_left + j * bar_width
        bar_edge = "black"
        bar_lw = 0.6 * compact_scale
        bars = ax.bar(
            xpos,
            vals,
            width=bar_width,
            color=color,
            alpha=0.9,
            edgecolor=bar_edge,
            linewidth=bar_lw,
            label=title,
        )
        if show_value_labels:
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.009,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=fs_bar_text,
                    fontweight=TEXT_WEIGHT,
                )

    ax.axhline(0.25, ls="--", color="#A67C52", lw=1.0 * compact_scale)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels, rotation=25, ha="center", fontsize=fs_tick, linespacing=0.9, rotation_mode="anchor")
    ax.tick_params(axis="x", pad=19)
    for tick in ax.get_xticklabels():
        tick.set_fontstyle("italic")
        tick.set_fontweight(TEXT_WEIGHT)

    ymax = float(np.nanmax(d[[m[0] for m in metrics]].to_numpy()))
    group_line_y = min(1.03, ymax + 0.075)
    ax.set_ylim(0.0, min(1.04, max(0.35, ymax + 0.075)))
    ax.set_xlim(x_left, x_right)

    if boundary_x is not None:
        ax.axvline(boundary_x, ls="--", color="#666666", lw=1.3 * compact_scale, alpha=0.95)
        # Align the top group guide lines with the visible x-axis span:
        # left axis limit -> vertical separator, and separator -> right axis limit.
        prop_x0 = x_left
        prop_x1 = boundary_x
        open_x0 = boundary_x
        open_x1 = x_right
        _draw_group_label_line(
            ax, prop_x0, prop_x1, group_line_y, "Proprietary", "#2E7D32", fs_group,
            line_width=2.0 * compact_scale, gap_pad=0.014
        )
        _draw_group_label_line(
            ax, open_x0, open_x1, group_line_y, "Open-Source", "#7C3AED", fs_group,
            line_width=2.0 * compact_scale, gap_pad=0.014
        )

    ax.set_ylabel("Accuracy", fontsize=fs_axis, fontstyle="italic", fontweight=TEXT_WEIGHT)
    ax.tick_params(axis="y", labelsize=fs_tick)
    for tick in ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.95),
        fontsize=fs_legend,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#DDDDDD",
        handlelength=1.2 * compact_scale,
        handletextpad=0.4 * compact_scale,
        columnspacing=0.7 * compact_scale,
        labelspacing=0.35 * compact_scale,
        borderpad=0.35 * compact_scale,
        borderaxespad=0.2 * compact_scale,
        prop={"size": fs_legend, "weight": TEXT_WEIGHT},
    )

    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.30, top=0.90)
    save_fig(fig, out_dir, stem)


def write_readme(out_dir: str):
    txt = f"""Temporal Cloze visualization suite (v2)

Generated figures
-----------------
figure_04_consistency_stacked_v2
    Distribution over 3/3, 2/3, 1/3, 0/3 solved dimensions.

figure_04_dataset_consistency_two_models_v2
    Dataset-source consistency distribution for the selected two models.

figure_05_error_source_heatmap_raw_v2
    Raw model failure signatures over subtype errors.

figure_15_top6_metric_bars_no_value_v2
    Selected models shown as grouped bars over S / A / P without bar-value labels.

Generated table
---------------
temporal_cloze_metrics_table_v2.csv
"""
    with open(os.path.join(out_dir, "README_figures_v2.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str, help="Path to analyze_report_*.json")
    parser.add_argument("--out_dir", type=str, default="pdf")
    parser.add_argument(
        "--overall-range",
        type=str,
        default="0.23,0.27",
        help="Filter range for overall acc; models inside [low, high] are dropped (default: 0.23,0.27).",
    )
    parser.add_argument(
        "--disable-acc-filter",
        action="store_true",
        help="Disable filtering by overall acc range.",
    )
    parser.add_argument(
        "--keep-zero-acc",
        action="store_true",
        help="Keep models with overall acc == 0. By default they are removed.",
    )
    parser.add_argument(
        "--fig15-font-scale",
        type=float,
        default=1.0,
        help="Uniform font scaling factor for figure_15 (default: 1.0).",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    report = load_json(args.json_path)
    df_all = build_dataframe(report)
    df = df_all.copy()
    if not args.disable_acc_filter:
        low, high = parse_range(args.overall_range)
        before_n = len(df)
        df = filter_models_by_acc(df, low, high, exclude_zero=not args.keep_zero_acc)
        print(
            f"[Filter] overall acc outside [{low}, {high}], "
            f"exclude_zero={not args.keep_zero_acc}: {before_n} -> {len(df)} models"
        )
        if df.empty:
            raise ValueError("No models left after acc filtering.")
    export_metrics_table(df, args.out_dir)

    plot_consistency_stacked(df, args.out_dir)
    plot_dataset_consistency_for_selected_models(args.out_dir)
    plot_error_source_heatmaps(df, args.out_dir)
    plot_top6_metric_bars(
        df_all,
        args.out_dir,
        font_scale=args.fig15_font_scale,
        show_value_labels=False,
        stem="figure_15_top6_metric_bars_no_value_v2",
    )

    print(f"[OK] Wrote figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
