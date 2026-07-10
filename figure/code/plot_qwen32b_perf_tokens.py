#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path

import matplotlib
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "temporal-cloze-matplotlib"),
)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_RESULTS_DIR = REPO_ROOT / "TempCloze" / "eval_results" / "open" / "eval_results"
DEFAULT_INSTRUCT = MODEL_RESULTS_DIR / "Qwen3VL-32B-Instruct.json"
DEFAULT_THINKING = MODEL_RESULTS_DIR / "Qwen3VL-32B-Thinking.json"
DEFAULT_OUT = REPO_ROOT / "figure" / "pics" / "figure_qwen32b_instruct_vs_thinking_perf_tokens.png"

DEFAULT_MODEL_COLORS = [
    "#EEDC77",
    "#90D87B",
]

BASE_FONTSIZE = 15





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruct", type=Path, default=DEFAULT_INSTRUCT)
    parser.add_argument("--thinking", type=Path, default=DEFAULT_THINKING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--instruct-acc", type=float, default=None)
    parser.add_argument("--thinking-acc", type=float, default=None)
    parser.add_argument("--font-scale", type=float, default=1.0)
    return parser.parse_args()


def approx_token_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def load_stats(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    total = 0
    correct = 0
    token_counts = []
    for _, entries in data.items():
        for dim in ("S", "A", "C"):
            if dim not in entries:
                continue
            entry = entries[dim]
            total += 1
            if entry.get("correct"):
                correct += 1
            raw = entry.get("raw_original") or entry.get("raw") or ""
            token_counts.append(approx_token_count(raw))
    if total == 0:
        raise ValueError(f"No valid entries found in {path}")
    acc = 100.0 * correct / total
    avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    return acc, avg_tokens


def token_axis_spec(value: float) -> tuple[float, np.ndarray]:
    if value < 1000:
        upper = 1000.0
        ticks = np.arange(0, 1000 + 1, 200, dtype=float)
        return upper, ticks

    target = value * 1.25
    magnitude = 10 ** math.floor(math.log10(target))
    upper = None
    for factor in (1, 2, 2.5, 5, 10):
        candidate = factor * magnitude
        if candidate >= target:
            upper = float(candidate)
            break
    if upper is None:
        upper = float(10 * magnitude)
    ticks = np.linspace(0, upper, 6)
    return upper, ticks


def format_token_tick(value: float) -> str:
    if abs(value) >= 1000:
        k_value = value / 1000.0
        return f"{k_value:.1f}k" if abs(k_value - round(k_value)) > 1e-9 else f"{int(round(k_value))}k"
    return f"{value:.0f}"


def plot(models, accs, avg_tokens, out_path: Path, font_scale: float = 1.0):
    fs = BASE_FONTSIZE * font_scale
    scale = 0.1
    groups = np.array([0.0, 0.26 * scale])

    width = 0.12 * scale

    colors = [DEFAULT_MODEL_COLORS[i % len(DEFAULT_MODEL_COLORS)] for i, _ in enumerate(models)]

    fig, ax1 = plt.subplots(figsize=(8.6, 4.6))


    ax2 = ax1.twinx()

    acc_x = groups[0] + np.array([-width / 2, width / 2])
    tok_x = groups[1] + np.array([-width / 2, width / 2])

    bars_acc = []
    bars_tok = []
    for i, _model in enumerate(models):
        bars_acc.append(
            ax1.bar(
                acc_x[i],
                accs[i],
                width=width,
                color=colors[i],
                alpha=0.95,
                edgecolor="none",
linewidth=0,
            )[0]
        )
        bars_tok.append(
            ax2.bar(
                tok_x[i],
                avg_tokens[i],
                width=width,
                color=colors[i],
                alpha=0.95,
                edgecolor="none",
linewidth=0,
            )[0]
        )

    inter_group_gap = max(0.0, (groups[1] - groups[0]) - 2 * width)
    left_edge = groups[0] - width
    right_edge = groups[1] + width
    ax1.set_xlim(left_edge - inter_group_gap, right_edge + inter_group_gap)


    ax1.set_xticks(groups)
    ax1.set_xticklabels(["Accuracy", "#Response Tokens"], fontsize=fs)
    ax1.tick_params(axis="y", labelsize=fs)
    ax2.tick_params(axis="y", labelsize=fs)


    ax1.set_ylim(0, 100)
    ax1.set_yticks([0, 25, 50, 75, 100])

    max_token_value = max(avg_tokens) if avg_tokens else 0.0
    ax2_max, ax2_ticks = token_axis_spec(max_token_value)
    ax2.set_ylim(0, ax2_max)
    ax2.set_yticks(ax2_ticks)
    ax2.set_yticklabels([format_token_tick(x) for x in ax2_ticks])



    for bar, v in zip(bars_acc, accs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.2,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=fs*0.88
        )
    for bar, v in zip(bars_tok, avg_tokens):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(avg_tokens) * 0.02,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=fs*0.88
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[0], edgecolor="black", linewidth=0.7),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[1], edgecolor="black", linewidth=0.7),
    ]
    ax1.legend(handles, models, loc="lower left", bbox_to_anchor=(0.0, 1.05, 1.0, 0.1), ncol=2, mode="expand", borderaxespad=0.0, frameon=False, fontsize=fs * 0.78, columnspacing=0.0, handletextpad=0.5, prop={"weight": "bold", "size": fs * 0.78})



    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)



def main():
    args = parse_args()
    acc_i, tok_i = load_stats(args.instruct)
    acc_t, tok_t = load_stats(args.thinking)

    if args.instruct_acc is not None:
        acc_i = args.instruct_acc
    if args.thinking_acc is not None:
        acc_t = args.thinking_acc

    models = [args.instruct.stem, args.thinking.stem]
    accs = [acc_i, acc_t]
    toks = [tok_i, tok_t]
    plot(models, accs, toks, args.out, font_scale=args.font_scale)

    print(f"Saved: {args.out}")
    print(f"{models[0]}: acc={acc_i:.2f}%, avg_tokens={tok_i:.1f}")
    print(f"{models[1]}: acc={acc_t:.2f}%, avg_tokens={tok_t:.1f}")


if __name__ == "__main__":
    main()
