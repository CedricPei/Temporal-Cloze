#!/usr/bin/env python3
"""Plot one radar chart from analyze_report JSON.

Chart design:
- One radar figure only
- Axes/vertices: models
- Series: metrics (default: S_acc, A_acc, C_acc, acc)

Examples:
  python plot_radar.py \
    --report video-cloze/analyze_report_final.json \
    --out video-cloze/radar_single.png

  python plot_radar.py \
    --report video-cloze/analyze_report_final.json \
    --topk 10 \
    --sort-by acc
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Avoid matplotlib cache-dir permission issues in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_METRICS = ["S_acc", "A_acc", "C_acc", "acc"]
PRETTY_METRIC = {
    "S_acc": "S",
    "A_acc": "A",
    "C_acc": "C",
    "acc": "Overall",
}


def _simplify_model_name(name: str) -> str:
    """Drop prefix before the first '_' for display."""
    if "_" not in name:
        return name
    return name.split("_", 1)[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one radar chart (model vertices, metric series)")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("video-cloze/analyze_report_final.json"),
        help="Path to analyze_report JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("video-cloze/radar_single.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric axes in radar chart",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="acc",
        help="Metric used to rank models when --models is not provided",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Top-k models to plot when --models is not provided; 0 means all",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names to plot. If set, --topk is ignored.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Temporal-Cloze Radar (Model Vertices)",
        help="Figure title",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="petroff10",
        help="Matplotlib style name. Defaults to petroff10; falls back silently if unavailable.",
    )
    parser.add_argument(
        "--only-overall-outside-range",
        action="store_true",
        help="If enabled, keep only models with overall acc outside the range in --overall-range.",
    )
    parser.add_argument(
        "--overall-range",
        type=str,
        default="0.23,0.27",
        help="Inclusive range for overall acc filtering, format: low,high (default: 0.23,0.27).",
    )
    parser.add_argument(
        "--radial-ticks",
        type=str,
        default="0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated radial tick values in [0,1], e.g. 0.05,0.2,0.6,0.8,1",
    )
    parser.add_argument(
        "--radial-scale",
        type=str,
        default="linear",
        choices=["linear", "equalized"],
        help="Radial scaling mode: linear (default) or equalized (piecewise mapping with equal visual intervals).",
    )
    return parser.parse_args()


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Metric value is not numeric: {value!r}")


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "models" not in data or not isinstance(data["models"], dict):
        raise ValueError(f"Invalid report format: missing dict key 'models' in {path}")
    return data


def select_models(models: dict, metrics: list[str], sort_by: str, topk: int, models_arg: str) -> list[tuple[str, dict]]:
    selected: list[tuple[str, dict]] = []

    if models_arg.strip():
        wanted = [m.strip() for m in models_arg.split(",") if m.strip()]
        for name in wanted:
            if name not in models:
                raise ValueError(f"Model not found: {name}")
            payload = models[name]
            if any(metric not in payload for metric in metrics):
                raise ValueError(f"Model {name} missing one of metrics: {metrics}")
            selected.append((name, payload))
    else:
        if sort_by not in metrics and sort_by != "acc":
            raise ValueError(f"--sort-by {sort_by} is not in metrics and not 'acc'")

        sortable = []
        for name, payload in models.items():
            if any(metric not in payload for metric in metrics):
                continue
            if sort_by not in payload:
                continue
            sortable.append((name, payload, _to_float(payload[sort_by])))
        sortable.sort(key=lambda x: x[2], reverse=True)
        if topk > 0:
            sortable = sortable[:topk]
        selected = [(name, payload) for name, payload, _ in sortable]

    if not selected:
        raise ValueError("No model selected for plotting")
    return selected


def _metric_label(metric: str) -> str:
    return PRETTY_METRIC.get(metric, metric)


def _parse_range(s: str) -> tuple[float, float]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid --overall-range: {s}. Expected format: low,high")
    low = float(parts[0])
    high = float(parts[1])
    if low > high:
        raise ValueError(f"Invalid --overall-range: low ({low}) cannot be greater than high ({high})")
    return low, high


def _filter_by_overall_outside_range(
    selected: list[tuple[str, dict]], low: float, high: float
) -> list[tuple[str, dict]]:
    filtered: list[tuple[str, dict]] = []
    for name, payload in selected:
        acc = _to_float(payload.get("acc"))
        if acc == 0.0:
            continue
        if acc < low or acc > high:
            filtered.append((name, payload))
    return filtered


def _parse_ticks(s: str) -> list[float]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("--radial-ticks cannot be empty")
    ticks = [float(x) for x in vals]
    if ticks != sorted(ticks):
        raise ValueError("--radial-ticks must be in ascending order")
    if ticks[0] < 0 or ticks[-1] > 1:
        raise ValueError("--radial-ticks values must be in [0,1]")
    return ticks


def _build_radial_mapper(radial_ticks: list[float], radial_scale: str):
    """Return (transform_fn, tick_positions).

    - linear: identity transform.
    - equalized: piecewise-linear transform where intervals between knots are
      equally spaced in display radius, which helps with long-tail compression.
    """
    if radial_scale == "linear":
        return (lambda x: x), radial_ticks

    knots = [0.0] + radial_ticks
    if knots[-1] < 1.0:
        knots.append(1.0)
    # unique + sorted while preserving valid numeric order
    knots = sorted(set(knots))
    knot_pos = np.linspace(0.0, 1.0, len(knots))

    def _transform(x: float) -> float:
        xx = min(max(float(x), 0.0), 1.0)
        return float(np.interp(xx, knots, knot_pos))

    tick_pos = [float(np.interp(t, knots, knot_pos)) for t in radial_ticks]
    return _transform, tick_pos


def plot_radar(
    selected: list[tuple[str, dict]],
    metrics: list[str],
    out_path: Path,
    title: str,
    style: str,
    radial_ticks: list[float],
    radial_scale: str,
) -> None:
    try:
        plt.style.use(style)
    except OSError:
        pass

    labels = [_simplify_model_name(name) for name, _ in selected]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig_size = max(10.0, min(22.0, 8.0 + n * 0.28))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    x_font = 10 if n > 24 else 11 if n > 16 else 12
    ax.set_xticklabels(labels, fontsize=x_font)

    radial_transform, radial_tick_positions = _build_radial_mapper(radial_ticks, radial_scale)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(radial_tick_positions)
    ax.set_yticklabels([f"{x:g}" for x in radial_ticks], fontsize=13)
    ax.grid(alpha=0.35, linewidth=0.9)

    colors = plt.rcParams.get("axes.prop_cycle", None)
    palette = colors.by_key().get("color", []) if colors is not None else []
    do_fill = len(metrics) <= 4

    for idx, metric in enumerate(metrics):
        values = [radial_transform(_to_float(payload[metric])) for _, payload in selected]
        values += values[:1]
        color = palette[idx % len(palette)] if palette else None
        ax.plot(
            angles,
            values,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=_metric_label(metric),
            color=color,
            alpha=0.95,
        )
        if do_fill:
            ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_title(title, pad=24, fontsize=15)
    ncol = 1 if len(metrics) <= 4 else 2
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.08, 0.5),
        frameon=False,
        fontsize=9,
        ncol=ncol,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    radial_ticks = _parse_ticks(args.radial_ticks)
    if len(metrics) < 3:
        raise ValueError("At least 3 metrics are recommended for radar chart")

    report = load_report(args.report)
    selected = select_models(
        models=report["models"],
        metrics=metrics,
        sort_by=args.sort_by,
        topk=args.topk,
        models_arg=args.models,
    )

    if args.only_overall_outside_range:
        low, high = _parse_range(args.overall_range)
        selected = _filter_by_overall_outside_range(selected, low, high)
        if not selected:
            raise ValueError(
                f"No model left after filtering overall acc outside [{low}, {high}]"
            )

    plot_radar(
        selected=selected,
        metrics=metrics,
        out_path=args.out,
        title=args.title,
        style=args.style,
        radial_ticks=radial_ticks,
        radial_scale=args.radial_scale,
    )

    print(f"Saved: {args.out}")
    print(f"Selected models (vertices): {len(selected)}")
    print("Metric series: " + ", ".join([_metric_label(m) for m in metrics]))
    if args.only_overall_outside_range:
        low, high = _parse_range(args.overall_range)
        print(f"Filter: keep only overall acc outside [{low}, {high}]")
        print("Filter: exclude models with overall acc == 0")
    print(f"Radial scale: {args.radial_scale}")


if __name__ == "__main__":
    main()
