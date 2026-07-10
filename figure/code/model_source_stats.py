#!/usr/bin/env python3
"""Compute by-source accuracy tables for selected Temporal-Cloze models.

Outputs one CSV table per model under figure/pics by default. Each table reports
dimension accuracy (S/A/P) and cumulative per-video accuracy (>=1, >=2, >=3).
The code uses output/<source>/meta.json to map each evaluated stem back to its
source dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_OUT_DIR = REPO_ROOT / "figure" / "pics"

DIMS = ("S", "A", "C")
DIM_LABEL = {"S": "S", "A": "A", "C": "P"}

DEFAULT_MODELS = {
    "Gemini2.5-Pro": REPO_ROOT / "TempCloze" / "eval_results" / "closed" / "eval_results" / "gemini-2.5-pro.json",
    "Qwen3.5-35B-A3B": REPO_ROOT / "TempCloze" / "eval_results" / "closed" / "eval_results" / "qwen3.5-35b-a3b.json",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def build_source_map(output_root: Path) -> dict[str, str]:
    stem_to_source: dict[str, str] = {}
    conflicts: dict[str, list[str]] = defaultdict(list)

    for source_dir in sorted(output_root.iterdir()):
        meta_path = source_dir / "meta.json"
        if not source_dir.is_dir() or not meta_path.exists():
            continue
        for filename in load_json(meta_path):
            stem = Path(filename).stem
            if stem in stem_to_source:
                conflicts[stem].extend([stem_to_source[stem], source_dir.name])
            stem_to_source[stem] = source_dir.name

    if conflicts:
        examples = ", ".join(f"{stem}: {sorted(set(srcs))}" for stem, srcs in list(conflicts.items())[:5])
        raise ValueError(f"Stems mapped to multiple sources: {examples}")
    return stem_to_source


def is_valid_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and isinstance(entry.get("correct"), bool)


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def source_sort_key(source: str) -> tuple[int, str]:
    return (1, source) if source == "overall" else (0, source)


def analyze_model_by_source(
    model_data: dict[str, Any],
    stem_to_source: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "dim_correct": {dim: 0 for dim in DIMS},
            "dim_total": {dim: 0 for dim in DIMS},
            "videos": [],
        }
    )

    missing_sources: list[str] = []
    for stem, entries in model_data.items():
        source = stem_to_source.get(stem)
        if source is None:
            missing_sources.append(stem)
            continue
        if not isinstance(entries, dict):
            continue

        per_video: dict[str, bool] = {}
        for dim in DIMS:
            entry = entries.get(dim)
            if not is_valid_entry(entry):
                continue
            correct = bool(entry["correct"])
            grouped[source]["dim_total"][dim] += 1
            grouped[source]["dim_correct"][dim] += int(correct)
            per_video[dim] = correct

        if all(dim in per_video for dim in DIMS):
            grouped[source]["videos"].append(per_video)

    if missing_sources:
        sample = ", ".join(missing_sources[:10])
        raise ValueError(f"{len(missing_sources)} stems missing source mapping, e.g. {sample}")

    overall = {
        "dim_correct": {dim: 0 for dim in DIMS},
        "dim_total": {dim: 0 for dim in DIMS},
        "videos": [],
    }
    for stats in grouped.values():
        for dim in DIMS:
            overall["dim_correct"][dim] += stats["dim_correct"][dim]
            overall["dim_total"][dim] += stats["dim_total"][dim]
        overall["videos"].extend(stats["videos"])
    grouped["overall"] = overall

    rows: list[dict[str, Any]] = []
    for source in sorted(grouped, key=source_sort_key):
        stats = grouped[source]
        videos = stats["videos"]
        row: dict[str, Any] = {
            "source": source,
            "n_videos": len(videos),
        }
        for dim in DIMS:
            row[DIM_LABEL[dim]] = pct(stats["dim_correct"][dim], stats["dim_total"][dim])

        correct_counts = [sum(int(video[dim]) for dim in DIMS) for video in videos]
        for k in (1, 2, 3):
            row[f">={k}"] = pct(sum(count >= k for count in correct_counts), len(correct_counts))

        rows.append(row)
    return rows


def safe_filename(model_name: str) -> str:
    out = []
    for ch in model_name.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {".", "-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("._-")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "S", "A", "P", ">=1", ">=2", ">=3"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row[key] for key in fieldnames}
            for key in ("S", "A", "P", ">=1", ">=2", ">=3"):
                out[key] = f"{out[key]:.2f}"
            writer.writerow(out)


def print_markdown(model_name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{model_name}")
    print("| source | S | A | P | >=1 | >=2 | >=3 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['source']} | "
            f"{row['S']:.2f} | {row['A']:.2f} | {row['P']:.2f} | "
            f"{row['>=1']:.2f} | {row['>=2']:.2f} | {row['>=3']:.2f} |"
        )


def parse_model_args(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_MODELS)

    models: dict[str, Path] = {}
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
            models[name.strip()] = Path(path).expanduser()
        else:
            path = Path(value).expanduser()
            models[path.stem] = path
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create by-source S/A/P and cumulative accuracy tables."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model specs as Name=/path/result.json or just /path/result.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory containing output/<source>/meta.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for per-model CSV tables.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Write CSVs without printing Markdown tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stem_to_source = build_source_map(args.output_root)
    models = parse_model_args(args.models)

    for model_name, path in models.items():
        if not path.exists():
            raise FileNotFoundError(path)
        rows = analyze_model_by_source(load_json(path), stem_to_source)
        out_path = args.out_dir / f"source_stats_{safe_filename(model_name)}.csv"
        write_csv(rows, out_path)
        print(f"Wrote {out_path}")
        if not args.no_print:
            print_markdown(model_name, rows)


if __name__ == "__main__":
    main()
