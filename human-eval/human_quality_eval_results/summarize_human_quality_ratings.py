#!/usr/bin/env python3
"""Summarize human quality eval rating counts and averages.

Run from this folder:
    python3 summarize_human_quality_ratings.py

Or from the repo root:
    python3 human-eval/human_quality_eval_results/summarize_human_quality_ratings.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


RATINGS = (1, 2, 3, 4, 5)


def load_ratings(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    responses = data.get("responses", {})
    if isinstance(responses, dict):
        response_items = responses.values()
    elif isinstance(responses, list):
        response_items = responses
    else:
        raise ValueError(f"{path.name}: responses must be a dict or list")

    ratings: list[int] = []
    for response in response_items:
        if not isinstance(response, dict) or "rating" not in response:
            continue

        rating = int(response["rating"])
        if rating not in RATINGS:
            raise ValueError(f"{path.name}: invalid rating {rating}")
        ratings.append(rating)

    return ratings


def summarize_file(path: Path) -> dict[str, object]:
    ratings = load_ratings(path)
    counts = Counter(ratings)
    total = len(ratings)
    average = sum(ratings) / total if total else 0.0

    row: dict[str, object] = {"evaluator": path.stem}
    for rating in RATINGS:
        row[f"rating_{rating}"] = counts[rating]
    row["total"] = total
    row["average"] = average
    return row


def print_table(rows: list[dict[str, object]]) -> None:
    headers = ["evaluator", "rating_1", "rating_2", "rating_3", "rating_4", "rating_5", "total", "average"]
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                **row,
                "average": f"{float(row['average']):.2f}",
            }
        )

    widths = {
        header: max(len(header), *(len(str(row[header])) for row in display_rows))
        for header in headers
    }

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in display_rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def build_overall_row(rows: list[dict[str, object]]) -> dict[str, object]:
    overall: dict[str, object] = {"evaluator": "OVERALL"}
    total_score = 0
    total_count = 0

    for rating in RATINGS:
        count = sum(int(row[f"rating_{rating}"]) for row in rows)
        overall[f"rating_{rating}"] = count
        total_score += rating * count
        total_count += count

    overall["total"] = total_count
    overall["average"] = total_score / total_count if total_count else 0.0
    return overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count 1-5 ratings and compute averages for human quality eval result JSON files."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing human_quality_eval_set_*.json files.",
    )
    parser.add_argument(
        "--pattern",
        default="human_quality_eval_set_*.json",
        help="Glob pattern for result JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.dir.glob(args.pattern))
    paths = [path for path in paths if path.is_file()]

    if not paths:
        raise SystemExit(f"No files found: {args.dir / args.pattern}")

    rows = [summarize_file(path) for path in paths]
    rows.append(build_overall_row(rows))

    print_table(rows)


if __name__ == "__main__":
    main()
