#!/usr/bin/env python3
"""Summarize Temporal-Cloze dataset construction filtering statistics.

The script reads the status ledgers under output/<source>/:
  - rejected.json: rejected or pruned samples with reason strings
  - meta.json: samples that passed gap sampling and were generated
  - llm_filter.json: samples judged by the LLM filter

It produces a canonicalized pipeline table. Some sources run quality before o3
and others run o3 before quality; the counts are therefore best interpreted as
stage-level attribution from the saved logs rather than a replay of wall-clock
execution order.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_STAGES = [
    ("duration_filtering", "Duration filtering"),
    ("o3_filtering", "GPT-o3 filtering"),
    ("acquisition_failures", "Download/local validation failures"),
    ("quality_filtering", "Quality filtering"),
    ("optical_flow_filtering", "Optical-flow filtering"),
    ("manual_pruning", "Manual pruning"),
]

PAPER_STAGES = [
    ("duration_filtering", "duration"),
    ("o3_filtering", "o3"),
    ("quality_filtering", "quality"),
    ("optical_flow_filtering", "optical_flow"),
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def reason_prefix(reason: Any) -> str:
    text = str(reason)
    if text.startswith("LLM_REJECT"):
        return "LLM_REJECT"
    if text.startswith("DURATION_OUT_OF_RANGE"):
        return "DURATION_OUT_OF_RANGE"
    if text.startswith("DURATION_PARSE_ERROR"):
        return "DURATION_PARSE_ERROR"
    return text.split(":", 1)[0].split("(", 1)[0].strip()


def stage_for_reason(reason: Any) -> str:
    prefix = reason_prefix(reason)
    if prefix in {"DURATION_OUT_OF_RANGE", "DURATION_PARSE_ERROR"}:
        return "duration_filtering"
    if prefix == "LLM_REJECT":
        return "o3_filtering"
    if prefix in {
        "DOWNLOAD_FAILED",
        "VIDEO_NOT_FOUND",
        "CAPTION_GENERATION_FAILED",
        "TRANSCODE_FAILED",
    }:
        return "acquisition_failures"
    if prefix in {"QUALITY_LOW", "FILTER_QUALITY_LOW"}:
        return "quality_filtering"
    if prefix in {"GAP_REJECT", "After 3 tries"}:
        return "optical_flow_filtering"
    if prefix == "PRUNED":
        return "manual_pruning"
    return "other_rejections"


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def source_stats(source_dir: Path) -> dict[str, Any]:
    rejected = load_json(source_dir / "rejected.json")
    meta = load_json(source_dir / "meta.json")
    llm_filter = load_json(source_dir / "llm_filter.json")

    all_names = set(rejected) | set(meta) | set(llm_filter)
    stage_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for reason in rejected.values():
        stage_counts[stage_for_reason(reason)] += 1
        reason_counts[reason_prefix(reason)] += 1

    removed_known = sum(stage_counts.values())
    final_kept = len(meta)
    pending_or_unaccounted = max(0, len(all_names) - removed_known - final_kept)

    return {
        "source": source_dir.name,
        "initial": len(all_names),
        "final_kept": final_kept,
        "pending_or_unaccounted": pending_or_unaccounted,
        "stage_counts": stage_counts,
        "reason_counts": reason_counts,
        "llm_filter_total": len(llm_filter),
        "llm_filter_pass": sum(
            1 for v in llm_filter.values()
            if isinstance(v, dict) and bool(v.get("pass"))
        ),
    }


def build_rows(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = {
        "source": "overall",
        "initial": sum(s["initial"] for s in stats),
        "final_kept": sum(s["final_kept"] for s in stats),
        "pending_or_unaccounted": sum(s["pending_or_unaccounted"] for s in stats),
        "stage_counts": Counter(),
        "reason_counts": Counter(),
        "llm_filter_total": sum(s["llm_filter_total"] for s in stats),
        "llm_filter_pass": sum(s["llm_filter_pass"] for s in stats),
    }
    for s in stats:
        total["stage_counts"].update(s["stage_counts"])
        total["reason_counts"].update(s["reason_counts"])

    for s in [*stats, total]:
        initial = s["initial"]
        stage_input = initial
        for stage_key, stage_name in DEFAULT_STAGES:
            removed = int(s["stage_counts"].get(stage_key, 0))
            kept_after = stage_input - removed
            rows.append({
                "source": s["source"],
                "stage": stage_name,
                "stage_key": stage_key,
                "input": stage_input,
                "removed": removed,
                "kept_after_stage": kept_after,
                "removed_pct_of_stage_input": pct(removed, stage_input),
                "removed_pct_of_initial": pct(removed, initial),
            })
            stage_input = kept_after

        pending = int(s["pending_or_unaccounted"])
        rows.append({
            "source": s["source"],
            "stage": "Pending/unaccounted in logs",
            "stage_key": "pending_or_unaccounted",
            "input": stage_input,
            "removed": pending,
            "kept_after_stage": stage_input - pending,
            "removed_pct_of_stage_input": pct(pending, stage_input),
            "removed_pct_of_initial": pct(pending, initial),
        })
        stage_input -= pending

        final_kept = int(s["final_kept"])
        rows.append({
            "source": s["source"],
            "stage": "Final kept",
            "stage_key": "final_kept",
            "input": stage_input,
            "removed": 0,
            "kept_after_stage": final_kept,
            "removed_pct_of_stage_input": 0.0,
            "removed_pct_of_initial": 0.0,
        })
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "stage",
        "stage_key",
        "input",
        "removed",
        "kept_after_stage",
        "removed_pct_of_stage_input",
        "removed_pct_of_initial",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["removed_pct_of_stage_input"] = f"{row['removed_pct_of_stage_input']:.2f}"
            out["removed_pct_of_initial"] = f"{row['removed_pct_of_initial']:.2f}"
            writer.writerow(out)


def build_wide_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(row["source"], {})[row["stage_key"]] = row

    ordered_sources = [s for s in by_source if s != "overall"]
    if "overall" in by_source:
        ordered_sources.append("overall")

    wide_rows: list[dict[str, Any]] = []
    for source in ordered_sources:
        source_rows = by_source[source]
        initial = int(source_rows["duration_filtering"]["input"])
        final_kept = int(source_rows["final_kept"]["kept_after_stage"])
        wide: dict[str, Any] = {
            "source": source,
            "input_n": initial,
        }
        for stage_key, short_name in PAPER_STAGES:
            stage = source_rows[stage_key]
            wide[f"{short_name}_removed"] = int(stage["removed"])
            wide[f"{short_name}_removed_pct"] = float(stage["removed_pct_of_stage_input"])
        wide["total_filtered_out_pct"] = pct(initial - final_kept, initial)
        wide_rows.append(wide)
    return wide_rows


def write_wide_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "input_n"]
    for _, short_name in PAPER_STAGES:
        fieldnames.extend([f"{short_name}_removed", f"{short_name}_removed_pct"])
    fieldnames.append("total_filtered_out_pct")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in list(out.items()):
                if key.endswith("_pct"):
                    out[key] = f"{value:.2f}"
            writer.writerow(out)


def print_markdown(rows: list[dict[str, Any]], sources: set[str] | None = None) -> None:
    selected = rows if sources is None else [r for r in rows if r["source"] in sources]
    print("| source | stage | input | removed | kept after | removed % of stage | removed % of initial |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in selected:
        print(
            f"| {r['source']} | {r['stage']} | {r['input']} | {r['removed']} | "
            f"{r['kept_after_stage']} | {r['removed_pct_of_stage_input']:.2f}% | "
            f"{r['removed_pct_of_initial']:.2f}% |"
        )


def print_wide_markdown(rows: list[dict[str, Any]], sources: set[str] | None = None) -> None:
    selected = rows if sources is None else [r for r in rows if r["source"] in sources]
    headers = ["source", "input"]
    for _, short_name in PAPER_STAGES:
        headers.extend([f"{short_name} removed", f"{short_name} %"])
    headers.append("total filtered out %")

    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for row in selected:
        cells = [row["source"], str(row["input_n"])]
        for _, short_name in PAPER_STAGES:
            cells.append(str(row[f"{short_name}_removed"]))
            cells.append(f"{row[f'{short_name}_removed_pct']:.2f}%")
        cells.append(f"{row['total_filtered_out_pct']:.2f}%")
        print("| " + " | ".join(cells) + " |")


def print_reason_summary(stats: list[dict[str, Any]]) -> None:
    print("\nReason summary:")
    for s in stats:
        parts = ", ".join(f"{k}={v}" for k, v in s["reason_counts"].most_common())
        print(f"- {s['source']}: {parts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dataset-construction pipeline filtering table."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Directory containing output/<source> status JSON files.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional source names to include, e.g. lvd tt favor.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("TempCloze/pipeline_filter_table.csv"),
        help="Long-form CSV output path.",
    )
    parser.add_argument(
        "--wide-csv",
        type=Path,
        default=Path("TempCloze/pipeline_filter_summary.csv"),
        help="Paper-friendly wide CSV output path.",
    )
    parser.add_argument(
        "--print-sources",
        nargs="*",
        default=["overall"],
        help="Sources to print as Markdown. Use 'all' to print every source.",
    )
    parser.add_argument(
        "--print-long",
        action="store_true",
        help="Print the long-form table instead of the paper-friendly wide table.",
    )
    parser.add_argument(
        "--reasons",
        action="store_true",
        help="Print raw rejected reason prefix counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dirs = [
        p for p in sorted(args.output_root.iterdir())
        if p.is_dir() and (p / "rejected.json").exists()
    ]
    if args.sources:
        wanted = set(args.sources)
        source_dirs = [p for p in source_dirs if p.name in wanted]
    if not source_dirs:
        raise SystemExit(f"No source status directories found under {args.output_root}")

    stats = [source_stats(p) for p in source_dirs]
    rows = build_rows(stats)
    wide_rows = build_wide_rows(rows)
    write_csv(rows, args.csv)
    write_wide_csv(wide_rows, args.wide_csv)

    print(f"Wrote {args.csv}")
    print(f"Wrote {args.wide_csv}")
    if args.print_sources == ["all"]:
        selected_sources = None
    else:
        selected_sources = set(args.print_sources)
    if args.print_long:
        print_markdown(rows, selected_sources)
    else:
        print_wide_markdown(wide_rows, selected_sources)
    if args.reasons:
        print_reason_summary(stats)


if __name__ == "__main__":
    main()
