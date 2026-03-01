"""汇总 eval_results 目录下所有模型结果 JSON，输出准确率表格 + 错误来源分析。"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
EVAL_RESULTS_DIR = ROOT / "eval_results"
D_CHOICES = ROOT / "choices"
DIMS = ["S", "A", "C"]

DISTRACTOR_NAMES = {
    "S": ["Rand1", "Rand2", "Rand3"],
    "A": ["Early", "Late", "Wide"],
    "C": ["Reverse", "Shuffle", "Loop"],
}


def print_table(headers: list[str], rows: list[list[str]], col_widths: list[int]):
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    head = "|" + "|".join(h.center(col_widths[i]) for i, h in enumerate(headers)) + "|"
    print(sep)
    print(head)
    print(sep)
    for cells in rows:
        line = "|" + "|".join(cells[i].ljust(col_widths[i]) for i in range(len(headers))) + "|"
        print(line)
    print(sep)


def main():
    if not EVAL_RESULTS_DIR.exists():
        print(f"Directory not found: {EVAL_RESULTS_DIR}")
        return
    files = sorted(EVAL_RESULTS_DIR.glob("*.json"))
    if not files:
        print(f"No *.json found in {EVAL_RESULTS_DIR}")
        return

    if D_CHOICES.exists():
        num_questions = sum(1 for p in D_CHOICES.iterdir() if p.is_dir())
    else:
        num_questions = 0
    print(f"Total questions (from choices): {num_questions}")

    for path in files:
        model = path.stem
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ---- 1. 准确率统计 ----
        dim_correct = {d: 0 for d in DIMS}
        dim_total = {d: 0 for d in DIMS}
        for stem, entries in data.items():
            for d in DIMS:
                if d not in entries:
                    continue
                dim_total[d] += 1
                if entries[d].get("correct"):
                    dim_correct[d] += 1

        total = sum(dim_total.values())
        correct_all = sum(dim_correct.values())

        right_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        num_stems = 0
        for stem, entries in data.items():
            dims_present = [d for d in DIMS if d in entries]
            if len(dims_present) != 3:
                continue
            num_stems += 1
            n_right = sum(1 for d in dims_present if entries[d].get("correct"))
            right_counts[n_right] += 1

        fmt = lambda v: f"{v:.2%}" if isinstance(v, float) else str(v)

        acc_row = [
            model,
            fmt(dim_correct["S"] / dim_total["S"] if dim_total["S"] else 0),
            fmt(dim_correct["A"] / dim_total["A"] if dim_total["A"] else 0),
            fmt(dim_correct["C"] / dim_total["C"] if dim_total["C"] else 0),
            fmt(correct_all / total if total else 0),
            fmt(right_counts[3] / num_stems if num_stems else 0),
            fmt(right_counts[2] / num_stems if num_stems else 0),
            fmt(right_counts[1] / num_stems if num_stems else 0),
            fmt(right_counts[0] / num_stems if num_stems else 0),
        ]

        headers = ["model", "S acc", "A acc", "C acc", "acc", "3/3", "2/3", "1/3", "0/3"]
        w0 = max(len(model) + 2, 6)
        col_widths = [w0] + [8] * 8
        print_table(headers, [acc_row], col_widths)

        # ---- 2. 错误来源分析（需要 option_map） ----
        error_counts: dict[str, Counter] = {d: Counter() for d in DIMS}
        error_totals: dict[str, int] = {d: 0 for d in DIMS}
        has_option_map = False

        for stem, entries in data.items():
            for d in DIMS:
                if d not in entries:
                    continue
                entry = entries[d]
                if entry.get("correct"):
                    continue
                omap = entry.get("option_map")
                if not omap:
                    continue
                has_option_map = True
                answer = entry.get("answer")
                if answer and answer in omap:
                    chosen = omap[answer]
                    error_counts[d][chosen] += 1
                    error_totals[d] += 1

        if not has_option_map:
            print("  (option_map not available, skipping error source analysis)\n")
            continue

        print(f"\n  Error source analysis ({model}):")
        for d in DIMS:
            n_err = error_totals[d]
            if n_err == 0:
                print(f"    {d}: no errors")
                continue
            names = DISTRACTOR_NAMES[d]
            parts = []
            for name in names:
                cnt = error_counts[d].get(name, 0)
                pct = cnt / n_err if n_err else 0
                parts.append(f"{name}={cnt}({pct:.0%})")
            print(f"    {d} ({n_err} errors): {', '.join(parts)}")
        print()


if __name__ == "__main__":
    main()
