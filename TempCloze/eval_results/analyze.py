"""汇总 eval_results 下所有模型结果，输出准确率表格 + 错误来源分析，并写入 JSON 报告。

用法:
  python analyze.py closed       # 分析闭源模型 (closed/eval_results/)
  python analyze.py open         # 分析开源模型 (open/eval_results/)
  python analyze.py all          # 合并分析所有模型
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent          # = video-cloze/eval_results/
CHOICES_DIR = ROOT.parent / "choices" # = video-cloze/choices/
EVAL_BASE = ROOT                       # = video-cloze/eval_results/
DIMS = ["S", "A", "C"]
VALID_ANSWERS = {"A", "B", "C", "D"}

DISTRACTOR_NAMES = {
    "A": ["Early", "Late", "Wide"],
    "C": ["Reverse", "Shuffle", "Loop"],
}

SCOPE_MAP = {
    "closed": [EVAL_BASE / "closed" / "eval_results"],
    "open":   [EVAL_BASE / "open" / "eval_results"],
    "all":    [EVAL_BASE / "closed" / "eval_results",
               EVAL_BASE / "open" / "eval_results"],
}


def collect_json_files(scope: str) -> list[Path]:
    dirs = SCOPE_MAP.get(scope, [])
    files = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.json")))
    return files


def is_valid(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    ans = entry.get("answer")
    return isinstance(ans, str) and ans.strip().upper() in VALID_ANSWERS


def print_table(headers: list[str], rows: list[list[str]], col_widths: list[int]):
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    fmt_row = lambda cells: "|" + "|".join(
        cells[i].center(col_widths[i]) if i == 0 else cells[i].ljust(col_widths[i])
        for i in range(len(headers))
    ) + "|"
    print(sep)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)


def pct(num: int, den: int) -> str:
    return f"{num / den:.2%}" if den else "N/A"


def analyze_model(model: str, data: dict) -> dict:
    dim_correct = {d: 0 for d in DIMS}
    dim_total = {d: 0 for d in DIMS}
    skipped = 0

    for entries in data.values():
        for d in DIMS:
            if d not in entries:
                continue
            if not is_valid(entries[d]):
                skipped += 1
                continue
            dim_total[d] += 1
            if entries[d].get("correct"):
                dim_correct[d] += 1

    total = sum(dim_total.values())
    correct_all = sum(dim_correct.values())

    # 联合准确率：S+A, S+C, A+C, S+A+C（仅 S/A/C 三维都有效的 stem）
    joint_groups = [("S", "A"), ("S", "C"), ("A", "C"), ("S", "A", "C")]
    joint_correct = {g: 0 for g in joint_groups}
    num_stems = 0
    for entries in data.values():
        valid_dims = [d for d in DIMS if d in entries and is_valid(entries[d])]
        if len(valid_dims) != 3:
            continue
        num_stems += 1
        for g in joint_groups:
            if all(entries[d].get("correct") for d in g):
                joint_correct[g] += 1

    joint_label = {("S","A"): "S+A", ("S","C"): "S+C",
                   ("A","C"): "A+C", ("S","A","C"): "S+A+C"}

    # 打印表格
    row = [model] + [pct(dim_correct[d], dim_total[d]) for d in DIMS] + [
        pct(correct_all, total),
        *(pct(joint_correct[g], num_stems) for g in joint_groups),
    ]
    headers = ["model", "S acc", "A acc", "C acc", "acc",
               *(joint_label[g] for g in joint_groups)]
    w0 = max(len(model) + 2, 8)
    print_table(headers, [row], [w0] + [8] * (len(headers) - 1))
    if skipped:
        print(f"  (skipped {skipped} invalid entries)")

    # 错误来源分析
    error_report = {}
    for d in ("A", "C"):
        counts = Counter()
        for entries in data.values():
            if d not in entries:
                continue
            e = entries[d]
            if not is_valid(e) or e.get("correct"):
                continue
            omap = e.get("option_map", {})
            ans = e.get("answer")
            if ans and ans in omap:
                counts[omap[ans]] += 1
        n_err = sum(counts.values())
        if n_err:
            parts = [f"{name}={counts.get(name, 0)}({pct(counts.get(name, 0), n_err)})"
                     for name in DISTRACTOR_NAMES[d]]
            print(f"  {d} errors ({n_err}): {', '.join(parts)}")
            error_report[d] = {
                "total": n_err,
                **{name: {"count": counts.get(name, 0),
                          "pct": round(counts.get(name, 0) / n_err, 4)}
                   for name in DISTRACTOR_NAMES[d]},
            }
    print()

    report = {
        "num_stems": num_stems, "skipped": skipped,
        "S_acc": dim_correct["S"] / dim_total["S"] if dim_total["S"] else 0,
        "A_acc": dim_correct["A"] / dim_total["A"] if dim_total["A"] else 0,
        "C_acc": dim_correct["C"] / dim_total["C"] if dim_total["C"] else 0,
        "acc": correct_all / total if total else 0,
        **{joint_label[g]: (joint_correct[g] / num_stems if num_stems else 0)
           for g in joint_groups},
    }
    if error_report:
        report["error_source"] = error_report
    return report


def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "all"
    if scope not in SCOPE_MAP:
        print(f"Unknown scope: {scope}, use 'closed', 'open', or 'all'")
        sys.exit(1)

    files = collect_json_files(scope)
    if not files:
        print(f"No JSON files found for scope '{scope}'")
        return

    num_questions = sum(1 for p in CHOICES_DIR.iterdir() if p.is_dir()) if CHOICES_DIR.exists() else 0
    print(f"Scope: {scope} | Total questions: {num_questions} | Models: {len(files)}\n")

    report = {"total_questions": num_questions, "scope": scope, "models": {}}
    for path in files:
        model = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        report["models"][model] = analyze_model(model, data)

    if scope == "all":
        report_path = EVAL_BASE / f"analyze_report_{scope}.json"
    else:
        report_path = EVAL_BASE / scope / f"analyze_report_{scope}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
