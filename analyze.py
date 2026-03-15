"""汇总 eval_results 目录下所有模型结果 JSON，输出准确率表格 + 错误来源分析（A/C），并写入 JSON 报告。

用法:
  python analyze.py video-cloze   # 分析 video-cloze 文件夹
  python analyze.py subset        # 分析 subset 文件夹
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
PRESET = sys.argv[1] if len(sys.argv) > 1 else "video-cloze"
PRESET_DIR = ROOT / PRESET
EVAL_RESULTS_DIR = PRESET_DIR / "eval_results"
D_CHOICES = PRESET_DIR / "choices"
DIMS = ["S", "A", "C"]
VALID_ANSWERS = {"A", "B", "C", "D"}

DISTRACTOR_NAMES = {
    "A": ["Early", "Late", "Wide"],
    "C": ["Reverse", "Shuffle", "Loop"],
}


def _normalize_answer_value(answer: object) -> str | None:
    """规范化 answer：仅允许单个 A/B/C/D，支持 <A>。"""
    if not isinstance(answer, str):
        return None
    s = answer.strip().upper()
    if not s:
        return None
    if s.startswith("<") and s.endswith(">") and len(s) >= 3:
        s = s[1:-1].strip()
    if len(re.findall(r"[ABCD]", s)) != 1:
        return None
    return s if re.fullmatch(r"[ABCD]", s) else None


def _parse_raw_answer_strict(raw: object) -> str | None:
    """
    严格解析 raw：
    - 必须是可直接 json.loads 的 JSON 字符串
    - 必须包含 answer/reason
    - answer 必须可规范化为单个 A/B/C/D
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if "answer" not in parsed or "reason" not in parsed:
        return None
    if not isinstance(parsed.get("reason"), str):
        return None
    return _normalize_answer_value(parsed.get("answer"))


def _is_valid_entry(entry: object) -> bool:
    """
    统计时的有效 entry 判定：
    - answer 必须合法
    - 若含 raw 字段，则 raw 需满足严格 JSON 且与 answer 一致
    """
    if not isinstance(entry, dict):
        return False

    answer = _normalize_answer_value(entry.get("answer"))
    if answer not in VALID_ANSWERS:
        return False

    if "raw" in entry:
        raw_answer = _parse_raw_answer_strict(entry.get("raw"))
        if raw_answer is None or raw_answer != answer:
            return False

    return True


def _report_suffix_from_eval_dir(eval_dir: Path) -> str:
    """从 eval_results 目录名提取后缀，例如 eval_results_no_thinking -> _no_thinking。"""
    name = eval_dir.name
    if name.startswith("eval_results"):
        return name[len("eval_results") :]
    return f"_{name}"


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

    report: dict = {"total_questions": num_questions, "models": {}}

    for path in files:
        model = path.stem
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ---- 1. 准确率统计 ----
        dim_correct = {d: 0 for d in DIMS}
        dim_total = {d: 0 for d in DIMS}
        invalid_entries = 0
        for stem, entries in data.items():
            for d in DIMS:
                if d not in entries:
                    continue
                entry = entries[d]
                if not _is_valid_entry(entry):
                    invalid_entries += 1
                    continue
                dim_total[d] += 1
                if entry.get("correct"):
                    dim_correct[d] += 1

        total = sum(dim_total.values())
        correct_all = sum(dim_correct.values())

        right_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        num_stems = 0
        for stem, entries in data.items():
            dims_present = [d for d in DIMS if d in entries and _is_valid_entry(entries[d])]
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
        if invalid_entries:
            print(f"  skipped invalid entries (strict answer/raw check): {invalid_entries}")

        model_report: dict = {
            "total_questions": num_stems,
            "invalid_entries_skipped": invalid_entries,
            "S_acc": dim_correct["S"] / dim_total["S"] if dim_total["S"] else 0,
            "A_acc": dim_correct["A"] / dim_total["A"] if dim_total["A"] else 0,
            "C_acc": dim_correct["C"] / dim_total["C"] if dim_total["C"] else 0,
            "acc": correct_all / total if total else 0,
            "3/3": right_counts[3] / num_stems if num_stems else 0,
            "2/3": right_counts[2] / num_stems if num_stems else 0,
            "1/3": right_counts[1] / num_stems if num_stems else 0,
            "0/3": right_counts[0] / num_stems if num_stems else 0,
        }

        # ---- 2. 错误来源分析（仅 A/C，需要 option_map） ----
        error_counts: dict[str, Counter] = {d: Counter() for d in ("A", "C")}
        error_totals: dict[str, int] = {d: 0 for d in ("A", "C")}
        has_option_map = False

        for stem, entries in data.items():
            for d in ("A", "C"):
                if d not in entries:
                    continue
                entry = entries[d]
                if not _is_valid_entry(entry):
                    continue
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

        if has_option_map:
            print(f"\n  Error source analysis ({model}):")
            error_report: dict = {}
            for d in ("A", "C"):
                n_err = error_totals[d]
                dim_err: dict = {"total_errors": n_err}
                if n_err == 0:
                    print(f"    {d}: no errors")
                else:
                    parts = []
                    for name in DISTRACTOR_NAMES[d]:
                        cnt = error_counts[d].get(name, 0)
                        pct = cnt / n_err if n_err else 0
                        dim_err[name] = {"count": cnt, "pct": round(pct, 4)}
                        parts.append(f"{name}={cnt}({pct:.0%})")
                    print(f"    {d} ({n_err} errors): {', '.join(parts)}")
                error_report[d] = dim_err
            model_report["error_source"] = error_report
            print()
        else:
            print("  (option_map not available, skipping error source analysis)\n")

        report["models"][model] = model_report

    # 写入 JSON 报告
    report_suffix = _report_suffix_from_eval_dir(EVAL_RESULTS_DIR)
    report_path = PRESET_DIR / f"analyze_report{report_suffix}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    if PRESET not in ("video-cloze", "subset"):
        print(f"Unknown preset: {PRESET}, use 'video-cloze' or 'subset'")
        sys.exit(1)
    print(f"Analyzing: {PRESET}")
    main()
