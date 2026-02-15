"""汇总 eval_results 目录下所有模型结果 JSON，输出 model / S acc / A acc / C acc / acc 表格。"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
EVAL_RESULTS_DIR = ROOT / "eval_results"
D_CHOICES = ROOT / "choices"
DIMS = ["S", "A", "C"]


def main():
    if not EVAL_RESULTS_DIR.exists():
        print(f"Directory not found: {EVAL_RESULTS_DIR}")
        return
    files = sorted(EVAL_RESULTS_DIR.glob("*.json"))
    if not files:
        print(f"No *.json found in {EVAL_RESULTS_DIR}")
        return

    rows = []
    for path in files:
        model = path.stem
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

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
        row = {
            "model": model,
            "S": dim_correct["S"] / dim_total["S"] if dim_total["S"] else 0,
            "A": dim_correct["A"] / dim_total["A"] if dim_total["A"] else 0,
            "C": dim_correct["C"] / dim_total["C"] if dim_total["C"] else 0,
            "acc": correct_all / total if total else 0,
        }
        rows.append(row)

    # 题目总数：以 choices 下子文件夹数量为准
    if D_CHOICES.exists():
        num_questions = sum(1 for p in D_CHOICES.iterdir() if p.is_dir())
    else:
        num_questions = 0
    print(f"Total questions (from choices): {num_questions}")

    # 表头
    headers = ["model", "S acc", "A acc", "C acc", "acc"]
    col_widths = [max(len(r["model"]) for r in rows) + 2, 8, 8, 8, 8]
    col_widths[0] = max(col_widths[0], 6)

    def fmt(val):
        return f"{val:.2%}" if isinstance(val, float) else str(val)

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    head = "|" + "|".join(h.center(col_widths[i]) for i, h in enumerate(headers)) + "|"
    print(sep)
    print(head)
    print(sep)
    for r in rows:
        cells = [r["model"], fmt(r["S"]), fmt(r["A"]), fmt(r["C"]), fmt(r["acc"])]
        line = "|" + "|".join(cells[i].ljust(col_widths[i]) for i in range(5)) + "|"
        print(line)
    print(sep)


if __name__ == "__main__":
    main()
