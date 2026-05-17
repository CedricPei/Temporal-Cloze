"""Temp-Mixed 结果分析脚本

读取 eval_results/ 下各模型 JSON，输出：
  - 每模型的总准确率
  - 错误来源分布（选了 S/A/C 中哪类干扰项）
  - 汇总到 analyze_report_mixed.json 和 accuracy_table_mixed.csv

用法:
  python analyze_mixed.py
"""

import csv
import json
from pathlib import Path

MIXED_DIR = Path(__file__).parent
RESULTS_DIR = MIXED_DIR / "eval_results"
MIXED_IDS_PATH = MIXED_DIR / "mixed_ids.json"


def load_results() -> dict[str, dict]:
    """加载所有模型结果，返回 {model_tag: {stem: entry}}。"""
    all_models = {}
    for p in sorted(RESULTS_DIR.glob("*.json")):
        model_tag = p.stem
        data = json.loads(p.read_text(encoding="utf-8"))
        all_models[model_tag] = data
    return all_models


def analyze_model(model_tag: str, results: dict) -> dict:
    """计算单模型的准确率和错误来源分布。"""
    total = len(results)
    if total == 0:
        return {"total": 0, "correct": 0, "acc": 0.0,
                "error_source": {}, "error_total": 0}

    correct_count = sum(1 for v in results.values() if v.get("correct"))
    acc = correct_count / total

    errors = [v for v in results.values()
              if not v.get("correct") and v.get("error_dim") is not None]
    error_dim_counts: dict[str, int] = {}
    for v in errors:
        d = v["error_dim"]
        error_dim_counts[d] = error_dim_counts.get(d, 0) + 1

    error_total = len(errors)
    error_source = {}
    for dim in ["S", "A", "C"]:
        cnt = error_dim_counts.get(dim, 0)
        error_source[dim] = {
            "count": cnt,
            "pct_of_errors": round(cnt / error_total, 4) if error_total > 0 else 0.0,
            "pct_of_total": round(cnt / total, 4) if total > 0 else 0.0,
        }

    return {
        "total": total,
        "correct": correct_count,
        "acc": round(acc, 4),
        "error_total": error_total,
        "unanswered": total - correct_count - error_total,
        "error_source": error_source,
    }


def print_report(report: dict):
    print(f"\n{'='*70}")
    print(f"{'Model':<40} {'Acc':>8}  {'Err→S':>7}  {'Err→A':>7}  {'Err→C':>7}")
    print(f"{'-'*70}")
    for model_tag, stats in sorted(report["models"].items(), key=lambda x: -x[1]["acc"]):
        acc = f"{stats['acc']:.1%}"
        es = stats["error_source"]
        s = f"{es['S']['pct_of_total']:.1%}" if es else "N/A"
        a = f"{es['A']['pct_of_total']:.1%}" if es else "N/A"
        c = f"{es['C']['pct_of_total']:.1%}" if es else "N/A"
        print(f"  {model_tag:<38} {acc:>8}  {s:>7}  {a:>7}  {c:>7}")
    print(f"{'='*70}")


def save_csv(report: dict, out_path: Path):
    rows = []
    for model_tag, stats in sorted(report["models"].items(), key=lambda x: -x[1]["acc"]):
        es = stats["error_source"]
        rows.append({
            "Model": model_tag,
            "Accuracy": f"{stats['acc']:.2%}",
            "N": stats["total"],
            "Correct": stats["correct"],
            "Err→S": f"{es['S']['pct_of_total']:.2%}",
            "Err→A": f"{es['A']['pct_of_total']:.2%}",
            "Err→C": f"{es['C']['pct_of_total']:.2%}",
            "Unanswered": stats.get("unanswered", 0),
        })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {out_path}")


def run():
    all_models = load_results()
    if not all_models:
        print(f"No result JSONs found in {RESULTS_DIR}")
        return

    report = {"models": {}}
    for model_tag, results in all_models.items():
        report["models"][model_tag] = analyze_model(model_tag, results)

    print_report(report)

    report_path = MIXED_DIR / "analyze_report_mixed.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved → {report_path}")

    csv_path = MIXED_DIR / "accuracy_table_mixed.csv"
    save_csv(report, csv_path)


if __name__ == "__main__":
    run()
