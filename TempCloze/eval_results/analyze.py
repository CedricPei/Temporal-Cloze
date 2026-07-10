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
DISPLAY_DIM = {"S": "S", "A": "A", "C": "P"}
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

# Keep report model identifiers stable even though raw result filenames use
# provider-specific spellings. Downstream figure code selects these names.
ALL_MODEL_NAMES = {
    "claude-opus-4-6": "Claude4.6-Opus",
    "claude-sonnet-4-6": "Claude4.6-Sonnet",
    "gemini-2.5-flash": "Gemini2.5-Flash",
    "gemini-2.5-pro": "Gemini2.5-Pro",
    "gemini-3-flash-preview": "Gemini3.Flash",
    "gpt-5.4": "GPT5.4",
    "grok-4.1": "Grok4.1",
    "kimi-k2.5": "KimiK2.5",
    "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
    "qwen3.5-397b-a17b": "Qwen3.5-397B-A17B",
    "qwen3.5-plus": "Qwen3.5-Plus",
    "seed-1-6": "Seed1.6",
    "seed-1-8-thinking": "Seed1.8-T",
    "seed-1-8": "Seed1.8-I",
    "GLM-4.6V-Flash": "GLM4.6V-Flash",
    "InternVL3-38B": "InternVL3-38B",
    "InternVL3-8B": "InternVL3-8B",
    "InternVL3_5-38B": "InternVL3.5-38B",
    "InternVL3_5-8B-HF": "InternVL3.5-8B",
    "Kimi-VL-A3B-Instruct": "KimiVL-A3B-I",
    "Kimi-VL-A3B-Thinking": "KimiVL-A3B-T",
    "LLaVA-Critic-R1-7B": "LLaVACriticR1-7B",
    "MiMo-VL-7B-RL": "MiMoVL-7B-RL",
    "MiMo-VL-7B-SFT": "MiMoVL-7B-SFT",
    "Molmo2-8B": "Molmo2-8B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5VL-7B-I",
    "Qwen3-VL-4B-Instruct": "Qwen3VL-4B-I",
    "Qwen3-VL-4B-Thinking": "Qwen3VL-4B-T",
    "Qwen3-VL-8B-Instruct": "Qwen3VL-8B-I",
    "Qwen3-VL-8B-Thinking": "Qwen3VL-8B-T",
    "Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen3VL-32B-Instruct": "Qwen3VL-32B-I",
    "Qwen3VL-32B-Thinking": "Qwen3VL-32B-T",
    "ThinkLite-VL-7B": "ThinkLiteVL-7B",
}

ALL_EXCLUDED_MODELS = {
    "molomoDisCa",
    "deepseek-vl2-tiny",
    "Kimi-VL-A3B-Thinking_old",
    "NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    "Qwen3-VL-4B-Thinking_nothinking_mode",
    "Qwen3-VL-8B-Thinking_nothinking_mode",
}

OPEN_MODEL_NAMES = {
    "InternVL3-38B": "vllm-OpenGVLab_InternVL3-38B",
    "InternVL3-8B": "vllm-OpenGVLab_InternVL3-8B",
    "InternVL3_5-38B": "vllm-OpenGVLab_InternVL3_5-38B",
    "InternVL3_5-8B-HF": "vllm-OpenGVLab_InternVL3_5-8B-HF",
    "Qwen2.5-VL-7B-Instruct": "vllm-Qwen_Qwen2.5-VL-7B-Instruct",
    "Qwen3VL-32B-Instruct": "vllm-Qwen_Qwen3VL-32B-Instruct",
    "Qwen3VL-32B-Thinking": "vllm-Qwen_Qwen3VL-32B-Thinking",
    "Qwen3-VL-4B-Instruct": "vllm-Qwen_Qwen3VL-4B-Instruct",
    "Qwen3-VL-4B-Thinking": "vllm-Qwen_Qwen3VL-4B-Thinking",
    "Qwen3-VL-8B-Instruct": "vllm-Qwen_Qwen3VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking": "vllm-Qwen_Qwen3VL-8B-Thinking",
    "Qwen3.5-9B": "vllm-Qwen_Qwen3.5-9B",
    "MiMo-VL-7B-RL": "vllm-XiaomiMiMo_MiMo-VL-7B-RL",
    "MiMo-VL-7B-SFT": "vllm-XiaomiMiMo_MiMo-VL-7B-SFT",
    "Molmo2-8B": "vllm-allenai_Molmo2-8B",
    "deepseek-vl2-tiny": "vllm-deepseek-ai_deepseek-vl2-tiny",
    "LLaVA-Critic-R1-7B": "vllm-lmms-lab_LLaVA-Critic-R1-7B",
    "Kimi-VL-A3B-Instruct": "vllm-moonshotai_Kimi-VL-A3B-Instruct",
    "Kimi-VL-A3B-Thinking": "vllm-moonshotai_Kimi-VL-A3B-Thinking",
    "Kimi-VL-A3B-Thinking_old": "vllm-moonshotai_Kimi-VL-A3B-Thinking_old",
    "ThinkLite-VL-7B": "vllm-russwang_ThinkLite-VL-7B",
    "GLM-4.6V-Flash": "vllm-zai-org_GLM-4.6V-Flash",
}

OPEN_EXCLUDED_MODELS = {
    "NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    "Qwen3-VL-4B-Thinking_nothinking_mode",
    "Qwen3-VL-8B-Thinking_nothinking_mode",
}


def report_model_name(scope: str, stem: str) -> str | None:
    if scope == "all":
        if stem in ALL_EXCLUDED_MODELS:
            return None
        return ALL_MODEL_NAMES.get(stem, stem)
    if scope == "open":
        if stem in OPEN_EXCLUDED_MODELS:
            return None
        return OPEN_MODEL_NAMES.get(stem, stem)
    return stem


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

    # 联合准确率：底层数据仍使用 S/A/C，报告统一展示为 S/A/P。
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

    joint_label = {("S","A"): "S+A", ("S","C"): "S+P",
                   ("A","C"): "A+P", ("S","A","C"): "S+A+P"}

    # 打印表格
    row = [model] + [pct(dim_correct[d], dim_total[d]) for d in DIMS] + [
        pct(correct_all, total),
        *(pct(joint_correct[g], num_stems) for g in joint_groups),
    ]
    headers = ["model", "S acc", "A acc", "P acc", "acc",
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
            display_dim = DISPLAY_DIM[d]
            print(f"  {display_dim} errors ({n_err}): {', '.join(parts)}")
            error_report[display_dim] = {
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
        "P_acc": dim_correct["C"] / dim_total["C"] if dim_total["C"] else 0,
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

    if CHOICES_DIR.exists():
        num_questions = sum(1 for p in CHOICES_DIR.iterdir() if p.is_dir())
    else:
        # choices/ may be intentionally omitted from Git. Preserve useful
        # report metadata by deriving the dataset size from model results.
        num_questions = max(
            (len(json.loads(path.read_text(encoding="utf-8"))) for path in files),
            default=0,
        )
    selected_files = [
        (path, report_model_name(scope, path.stem))
        for path in files
    ]
    selected_files = [(path, name) for path, name in selected_files if name is not None]
    print(f"Scope: {scope} | Total questions: {num_questions} | Models: {len(selected_files)}\n")

    report = {"total_questions": num_questions, "scope": scope, "models": {}}
    for path, model in selected_files:
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
