"""选项排列偏差实验（4 组排列，全部独立评测）。

排列 A：fixed_permutations.json 基础顺序
排列 B：A 循环右移 1 位
排列 C：A 循环右移 2 位
排列 D：A 循环右移 3 位

调用量：150 × 3 × 4 = 1800 次 / 模型
结果：eval_results/permutation_{MODEL_TAG}.json（存全部 A/B/C/D）

Usage:
  python eval_permutation.py
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import DIMENSIONS, SUBSET_DIR, build_content, call_api, load_subset_stems, make_client
from tqdm import tqdm

EVAL_MODEL  = "doubao-seed-1-6-251015"
MODEL_TAG   = EVAL_MODEL.split("/")[-1]
NUM_WORKERS = 16
NUM_FRAMES  = 16

RESULTS_DIR = SUBSET_DIR / "eval_results"
FIXED_PERMS = json.loads((SUBSET_DIR / "fixed_permutations.json").read_text())
PERM_SHIFTS = {"A": 0, "B": 1, "C": 2, "D": 3}
ALL_PERMS   = list(PERM_SHIFTS.keys())


def chosen_clip(answer: str, option_map: dict) -> str:
    return option_map.get(answer, answer) if answer else "N/A"


def rotate(options: list, shift: int) -> list:
    shift = shift % len(options)
    return options[-shift:] + options[:-shift]


def eval_one(client, model, stem, dim, perm_label):
    base_options = [tuple(x) for x in FIXED_PERMS[f"{stem}|{dim}"]]
    options = rotate(base_options, PERM_SHIFTS[perm_label])
    letters = [chr(65 + i) for i in range(len(options))]
    content, correct_letter, option_map = build_content(stem, options, NUM_FRAMES)
    answer, reason = call_api(client, model, content, letters)
    return {
        "perm": perm_label,
        "correct": answer == correct_letter,
        "answer": answer,
        "reason": reason,
        "option_map": option_map,
    }


def run():
    stems = load_subset_stems()
    client = make_client()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / f"permutation_{MODEL_TAG}.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    tasks = [
        (stem, dim, p)
        for stem in stems
        for dim in DIMENSIONS
        for p in ALL_PERMS
        if f"{stem}|{dim}|perm{p}" not in all_results
    ]

    print(f"Model: {EVAL_MODEL}", flush=True)
    print(f"Tasks (A+B+C+D): {len(tasks)}, already done: {len(all_results)}", flush=True)

    if tasks:
        saved = 0
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(eval_one, client, EVAL_MODEL, s, d, p): (s, d, p)
                for s, d, p in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Permutation"):
                stem, dim, p = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"[error] {stem} {dim} perm{p}: {e}", flush=True)
                    continue
                if result["answer"] is None:
                    print(f"[error] {stem} {dim} perm{p}: {result['reason']}", flush=True)
                    continue

                all_results[f"{stem}|{dim}|perm{p}"] = result
                saved += 1
                if saved % 20 == 0:
                    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

        results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"Saved {saved} new results → {results_path.name}", flush=True)

    _analyze(stems, all_results)


def _analyze(stems, all_results):
    dims = list(DIMENSIONS.keys())
    pairs = [("A","B"),("A","C"),("A","D"),("B","C"),("B","D"),("C","D")]

    quads = {
        (stem, dim): {p: all_results.get(f"{stem}|{dim}|perm{p}") for p in ALL_PERMS}
        for stem in stems for dim in dims
    }
    quads = {k: v for k, v in quads.items() if all(v[p] is not None for p in ALL_PERMS)}
    total = len(quads)

    print(f"\n{'='*60}")
    print(f"=== Permutation Analysis: {EVAL_MODEL} ===")
    print(f"Complete quads (A+B+C+D): {total} / {len(stems) * len(dims)}")

    if total == 0:
        print("Not enough data yet.")
        return

    # 准确率：每个维度 mean ± std 跨 4 组排列
    print("\n--- Accuracy by dimension (mean ± std across 4 permutations) ---")
    for d in dims:
        sub = {k: v for k, v in quads.items() if k[1] == d}
        n = len(sub)
        if n == 0:
            print(f"  {d}: N/A")
            continue
        perm_accs = [sum(1 for e in sub.values() if e[p].get("correct")) / n for p in ALL_PERMS]
        mean_acc = sum(perm_accs) / 4
        std = (sum((a - mean_acc) ** 2 for a in perm_accs) / 4) ** 0.5
        perm_str = "  ".join(f"{p}={a:.1%}" for p, a in zip(ALL_PERMS, perm_accs))
        print(f"  {d}: mean={mean_acc:.1%}  std={std:.1%}  [{perm_str}]")

    # Clip flip rate
    print("\n--- Clip flip rate (actual selected clip changes between permutations) ---")
    for p1, p2 in pairs:
        flipped = sum(
            1 for e in quads.values()
            if chosen_clip(e[p1].get("answer"), e[p1].get("option_map", {})) !=
               chosen_clip(e[p2].get("answer"), e[p2].get("option_map", {}))
        )
        print(f"  {p1} vs {p2}: {flipped}/{total} = {flipped/total:.1%}")

    print("\n  Per-dim clip flip rate (A vs D):")
    for d in dims:
        sub = {k: v for k, v in quads.items() if k[1] == d}
        n = len(sub)
        flipped = sum(
            1 for e in sub.values()
            if chosen_clip(e["A"].get("answer"), e["A"].get("option_map", {})) !=
               chosen_clip(e["D"].get("answer"), e["D"].get("option_map", {}))
        )
        print(f"    {d}: {flipped}/{n} = {flipped/n:.1%}" if n else f"    {d}: N/A")

    # Correctness flip rate
    print("\n--- Correctness flip rate (correct/wrong changes between permutations) ---")
    for p1, p2 in pairs:
        flipped = sum(
            1 for e in quads.values()
            if e[p1].get("correct") != e[p2].get("correct")
        )
        print(f"  {p1} vs {p2}: {flipped}/{total} = {flipped/total:.1%}")

    print("\n  Per-dim correctness flip rate (A vs D):")
    for d in dims:
        sub = {k: v for k, v in quads.items() if k[1] == d}
        n = len(sub)
        flipped = sum(1 for e in sub.values() if e["A"].get("correct") != e["D"].get("correct"))
        print(f"    {d}: {flipped}/{n} = {flipped/n:.1%}" if n else f"    {d}: N/A")

    # 正确性一致性
    print("\n--- Correctness consistency (how many of 4 perms correct) ---")
    cnt = Counter(
        sum(1 for p in ALL_PERMS if e[p].get("correct"))
        for e in quads.values()
    )
    for n in [4, 3, 2, 1, 0]:
        print(f"  {n}/4 correct: {cnt[n]}/{total} = {cnt[n]/total:.1%}")


if __name__ == "__main__":
    run()
