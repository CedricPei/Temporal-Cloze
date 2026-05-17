"""pass@k 评测：固定 16 帧，每题重复 NUM_REPEATS 次，计算 pass@1 ~ pass@5。

temperature > 0 使模型每次给出不同回答。
pass@k = 1 - C(n-c, k) / C(n, k)，其中 n=总次数, c=正确次数。

结果写入 eval_results/passk_{MODEL_TAG}.json
"""

import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import DIMENSIONS, SUBSET_DIR, build_content, call_api, load_subset_stems, make_client
from tqdm import tqdm

EVAL_MODEL  = "doubao-seed-1-6-251015"
MODEL_TAG   = EVAL_MODEL.split("/")[-1]
NUM_WORKERS = 16
NUM_FRAMES  = 16
NUM_REPEATS = 5 
TEMPERATURE = 0.7
MAX_K       = 5

RESULTS_DIR  = SUBSET_DIR / "eval_results"
FIXED_PERMS  = json.loads((SUBSET_DIR / "fixed_permutations.json").read_text())


def pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k = 1 - C(n-c, k) / C(n, k)"""
    if n == 0 or c == 0 or k > n:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def eval_one(client, model, stem, dim, repeat_idx):
    options = [tuple(x) for x in FIXED_PERMS[f"{stem}|{dim}"]]
    letters = [chr(65 + i) for i in range(len(options))]
    content, correct_letter, option_map = build_content(stem, options, NUM_FRAMES)
    answer, reason = call_api(client, model, content, letters, temperature=TEMPERATURE)
    return {
        "correct": answer == correct_letter,
        "answer": answer,
        "reason": reason,
        "option_map": option_map,
    }


def run():
    stems = load_subset_stems()
    client = make_client()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"passk_{MODEL_TAG}.json"

    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    tasks = [
        (stem, dim, r)
        for stem in stems
        for dim in DIMENSIONS
        for r in range(NUM_REPEATS)
        if f"{stem}|{dim}|r{r}" not in all_results
    ]

    print(f"Model: {EVAL_MODEL}")
    print(f"Stems: {len(stems)}, repeats: {NUM_REPEATS}, temperature: {TEMPERATURE}")
    print(f"Tasks: {len(tasks)}, already done: {len(all_results)}")

    if tasks:
        saved = 0
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(eval_one, client, EVAL_MODEL, s, d, r): (s, d, r)
                for s, d, r in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="pass@k"):
                stem, dim, r = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"[error] {stem} {dim} r{r}: {e}", flush=True)
                    continue
                if result["answer"] is None:
                    print(f"[error] {stem} {dim} r{r}: {result['reason']}", flush=True)
                    continue

                all_results[f"{stem}|{dim}|r{r}"] = result
                saved += 1
                if saved % 20 == 0:
                    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

        results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"Saved {saved} new results → {results_path.name}")

    # 计算 pass@k
    groups = defaultdict(list)
    for key, v in all_results.items():
        stem_k, dim_k, _ = key.rsplit("|", 2)
        groups[f"{stem_k}|{dim_k}"].append(v)

    print(f"\n=== pass@k ({len(groups)} question-dim pairs) ===")
    for k in range(1, MAX_K + 1):
        scores = [
            pass_at_k(len(entries), sum(e["correct"] for e in entries), k)
            for entries in groups.values()
            if len(entries) >= k
        ]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  pass@{k}: {avg:.2%}  (n={len(scores)})")


if __name__ == "__main__":
    run()
