"""测试帧数 (8/12/16/20) 对模型准确率的影响。

对 subset 中每道题的 S/A/C，分别用 8、12、16、20 帧评测。
结果：eval_results/frames_{MODEL_TAG}.json

Usage:
  python eval_frames.py
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from common import (DIMENSIONS, SUBSET_DIR, build_content,
                    call_api, load_subset_stems, make_client)
from tqdm import tqdm

EVAL_MODEL = "doubao-seed-1-6-251015"
# EVAL_MODEL = "qwen3.5-397b-a17b"
MODEL_TAG = EVAL_MODEL.split("/")[-1]
NUM_WORKERS = 16
FRAME_COUNTS = [8, 12, 16, 20]  

RESULTS_DIR = SUBSET_DIR / "eval_results"
FIXED_PERMS = json.loads((SUBSET_DIR / "fixed_permutations.json").read_text())


def eval_one(client, model, stem, dim, num_frames):
    options = [tuple(x) for x in FIXED_PERMS[f"{stem}|{dim}"]]
    letters = [chr(65 + i) for i in range(len(options))]

    content, correct_letter, option_map = build_content(stem, options, num_frames)
    answer, reason = call_api(client, model, content, letters)

    return {
        "num_frames": num_frames,
        "correct": answer == correct_letter,
        "answer": answer,
        "reason": reason,
        "option_map": option_map,
    }


def run():
    stems = load_subset_stems()
    client = make_client()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"frames_{MODEL_TAG}.json"

    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    tasks = []
    for stem in stems:
        for dim in DIMENSIONS:
            for nf in FRAME_COUNTS:
                key = f"{stem}|{dim}|{nf}"
                if key in all_results:
                    continue
                tasks.append((stem, dim, nf))

    print(f"Model: {EVAL_MODEL}")
    print(f"Stems: {len(stems)}, frame counts: {FRAME_COUNTS}")
    print(f"Tasks: {len(tasks)}, already done: {len(all_results)}")

    if not tasks:
        print("No tasks to run.")
        return

    saved = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(eval_one, client, EVAL_MODEL, s, d, nf): (s, d, nf)
            for s, d, nf in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating frames"):
            stem, dim, nf = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"[error] {stem} {dim} {nf}f: {e}", flush=True)
                continue
            if result["answer"] is None:
                print(f"[error] {stem} {dim} {nf}f: {result['reason']}", flush=True)
                continue

            key = f"{stem}|{dim}|{nf}"
            all_results[key] = result
            saved += 1

            if saved % 10 == 0:
                results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nSaved {saved} results to {results_path}")

    # 汇总
    print("\n=== Accuracy by frame count ===")
    for nf in FRAME_COUNTS:
        correct = sum(1 for k, v in all_results.items() if v["num_frames"] == nf and v["correct"])
        total = sum(1 for k, v in all_results.items() if v["num_frames"] == nf)
        print(f"  {nf:2d} frames: {correct}/{total} = {correct/total:.2%}" if total else f"  {nf} frames: N/A")


if __name__ == "__main__":
    run()
