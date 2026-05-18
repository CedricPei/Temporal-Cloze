"""边界上下文帧数实验。

BEGINNING 使用 16 帧 bin-centered 采样后的 last K，END 使用 first K。
四个候选始终各使用 16 帧。Full 使用 BEGINNING/END 各 16 帧。

结果：eval_results/boundary_k_{MODEL_TAG}.json

Usage:
  python eval_boundary_k.py
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import (CHOICES_DIR, DIMENSIONS, EVAL_PROMPT, SUBSET_DIR, call_api,
                    load_subset_stems, make_client, sample_and_encode)
from tqdm import tqdm

MODELS = [
    "doubao-seed-1-6-251015",
    "gemini-2.5-pro",
    "qwen3.5-397b-a17b",
    "qwen3-vl-8b-instruct",
]
NUM_WORKERS = 16
NUM_FRAMES = 16
CONDITIONS = [("K1", 1), ("K4", 4), ("K8", 8), ("K12", 12), ("Full", None)]

RESULTS_DIR = SUBSET_DIR / "eval_results"
FIXED_PERMS = json.loads((SUBSET_DIR / "fixed_permutations.json").read_text())


def build_content_boundary(stem: str, options: list[tuple[str, bool]], k: int | None):
    base = CHOICES_DIR / stem
    letters = [chr(65 + i) for i in range(len(options))]
    correct_letter = next(letters[i] for i, (_, is_gt) in enumerate(options) if is_gt)
    option_map = {letters[i]: Path(rel).stem for i, (rel, _) in enumerate(options)}

    before = sample_and_encode(base / "before.mp4", NUM_FRAMES)
    after = sample_and_encode(base / "after.mp4", NUM_FRAMES)
    if k is not None:
        before = before[-k:]
        after = after[:k]

    content: list[dict] = [{"type": "text", "text": EVAL_PROMPT}]
    content.append({"type": "text", "text": "[BEGINNING]"})
    for b in before:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

    content.append({"type": "text", "text": "[END]"})
    for b in after:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

    for i, (rel_path, _) in enumerate(options):
        content.append({"type": "text", "text": f"[Candidate {letters[i]}]"})
        for b in sample_and_encode(base / rel_path, NUM_FRAMES):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

    return content, correct_letter, option_map, len(before), len(after)


def eval_one(client, model: str, stem: str, dim: str, condition: str, k: int | None):
    options = [tuple(x) for x in FIXED_PERMS[f"{stem}|{dim}"]]
    letters = [chr(65 + i) for i in range(len(options))]
    content, correct_letter, option_map, before_count, after_count = build_content_boundary(stem, options, k)
    answer, reason = call_api(client, model, content, letters)

    return {
        "condition": condition,
        "k": k,
        "before_frames": before_count,
        "candidate_frames": NUM_FRAMES,
        "after_frames": after_count,
        "correct": answer == correct_letter,
        "answer": answer,
        "reason": reason,
        "option_map": option_map,
    }


def run_model(model: str, stems: list[str]):
    model_tag = model.split("/")[-1]
    client = make_client()
    results_path = RESULTS_DIR / f"boundary_k_{model_tag}.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    tasks = [
        (stem, dim, condition, k)
        for stem in stems
        for dim in DIMENSIONS
        for condition, k in CONDITIONS
        if f"{stem}|{dim}|{condition}" not in all_results
    ]

    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Tasks: {len(tasks)}, already done: {len(all_results)}", flush=True)

    if tasks:
        saved = 0
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(eval_one, client, model, s, d, c, k): (s, d, c)
                for s, d, c, k in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"BoundaryK[{model_tag}]"):
                stem, dim, condition = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"[error] {stem} {dim} {condition}: {e}", flush=True)
                    continue
                if result["answer"] is None:
                    print(f"[error] {stem} {dim} {condition}: {result['reason']}", flush=True)
                    continue

                all_results[f"{stem}|{dim}|{condition}"] = result
                saved += 1
                if saved % 20 == 0:
                    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

        results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"Saved {saved} new results -> {results_path.name}", flush=True)

    _summarize(model_tag, all_results)


def _summarize(model_tag: str, all_results: dict):
    print(f"\n--- Accuracy ({model_tag}) ---", flush=True)
    header = f"  {'Dim':<8}" + "".join(f"{c:>10}" for c, _ in CONDITIONS)
    print(header, flush=True)

    for dim in DIMENSIONS:
        row = f"  {dim:<8}"
        for condition, _ in CONDITIONS:
            entries = [v for k, v in all_results.items() if k.endswith(f"|{dim}|{condition}")]
            if entries:
                acc = sum(1 for e in entries if e.get("correct")) / len(entries)
                row += f"{acc:>9.1%} "
            else:
                row += f"{'N/A':>10}"
        print(row, flush=True)

    row = f"  {'Overall':<8}"
    for condition, _ in CONDITIONS:
        entries = [v for k, v in all_results.items() if k.rsplit("|", 1)[1] == condition]
        if entries:
            acc = sum(1 for e in entries if e.get("correct")) / len(entries)
            row += f"{acc:>9.1%} "
        else:
            row += f"{'N/A':>10}"
    print(row, flush=True)


def run():
    stems = load_subset_stems()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Stems: {len(stems)}, conditions: {[c for c, _ in CONDITIONS]}, candidate frames: {NUM_FRAMES}", flush=True)
    for model in MODELS:
        run_model(model, stems)


if __name__ == "__main__":
    run()
