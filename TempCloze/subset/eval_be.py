"""上下文消融：只有 BEGINNING / 只有 END / BEGINNING+END，对 4 个模型的影响。

固定 permutation（fixed_permutations.json）、固定 16 帧。
- mode=B  ：只给 BEGINNING + 候选
- mode=E  ：只给 END + 候选
- mode=BE ：BEGINNING + END + 候选（复用 frames_{MODEL_TAG}.json 中 num_frames=16 的结果，
            通过 reuse_be_from_frames16.py 预先写入 be_{MODEL_TAG}.json）

结果：eval_results/be_{MODEL_TAG}.json
key 格式：{stem}|{dim}|{mode}

Usage:
  python eval_be.py
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import (CHOICES_DIR, DIMENSIONS, SUBSET_DIR, call_api,
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
MODES = ["B", "E"]   # BE 由 reuse_be_from_frames16.py 复用 16 帧结果生成

RESULTS_DIR = SUBSET_DIR / "eval_results"
FIXED_PERMS = json.loads((SUBSET_DIR / "fixed_permutations.json").read_text())

PROMPT_B_ONLY = """You are given:
- **BEGINNING**: the first part of a video (frames in temporal order).
The middle and the end of the video are NOT shown.

You will then see four candidate middle segments, labeled A, B, C, D. Each candidate is a short clip; exactly one is the true middle that directly follows the BEGINNING. The others are wrong.

Task: Which candidate is the correct middle? Choose one of A, B, C, D.

**Output JSON only** : {"answer": "<A, B, C, or D>", "reason": "<one or two sentences>"}"""

PROMPT_E_ONLY = """You are given:
- **END**: the last part of a video (frames in temporal order).
The beginning and the middle of the video are NOT shown.

You will then see four candidate middle segments, labeled A, B, C, D. Each candidate is a short clip; exactly one is the true middle that directly precedes the END. The others are wrong.

Task: Which candidate is the correct middle? Choose one of A, B, C, D.

**Output JSON only** : {"answer": "<A, B, C, or D>", "reason": "<one or two sentences>"}"""

PROMPTS = {"B": PROMPT_B_ONLY, "E": PROMPT_E_ONLY}
CONTEXT_FILE = {"B": "before.mp4", "E": "after.mp4"}
CONTEXT_LABEL = {"B": "[BEGINNING]", "E": "[END]"}


def build_content_partial(stem, options, mode, num_frames=NUM_FRAMES):
    """构建只有 B 或只有 E 的请求 content，返回 (content, correct_letter, option_map)。"""
    base = CHOICES_DIR / stem
    letters = [chr(65 + i) for i in range(len(options))]
    correct_letter = next(letters[i] for i, (_, is_gt) in enumerate(options) if is_gt)
    option_map = {letters[i]: Path(rel).stem for i, (rel, _) in enumerate(options)}

    content = [{"type": "text", "text": PROMPTS[mode]},
               {"type": "text", "text": CONTEXT_LABEL[mode]}]
    for b in sample_and_encode(base / CONTEXT_FILE[mode], num_frames):
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    for i, (rel_path, _) in enumerate(options):
        content.append({"type": "text", "text": f"[Candidate {letters[i]}]"})
        for b in sample_and_encode(base / rel_path, num_frames):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    return content, correct_letter, option_map


def eval_one(client, model, stem, dim, mode):
    options = [tuple(x) for x in FIXED_PERMS[f"{stem}|{dim}"]]
    letters = [chr(65 + i) for i in range(len(options))]
    content, correct_letter, option_map = build_content_partial(stem, options, mode)
    answer, reason = call_api(client, model, content, letters)
    return {
        "mode": mode,
        "correct": answer == correct_letter,
        "answer": answer,
        "reason": reason,
        "option_map": option_map,
    }


def run_model(model: str, stems: list[str]):
    model_tag = model.split("/")[-1]
    client = make_client()
    results_path = RESULTS_DIR / f"be_{model_tag}.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    tasks = [
        (stem, dim, mode)
        for stem in stems
        for dim in DIMENSIONS
        for mode in MODES
        if f"{stem}|{dim}|{mode}" not in all_results
    ]

    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Tasks (B+E): {len(tasks)}, already done: {len(all_results)}", flush=True)

    if not tasks:
        print("No new tasks.", flush=True)
        _summarize(model_tag, all_results, stems)
        return

    saved = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(eval_one, client, model, s, d, m): (s, d, m)
            for s, d, m in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"BE[{model_tag}]"):
            stem, dim, mode = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"[error] {stem} {dim} {mode}: {e}", flush=True)
                continue
            if result["answer"] is None:
                print(f"[error] {stem} {dim} {mode}: {result['reason']}", flush=True)
                continue

            all_results[f"{stem}|{dim}|{mode}"] = result
            saved += 1
            if saved % 20 == 0:
                results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"Saved {saved} new results → {results_path.name}", flush=True)
    _summarize(model_tag, all_results, stems)


def _summarize(model_tag: str, all_results: dict, stems: list[str]):
    dims = list(DIMENSIONS.keys())
    all_modes = ["B", "E", "BE"]

    print(f"\n--- Accuracy ({model_tag}) ---", flush=True)
    header = f"  {'Dim':<10}" + "".join(f"{m:>10}" for m in all_modes)
    print(header, flush=True)
    for d in dims:
        row = f"  {d:<10}"
        for m in all_modes:
            entries = [v for k, v in all_results.items()
                       if k.endswith(f"|{d}|{m}")]
            if entries:
                acc = sum(1 for e in entries if e.get("correct")) / len(entries)
                row += f"{acc:>9.1%} "
            else:
                row += f"{'N/A':>10}"
        print(row, flush=True)

    row = f"  {'Overall':<10}"
    for m in all_modes:
        entries = [v for k, v in all_results.items() if k.rsplit("|", 1)[1] == m]
        if entries:
            acc = sum(1 for e in entries if e.get("correct")) / len(entries)
            row += f"{acc:>9.1%} "
        else:
            row += f"{'N/A':>10}"
    print(row, flush=True)


def run():
    stems = load_subset_stems()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Stems: {len(stems)}, frames: {NUM_FRAMES}, modes: {MODES} (BE 复用 frames=16)", flush=True)
    for model in MODELS:
        run_model(model, stems)


if __name__ == "__main__":
    run()
