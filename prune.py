"""删除「某模型 S/A/C 全对」的题目：删 choices 与 downloaded 中对应文件，并写入 {prefix}_rejected.json。

Usage:
  python prune.py          # 默认处理 lvd
  python prune.py tt       # 处理 video-tt
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PRESET = sys.argv[1] if len(sys.argv) > 1 else "lvd"
EVAL_RESULTS_DIR = ROOT / "eval_results"
CHOICES_DIR = ROOT / "choices"
DOWNLOADED_DIR = ROOT / "src" / "downloaded"
REJECTED_PATH = ROOT / "output" / f"{PRESET}_rejected.json"

# 使用该模型的评测结果判定「全对」
EVAL_JSON = EVAL_RESULTS_DIR / "gemini-2.5-pro.json"
PRUNE_REASON = "PRUNED: S/A/C all correct in gemini-2.5-pro eval"


def run():
    if not EVAL_JSON.exists():
        print(f"Eval file not found: {EVAL_JSON}")
        return

    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 找出 S、A、C 全部 correct 的 stem
    to_prune = []
    for stem, dims in results.items():
        if not isinstance(dims, dict):
            continue
        s_ok = dims.get("S", {}).get("correct") is True
        a_ok = dims.get("A", {}).get("correct") is True
        c_ok = dims.get("C", {}).get("correct") is True
        if s_ok and a_ok and c_ok:
            to_prune.append(stem)

    if not to_prune:
        print("No stems with S/A/C all correct. Nothing to prune.")
        return

    print(f"Pruning {len(to_prune)} stems: S/A/C all correct in {EVAL_JSON.name}")

    # 加载已有 rejected，待会合并写回
    if REJECTED_PATH.exists():
        with open(REJECTED_PATH, "r", encoding="utf-8") as f:
            rejected = json.load(f)
    else:
        rejected = {}

    for stem in to_prune:
        key = f"{stem}.mp4"

        # 删除 choices/{stem}/
        choices_stem = CHOICES_DIR / stem
        if choices_stem.exists():
            shutil.rmtree(choices_stem)
            print(f"  removed choices/{stem}/")

        # 删除 src/downloaded/{stem}.mp4
        video_path = DOWNLOADED_DIR / key
        if video_path.exists():
            video_path.unlink()
            print(f"  removed downloaded/{key}")

        rejected[key] = PRUNE_REASON

    REJECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REJECTED_PATH, "w", encoding="utf-8") as f:
        json.dump(rejected, f, indent=2, ensure_ascii=False)
    print(f"Updated {REJECTED_PATH} ({len(to_prune)} new entries).")

    # 从 eval_results 下每个 json 中删除对应题目
    for path in EVAL_RESULTS_DIR.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        changed = False
        for stem in to_prune:
            if stem in data:
                del data[stem]
                changed = True
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  removed stems from {path.name}")
    print("Done.")


if __name__ == "__main__":
    run()
