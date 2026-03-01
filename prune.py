"""删除「某模型 S/A/C 全对」的题目，并同步清理 meta.json。

操作:
  1. 删 choices/{stem}/ 目录
  2. 写入 rejected.json
  4. 从 eval_results/*.json 删除对应条目
  5. 从 meta.json 删除对应条目

Usage:
  python prune.py              # 默认处理 lvd
  python prune.py tt           # 处理 video-tt
  python prune.py favor        # 处理 FAVOR-Bench
  python prune.py care         # 处理 CareBench
  python prune.py dailyomni   # 处理 DailyOmni
  python prune.py egolife     # 处理 EgoLife
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PRESET = sys.argv[1] if len(sys.argv) > 1 else "lvd"
EVAL_RESULTS_DIR = ROOT / "eval_results"
CHOICES_DIR = ROOT / "choices"
REJECTED_PATH = ROOT / "output" / PRESET / "rejected.json"
META_PATH = ROOT / "output" / PRESET / "meta.json"

EVAL_JSON = EVAL_RESULTS_DIR / "gemini-2.5-pro.json"
PRUNE_REASON = "PRUNED: S/A/C all correct in gemini-2.5-pro eval"


def _load(path: Path) -> dict:
    return json.load(open(path, encoding="utf-8")) if path.exists() else {}


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run():
    if not EVAL_JSON.exists():
        print(f"Eval file not found: {EVAL_JSON}")
        return

    results = _load(EVAL_JSON)

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
    else:
        print(f"Pruning {len(to_prune)} stems: S/A/C all correct in {EVAL_JSON.name}")

        rejected = _load(REJECTED_PATH)
        meta = _load(META_PATH)

        for stem in to_prune:
            key = f"{stem}.mp4"

            choices_stem = CHOICES_DIR / stem
            if choices_stem.exists():
                shutil.rmtree(choices_stem)
                print(f"  removed choices/{stem}/")

            rejected[key] = PRUNE_REASON
            meta.pop(key, None)

        _save(rejected, REJECTED_PATH)
        print(f"Updated {REJECTED_PATH} ({len(to_prune)} new entries).")

        _save(meta, META_PATH)

        for path in EVAL_RESULTS_DIR.glob("*.json"):
            data = _load(path)
            changed = False
            for stem in to_prune:
                if stem in data:
                    del data[stem]
                    changed = True
            if changed:
                _save(data, path)
                print(f"  removed stems from {path.name}")

    print("Done.")


if __name__ == "__main__":
    run()
