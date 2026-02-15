"""一键跑完整流程：

1. src/download.py  随机下载 + LLM 预筛
2. src/filter.py    质量 + gap 检查，写 meta.json / download_rejected.json
3. src/generate.py  按 meta 生成 choices，并删除对应 downloaded 原视频
4. eval.py          用当前配置的模型做评测（eval.py 里的 EVAL_MODEL）
5. prune.py         根据 gemini-2.5-pro 的评测结果裁剪题目
6. summary_eval.py  汇总各模型结果，并打印题目总数
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run_step(cmd: list[str]) -> None:
    """运行一个子进程，失败则直接退出整个 pipeline。"""
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main() -> None:
    # 1) 下载 + LLM 预筛
    run_step([sys.executable, str(ROOT / "src" / "download.py")])

    # 2) 质量 + gap 过滤，写 meta.json / download_rejected.json
    run_step([sys.executable, str(ROOT / "src" / "filter.py")])

    # 3) 生成题目（choices），并删除对应 downloaded 原视频
    run_step([sys.executable, str(ROOT / "src" / "generate.py")])

    # 4) 评测（eval.py 内部的 EVAL_MODEL 决定用哪个模型）
    run_step([sys.executable, str(ROOT / "eval.py")])

    # 5) 按 gemini-2.5-pro 结果裁剪题目
    run_step([sys.executable, str(ROOT / "prune.py")])

    # 6) 汇总各模型评测结果
    run_step([sys.executable, str(ROOT / "analyze.py")])


if __name__ == "__main__":
    main()

