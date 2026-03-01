"""一键跑完整流程：

1. src/download_*.py    随机下载 + LLM 预筛 + 质量检查
2. src/generate.py      gap 检测 + 生成 choices
3. eval.py              用当前配置的模型做评测
4. prune.py             根据 gemini-2.5-pro 的评测结果裁剪题目
5. analyze.py           汇总各模型结果，并打印题目总数

Usage:
  python pipeline.py              # 默认 lvd
  python pipeline.py lvd          # LVD 数据源
  python pipeline.py tt           # Video-TT 数据源
  python pipeline.py favor        # FAVOR-Bench 数据集
  python pipeline.py care         # CareBench 数据集
  python pipeline.py egolife       # EgoLife 数据集
  python pipeline.py mira          # MiraData 数据集
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

PRESET = sys.argv[1] if len(sys.argv) > 1 else "lvd"

DOWNLOAD_SCRIPTS = {
    "lvd": [sys.executable, str(ROOT / "src" / "download_lvd.py")],
    "tt": [sys.executable, str(ROOT / "src" / "download_tt.py")],
    "favor": [sys.executable, str(ROOT / "src" / "download_favor.py")],
    "care": [sys.executable, str(ROOT / "src" / "download_care.py")],
    "egolife": [sys.executable, str(ROOT / "src" / "egolife" / "download_egolife.py")],
    "mira": [sys.executable, str(ROOT / "src" / "mira" / "download_mira.py")],
}

GENERATE_PRESET = {
    "lvd": "lvd",
    "tt": "tt",
    "favor": "favor",
    "care": "care",
    "egolife": "egolife",
    "mira": "mira",
}


def run_step(cmd: list[str]) -> None:
    """运行一个子进程，失败则直接退出整个 pipeline。"""
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main() -> None:
    if PRESET not in DOWNLOAD_SCRIPTS:
        print(f"未知 preset: {PRESET}，可选: {', '.join(DOWNLOAD_SCRIPTS)}")
        sys.exit(1)

    gen_preset = GENERATE_PRESET[PRESET]

    # 1) 下载 + LLM 预筛 + 质量检查
    run_step(DOWNLOAD_SCRIPTS[PRESET])

    # 2) Gap 检测 + 生成 choices
    run_step([sys.executable, str(ROOT / "src" / "generate.py"), gen_preset])

    # 3) 评测
    run_step([sys.executable, str(ROOT / "eval.py")])

    # # 4) 裁剪
    # run_step([sys.executable, str(ROOT / "prune.py"), gen_preset])

    # 5) 汇总
    run_step([sys.executable, str(ROOT / "analyze.py")])


if __name__ == "__main__":
    main()

