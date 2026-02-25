"""从 HuggingFace 下载 video-tt 数据集的 test split，按 duration 过滤。

过滤条件：
- duration 在 [12, 90] 秒之间（与 hdvg 管线一致）

输出：src/video_tt_filtered.csv
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path

ROOT = Path(__file__).parent
OUT_CSV = ROOT / "video_tt_filtered.csv"

MIN_DURATION = 12.0
MAX_DURATION = 90.0


def main() -> None:
    print("Loading lmms-lab/video-tt (test split) from HuggingFace ...")
    ds = load_dataset("lmms-lab/video-tt", split="test")
    df = ds.to_pandas()
    print(f"  Total rows: {len(df)}")

    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df[df["duration"].between(MIN_DURATION, MAX_DURATION)]
    print(f"  After duration filter [{MIN_DURATION}, {MAX_DURATION}]s: {len(df)}")

    keep_cols = ["qid", "video_id", "capability", "question", "duration", "answer", "youtube_url"]
    df = df[[c for c in keep_cols if c in df.columns]]

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
