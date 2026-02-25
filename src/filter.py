"""Causal Gap Identification Framework - 视频因果间隙识别

Pipeline: Quality Check → Gap Detection
Output:   {prefix}_meta.json  # 每个视频一条记录，字段: keep / gap_start / gap_end / reason

Usage:
  python filter.py          # 默认处理 lvd
  python filter.py tt       # 处理 video-tt
"""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
SRC = Path(__file__).parent

PRESET = sys.argv[1] if len(sys.argv) > 1 else "lvd"

# Gap 新逻辑：中间 50% 内随机取一段，时长占 20–40%，要求 Mean Magnitude > MAG_THRESHOLD
GAP_LEN_RATIO_MIN = 0.2
GAP_LEN_RATIO_MAX = 0.4
MIDDLE_MARGIN = 0.25   # 首尾各 25% 不考虑
MAG_THRESHOLD = 1.0
GAP_MAX_TRIES = 3      # 若 segment 不动则最多重抽次数
RESIZE_HEIGHT = 256    # 固定高度，宽度按比例缩放


# ==================== Step 1: Quality Check ====================

class QualityChecker:
    def check(self, path: Path) -> str:
        """检查视频质量，返回等级: 'LOW' / 'MEDIUM' / 'HIGH'"""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return "LOW"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 else 0
        bitrate = path.stat().st_size * 8 / duration / 1000 if duration > 0 else 0

        blur_scores = []
        for _ in range(min(10, frames)):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        cap.release()

        sharpness = sum(blur_scores) / len(blur_scores) if blur_scores else 0
        # 不做分辨率检查，仅按码率和清晰度评级
        if bitrate < 200 or sharpness < 30:
            return "LOW"
        if bitrate >= 1000 and sharpness >= 100:
            return "HIGH"
        return "MEDIUM"


class GapDetector:
    """掐头去尾（首尾 25% 不考虑），在中间 50% 内随机取一段（时长 20–40%），
    对 256p 帧做 Farneback 光流；若 Mean Magnitude > MAG_THRESHOLD 则保留。"""

    def detect(self, path: Path) -> dict:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {"keep": False, "gap_start": None, "gap_end": None, "reason": "Cannot read video"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
        if fps <= 0 or duration <= 0:
            cap.release()
            return {"keep": False, "gap_start": None, "gap_end": None, "reason": "Invalid fps/duration"}

        t_lo = MIDDLE_MARGIN * duration
        t_hi = (1 - MIDDLE_MARGIN) * duration
        if t_hi - t_lo < duration * GAP_LEN_RATIO_MIN:
            cap.release()
            return {"keep": False, "gap_start": None, "gap_end": None,
                    "reason": f"Middle window too short (T={duration:.1f}s)"}

        last_reason = ""
        for attempt in range(GAP_MAX_TRIES):
            gap_len = random.uniform(
                duration * GAP_LEN_RATIO_MIN,
                min(duration * GAP_LEN_RATIO_MAX, t_hi - t_lo),
            )
            start = random.uniform(t_lo, t_hi - gap_len)
            end = start + gap_len

            start_frame = int(start * fps)
            end_frame = int(end * fps)
            if end_frame <= start_frame + 1:
                last_reason = "Segment too short (frames)"
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            prev = None
            magnitudes = []
            resize_wh = None  # (width, height) 按首帧比例，固定高度 RESIZE_HEIGHT
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape[:2]
                if resize_wh is None:
                    new_h = RESIZE_HEIGHT
                    new_w = int(round(RESIZE_HEIGHT * w / h))
                    resize_wh = (new_w, new_h)
                gray = cv2.resize(gray, resize_wh)
                if prev is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    magnitudes.append(float(mag.mean()))
                prev = gray

            if not magnitudes:
                last_reason = "Cannot extract frames in segment"
                continue

            mean_mag = sum(magnitudes) / len(magnitudes)
            if mean_mag > MAG_THRESHOLD:
                cap.release()
                return {"keep": True, "gap_start": round(start, 2), "gap_end": round(end, 2), "reason": None}
            last_reason = f"Mean magnitude {mean_mag:.3f} <= {MAG_THRESHOLD}"

        cap.release()
        return {
            "keep": False,
            "gap_start": None,
            "gap_end": None,
            "reason": f"After {GAP_MAX_TRIES} tries: {last_reason}",
        }


# ==================== Pipeline ====================

def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run(video_dir: Path, output_dir: Path, prefix: str = "lvd") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checker = QualityChecker()
    detector = GapDetector()

    gap_path = output_dir / f"{prefix}_meta.json"

    if gap_path.exists():
        with open(gap_path, "r", encoding="utf-8") as f:
            raw_results: dict = json.load(f)
        gap_results: dict[str, dict] = {}
        for name, rec in raw_results.items():
            gap_results[name] = {
                "keep": rec.get("keep"),
                "gap_start": rec.get("gap_start"),
                "gap_end": rec.get("gap_end"),
                "reason": rec.get("reason"),
            }
    else:
        gap_results: dict[str, dict] = {}

    videos = sorted([f for f in video_dir.iterdir() if f.suffix.lower() == ".mp4"])
    done_names = {
        name for name, rec in gap_results.items()
        if rec.get("gap_start") is not None or rec.get("reason") is not None
    }
    new_videos = [v for v in videos if v.name not in done_names]

    rejected_path = output_dir / f"{prefix}_rejected.json"
    if rejected_path.exists():
        with open(rejected_path, "r", encoding="utf-8") as f:
            rejected = json.load(f)
    else:
        rejected = {}

    def reject_video(video_path: Path, reason: str) -> None:
        if video_path.exists():
            video_path.unlink()
        rejected[video_path.name] = reason
        save_json(rejected, rejected_path)

    # ---- Step 1: Quality Check + Step 2: Gap Detection ----
    quality_todo = new_videos
    combined_results = gap_results  # 直接在原有结构上增量更新

    for video in tqdm(quality_todo, desc="Quality + Gap", total=len(quality_todo)):
        name = video.name
        q = checker.check(video)

        # 若质量过低，直接记一条记录，删视频并加入 rejected
        if q == "LOW":
            combined_results[name] = {
                "keep": False,
                "gap_start": None,
                "gap_end": None,
                "reason": "QUALITY_LOW",
            }
            save_json(combined_results, gap_path)
            reject_video(video, "FILTER_QUALITY_LOW")
            continue

        # 质量通过，再跑 GapDetector
        result = detector.detect(video)
        combined_results[name] = {
            "keep": result["keep"],
            "gap_start": result["gap_start"],
            "gap_end": result["gap_end"],
            "reason": result["reason"],
        }
        save_json(combined_results, gap_path)
        if not result["keep"]:
            reject_video(video, result["reason"] or "FILTER_GAP_REJECT")


if __name__ == "__main__":
    run(
        video_dir=SRC / "downloaded",
        output_dir=ROOT / "output",
        prefix=PRESET,
    )
