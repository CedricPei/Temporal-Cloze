"""公共工具：帧采样、响应解析、API 调用、路径常量。"""

import base64
import json
import os
import re
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parent.parent          # video-cloze/
PROJECT_ROOT = ROOT.parent                   # Temporal-Cloze/
load_dotenv(PROJECT_ROOT / ".env")

SUBSET_DIR = Path(__file__).parent           # video-cloze/subset/
SUBSET_IDS = SUBSET_DIR / "subset_ids.json"
CHOICES_DIR = ROOT / "choices"

MAX_HEIGHT = 360
MAX_RETRIES = 3

DIMENSIONS = {
    "S": ["S/Rand1.mp4", "S/Rand2.mp4", "S/Rand3.mp4"],
    "A": ["A/Early.mp4", "A/Late.mp4", "A/Wide.mp4"],
    "C": ["C/Reverse.mp4", "C/Shuffle.mp4", "C/Loop.mp4"],
}

EVAL_PROMPT = """You are given:
- **BEGINNING**: the first part of a video (frames in temporal order).
- **END**: the last part of the same video (frames in temporal order).
The middle segment between BEGINNING and END was removed.

You will then see four candidate middle segments, labeled A, B, C, D. Each candidate is a short clip; exactly one is the true middle that connects BEGINNING to END. The others are wrong.

Task: Which candidate is the correct middle? Choose one of A, B, C, D.

**Output JSON only** : {"answer": "<A, B, C, or D>", "reason": "<one or two sentences>"}"""


def load_subset_stems() -> list[str]:
    data = json.loads(SUBSET_IDS.read_text(encoding="utf-8"))
    return [s for s in data if (CHOICES_DIR / s / "GT.mp4").exists()]


def make_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def sample_and_encode(video_path: Path, num_frames: int = 16) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 2:
        cap.release()
        return []
    n = min(num_frames, total - 2)
    indices = [1 + int((i + 0.5) * (total - 2) / n) for i in range(n)]
    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            if h > MAX_HEIGHT:
                scale = MAX_HEIGHT / h
                frame = cv2.resize(frame, (int(w * scale), MAX_HEIGHT))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return out


def _try_parse_json(text: str, letters: list[str]) -> tuple[str | None, str]:
    """尝试解析 JSON 字符串，成功返回 (answer, reason)，失败返回 (None, "")。"""
    try:
        parsed = json.loads(text)
        answer = (parsed.get("answer") or "").strip().upper()
        reason = parsed.get("reason") or ""
        if answer in letters:
            return answer, reason
    except (json.JSONDecodeError, AttributeError):
        pass
    return None, ""


def parse_eval_response(raw: str, letters: list[str]) -> tuple[str | None, str]:
    raw = (raw or "").strip()
    if not raw:
        return None, ""

    # 步骤1：如果有 ```json``` 块，优先从中提取
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        answer, reason = _try_parse_json(m.group(1).strip(), letters)
        if answer is not None:
            return answer, reason

    # 步骤2：寻找第一个 { 到最后一个 }，尝试解析
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        answer, reason = _try_parse_json(raw[start: end + 1], letters)
        if answer is not None:
            return answer, reason

    return None, raw


def build_content(stem: str, options: list[tuple[str, bool]],
                  num_frames: int = 16) -> tuple[list[dict], str, dict]:
    """构建 API 请求 content，返回 (content, correct_letter, option_map)。"""
    base = CHOICES_DIR / stem
    letters = [chr(65 + i) for i in range(len(options))]
    correct_letter = next(letters[i] for i, (_, is_gt) in enumerate(options) if is_gt)
    option_map = {letters[i]: Path(rel).stem for i, (rel, _) in enumerate(options)}

    content: list[dict] = [{"type": "text", "text": EVAL_PROMPT}]
    content.append({"type": "text", "text": "[BEGINNING]"})
    for b in sample_and_encode(base / "before.mp4", num_frames):
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    content.append({"type": "text", "text": "[END]"})
    for b in sample_and_encode(base / "after.mp4", num_frames):
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    for i, (rel_path, _) in enumerate(options):
        content.append({"type": "text", "text": f"[Candidate {letters[i]}]"})
        for b in sample_and_encode(base / rel_path, num_frames):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})


    return content, correct_letter, option_map


def call_api(client: OpenAI, model: str, content: list[dict],
             letters: list[str], temperature: float = 0) -> tuple[str | None, str]:
    """调用 API 并解析，返回 (answer, reason)。失败返回 (None, error_msg)。"""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=8000,
                temperature=temperature,
            )
            if not resp.choices or not resp.choices[0].message.content:
                raise ValueError("Empty response from API")
            raw = resp.choices[0].message.content.strip()
            answer, reason = parse_eval_response(raw, letters)
            if answer is None:
                raise ValueError(f"Cannot parse: {raw[:200]}")
            return answer, reason
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 5)
            else:
                return None, str(e)[:200]
