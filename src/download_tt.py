"""下载 video-tt 视频 + Gemini VLM 生成 caption + o3 过滤。

Pipeline (每个候选):
1. 从 video_tt_filtered.csv 加载候选列表
2. yt-dlp 下载完整 YouTube 视频 → src/downloaded/<qid>.mp4
3. 将视频 + Question + Answer 发给 Gemini VLM，生成 caption
4. 将 caption 发给 o3，判断是否适合 Temporal Cloze
5. 不适合则删除视频

需要在 .env 中设置:
- OPENAI_API_KEY / OPENAI_BASE_URL  (OpenRouter，用于 Gemini caption + o3 过滤)
"""

import base64
import csv
import json
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
CSV_PATH = ROOT / "video_tt_filtered.csv"
OUT_DIR = ROOT / "downloaded"
CHOICES_DIR = PROJECT_ROOT / "choices"
COOKIES_PATH = ROOT / "cookies.txt"

CAPTION_LOG_PATH = PROJECT_ROOT / "output" / "tt_captions.json"
LLM_LOG_PATH = PROJECT_ROOT / "output" / "tt_llm_filter.json"
REJECT_LOG_PATH = PROJECT_ROOT / "output" / "tt_rejected.json"

NUM_VIDEOS = 100
MAX_HEIGHT = 360
NUM_WORKERS = 6
GEMINI_MODEL = "google/gemini-2.5-flash"

load_dotenv(PROJECT_ROOT / ".env")
log = logging.getLogger(__name__)


# ==================== 数据加载 ====================

def load_candidates() -> list[dict]:
    candidates: list[dict] = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("qid") or "").strip()
            video_id = (row.get("video_id") or "").strip()
            youtube_url = (row.get("youtube_url") or "").strip()
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not qid or not video_id or not youtube_url:
                continue
            candidates.append({
                "qid": qid,
                "video_id": video_id,
                "youtube_url": youtube_url,
                "question": question,
                "answer": answer,
            })
    return candidates


# ==================== YouTube 下载 ====================

def download_one(qid: str, url: str) -> tuple[str, bool]:
    """下载完整 YouTube 视频 → <qid>.mp4"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{qid}.mp4"
    if out_path.exists():
        return qid, True

    cmd = [
        "yt-dlp",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "--merge-output-format", "mp4",
        "-f", f"bv*[height<={MAX_HEIGHT}]+ba/b[height<={MAX_HEIGHT}]",
        "-o", str(out_path),
    ]
    if COOKIES_PATH.exists():
        cmd += ["--cookies", str(COOKIES_PATH)]
    cmd.append(url)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return qid, True
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode(errors="replace").strip()
        print(f"[DOWNLOAD FAILED] {qid} | {url}\n  {stderr}", flush=True)
        if "Video unavailable" in stderr or "Private video" in stderr or "HTTP Error 500" in stderr:
            return qid, False
        sys.exit(1)


# ==================== Gemini VLM Caption 生成（通过 OpenRouter） ====================

CAPTION_PROMPT = """Watch this video carefully. The video has an associated question and correct answer:

Question: {question}
Correct Answer: {answer}

Based on the video content, generate a detailed caption that:
1. Describes the visual content and temporal progression of events
2. Naturally incorporates the insight from the question and answer
3. Focuses on what happens at the beginning, middle, and end of the video

Output only the caption text, nothing else."""


def _encode_video_b64(video_path: Path) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_caption(client: OpenAI, video_path: Path, question: str, answer: str) -> str | None:
    try:
        b64 = _encode_video_b64(video_path)
        prompt = CAPTION_PROMPT.format(question=question, answer=answer)
        resp = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:video/mp4;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=1024,
        )
        content = resp.choices[0].message.content
        return content.strip() if content else None
    except Exception as e:
        print(f"[GEMINI ERROR] {video_path.name} | {e}", flush=True)
        sys.exit(1)


# ==================== o3 过滤 ====================

LLM_PROMPT = """[Temporal Cloze Task Suitability]
Can this video be used for a "temporal cloze" task?
(Given the beginning and end of a video, predict what happens in the middle gap)

✓ PASS: Video has causal/temporal continuity where the middle can be inferred from before & after
- Cooking process (ingredients → [gap] → finished dish)
- Object transformation (intact → [gap] → broken/changed)
- Goal-directed actions (start state → [gap] → end state)

✗ REJECT: Middle segment cannot be meaningfully predicted
- Random/unrelated scenes
- Repetitive loops (same action throughout)
- Static scenes with no progression

Caption: {caption}

JSON: {{"pass": true/false, "reason": "one or two brief sentences"}}
"""


def _build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def llm_filter_one(client: OpenAI, caption: str, name: str = "") -> tuple[bool, str]:
    caption = (caption or "").strip()
    if not caption:
        return False, "EMPTY"
    try:
        resp = client.chat.completions.create(
            model="openai/o3",
            messages=[
                {"role": "system", "content": "You are a video content classifier. Output valid JSON only."},
                {"role": "user", "content": LLM_PROMPT.format(caption=caption)},
            ],
            max_tokens=500,
        )
    except Exception as e:
        print(f"[O3 ERROR] {name} | {e}", flush=True)
        sys.exit(1)
    content = resp.choices[0].message.content
    if not content:
        return False, "NO_RESPONSE"
    try:
        data = json.loads(content.strip())
        return bool(data.get("pass")), data.get("reason", "")
    except json.JSONDecodeError:
        return False, content


# ==================== 持久化辅助 ====================

def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ==================== Main ====================

def main() -> None:
    if not CSV_PATH.exists():
        print(f"找不到 {CSV_PATH}，请先运行 fetch_video_tt.py 生成过滤后的 CSV。")
        return

    candidates = load_candidates()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    captions = _load_json(CAPTION_LOG_PATH)
    llm_results = _load_json(LLM_LOG_PATH)
    rejected = _load_json(REJECT_LOG_PATH)

    passed_names: set[str] = {
        name for name, res in llm_results.items() if res.get("pass")
    }

    client = _build_client()

    # 构建待处理池：跳过已 rejected、已通过 o3、已生成 choices 的
    pool: list[dict] = []
    for c in candidates:
        name = f"{c['qid']}.mp4"
        if name in rejected or name in passed_names:
            continue
        if (CHOICES_DIR / c["qid"]).exists():
            continue
        pool.append(c)

    # 线程安全锁
    cap_lock = threading.Lock()
    llm_lock = threading.Lock()
    rej_lock = threading.Lock()
    counter_lock = threading.Lock()
    downloaded_count = 0

    def process_candidate(c: dict) -> None:
        nonlocal downloaded_count
        name = f"{c['qid']}.mp4"
        video_path = OUT_DIR / name

        with counter_lock:
            if downloaded_count >= NUM_VIDEOS:
                return

        # 1) 下载
        if not video_path.exists():
            _, success = download_one(c["qid"], c["youtube_url"])
            if not success:
                with rej_lock:
                    rejected[name] = "DOWNLOAD_FAILED"
                    _save_json(rejected, REJECT_LOG_PATH)
                return

        # 2) Gemini 生成 caption（如果已有则复用）
        existing_caption = captions.get(name)
        if existing_caption:
            caption = existing_caption
        else:
            caption = generate_caption(client, video_path, c["question"], c["answer"])
            if not caption:
                with rej_lock:
                    rejected[name] = "CAPTION_GENERATION_FAILED"
                    _save_json(rejected, REJECT_LOG_PATH)
                if video_path.exists():
                    video_path.unlink()
                return
            with cap_lock:
                captions[name] = caption
                _save_json(captions, CAPTION_LOG_PATH)

        # 3) o3 过滤（如果已有结果则复用）
        existing_llm = llm_results.get(name)
        if existing_llm:
            ok = existing_llm.get("pass", False)
            reason = existing_llm.get("reason", "")
        else:
            ok, reason = llm_filter_one(client, caption, name)
            with llm_lock:
                llm_results[name] = {"pass": ok, "reason": reason}
                _save_json(llm_results, LLM_LOG_PATH)

        if not ok:
            with rej_lock:
                rejected[name] = f"LLM_REJECT: {reason}"
                _save_json(rejected, REJECT_LOG_PATH)
            if video_path.exists():
                video_path.unlink()
            return

        # 4) 通过 → 计数
        with counter_lock:
            if downloaded_count < NUM_VIDEOS:
                downloaded_count += 1

    max_workers = min(NUM_WORKERS, max(1, os.cpu_count() or 1))
    if not pool:
        print("No valid candidates to process.")
        return

    print(f"Processing {len(pool)} candidates with up to {max_workers} workers ...")
    target_total = min(NUM_VIDEOS, len(pool))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(
            executor.map(process_candidate, pool),
            total=target_total,
            desc="Download+Caption+Filter",
        ):
            with counter_lock:
                if downloaded_count >= NUM_VIDEOS:
                    break

    print(f"Kept {downloaded_count}/{NUM_VIDEOS} videos.")


if __name__ == "__main__":
    main()
