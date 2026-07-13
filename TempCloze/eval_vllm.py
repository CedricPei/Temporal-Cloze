"""Temporal Cloze 评测脚本（vLLM 推理服务版）

用法:
  python eval_vllm.py                                    # 评测全部题目
  python eval_vllm.py xMVtFHXrMR4.15_3                   # 评测指定题目
  python eval_vllm.py xMVtFHXrMR4.15_3 2Tz4_p9U56w.13_1  # 评测多个题目
  python eval_vllm.py --output-name custom.json         # 自定义结果文件名

请确保已启动 vLLM 服务（OpenAI 兼容 API），并加载支持视觉的模型，例如:

  python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-VL-7B-Instruct --port 8000 \\
    --max-model-len 16384 --limit-mm-per-prompt '{"image": 96}'

多图评测时必须在启动 vLLM 时加上 --limit-mm-per-prompt：默认每请求仅允许 1 张图，
本脚本每次请求会发送 6*NUM_FRAMES 张图（默认 96 张），故需至少 --limit-mm-per-prompt '{"image": 96}'。

然后设置环境变量（可选）:
  VLLM_BASE_URL=http://127.0.0.1:8000
  VLLM_MODEL=Qwen/Qwen2-VL-7B-Instruct
  EVAL_NUM_FRAMES=16
"""

import argparse
import base64
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# ==================== 基本配置 ====================
NUM_FRAMES = max(1, int(os.environ.get("EVAL_NUM_FRAMES", "16")))
NUM_WORKERS = int(os.environ.get("EVAL_NUM_WORKERS", "16"))
TASK_ORDER = os.environ.get("EVAL_TASK_ORDER", "by_stem").strip().lower()
FRAME_RETRY_SCHEDULE = [16, 12, 4, 2, 1]
# vLLM OpenAI 兼容接口地址与模型名
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8002").rstrip("/")
EVAL_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct").strip()


def _resolve_path(env_key: str, default_rel: str) -> Path:
    val = os.environ.get(env_key, "").strip()
    if not val:
        return ROOT / default_rel
    p = Path(val)
    return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()


CHOICES_DIR = _resolve_path("CHOICES_DIR", "choices")
RESULTS_DIR = _resolve_path("EVAL_RESULTS_DIR", "eval_results/customization_exp")
LOG_DIR = ROOT / "logs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", help="optional stem names to evaluate")
    parser.add_argument(
        "--output-name",
        default="",
        help="custom result JSON filename; .json will be appended if omitted",
    )
    return parser.parse_args()


def _resolve_results_path(output_name: str) -> Path:
    if not output_name:
        return RESULTS_DIR / f"{MODEL_TAG}.json"
    name = output_name.strip()
    if not name.endswith(".json"):
        name += ".json"
    return RESULTS_DIR / name

# 配置

# 全局有效帧数：遇 413 或 Too many images 时下调，所有请求共享（线程安全）
_effective_num_frames = NUM_FRAMES
_frames_lock = threading.Lock()
# 结果文件名与日志使用模型名
MODEL_TAG = f"vllm-{EVAL_MODEL.replace('/', '_')}"

# S / A / C 三个维度的选项
DIMENSIONS = {
    "S": ["S/Rand1.mp4", "S/Rand2.mp4", "S/Rand3.mp4"],
    "A": ["A/Early.mp4", "A/Late.mp4", "A/Wide.mp4"],
    "C": ["C/Reverse.mp4", "C/Shuffle.mp4", "C/Loop.mp4"],
}
VALID_ANSWERS = {"A", "B", "C", "D"}
ANSWER_LETTERS = ["A", "B", "C", "D"]

# Logging: 仅输出到日志文件，不输出到控制台
LOG_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
_fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", datefmt="%H:%M:%S")
_fh = logging.FileHandler(LOG_DIR / f"eval_{MODEL_TAG}.log", encoding="utf-8")
_fh.setFormatter(_fmt)
log.addHandler(_fh)


# ==================== 帧采样与编码 ====================

def sample_and_encode(video_path: Path, num_frames: int | None = None) -> list[str]:
    """均匀采样 num_frames 帧并编码为 base64 JPEG，默认用全局 NUM_FRAMES"""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 2:
        cap.release()
        return []

    nf = num_frames if num_frames is not None else NUM_FRAMES
    n = min(nf, total - 2)
    indices = [1 + int((i + 0.5) * (total - 2) / n) for i in range(n)]

    out: list[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return out


# ==================== Prompt & 解析 ====================

def build_prompt() -> str:
    return """You are given:
- **BEGINNING**: the first part of a video (frames in temporal order).
- **END**: the last part of the same video (frames in temporal order).
The middle segment between BEGINNING and END was removed.

You will then see four candidate middle segments, labeled A, B, C, D. Each candidate is a short clip; exactly one is the true middle that connects BEGINNING to END. The others are wrong.

Task: Which candidate is the correct middle? Choose one of A, B, C, D.

Output JSON only: {"answer": "<A, B, C, or D>", "reason": "<one or two sentences>"}"""


def _extract_response_text(resp_json: dict) -> str:
    """从 vLLM OpenAI 兼容 /v1/chat/completions 返回中提取 message.content 文本。"""
    if not resp_json:
        raise ValueError("Empty response from vLLM")
    choices = resp_json.get("choices") or []
    if not choices:
        raise ValueError("Response has no choices")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        raise ValueError("Message has no content")
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
        raise ValueError("Message content is empty")
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            else:
                parts.append(str(part))
        text = " ".join(p for p in parts if p).strip()
        if text:
            return text
        raise ValueError("Message content list is empty")
    raise ValueError("Message content is not str or list")


def _is_html_response(text: str) -> bool:
    """接口偶尔可能返回 HTML 错误页，这种要忽略。"""
    s = (text or "").strip()
    if not s or len(s) < 50:
        return False
    s_lower = s[:500].lower()
    return s_lower.startswith("<!doctype") or s_lower.startswith("<html")


def _is_model_length_error(err_str: str) -> bool:
    s = (err_str or "").lower()
    keywords = [
        "maximum sequence length",
        "sequence length is longer than the specified maximum sequence length",
        "max model len",
        "max_model_len",
        "indexing errors",
        "context length",
        # vLLM 在上下文已超长时，可能把可用 completion 预算算成负数并报这个错
        "max_tokens must be at least 1",
        "(parameter=max_tokens",
        # 多图输入常见超限/配额错误，也应触发降帧重试
        "too many images",
        "limit-mm-per-prompt",
    ]
    return any(k in s for k in keywords)


def parse_eval_response(text: str, letters: list[str]) -> tuple[str | None, str]:
    """从模型返回文本中解析 answer、reason。"""
    text = (text or "").strip()
    if not text:
        return None, ""

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        answer = (parsed.get("answer") or "").strip().upper()
        reason = parsed.get("reason") or ""
        if answer in letters:
            return answer, reason
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            answer = (parsed.get("answer") or "").strip().upper()
            reason = parsed.get("reason") or ""
            if answer in letters:
                return answer, reason
        except json.JSONDecodeError:
            pass

    return None, ""


def _is_valid_answer(answer: object) -> bool:
    return isinstance(answer, str) and answer.strip().upper() in VALID_ANSWERS


# ==================== 单题评测（调用 vLLM OpenAI 兼容 API） ====================

def _vllm_chat(content: list[dict]) -> dict:
    """调用 vLLM 的 OpenAI 兼容 /v1/chat/completions 接口。"""
    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": EVAL_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 8192,
    }
    resp = requests.post(url, json=payload, timeout=600)
    if not resp.ok:
        try:
            err_body = (resp.text or resp.content and resp.content.decode("utf-8", errors="replace")) or ""
        except Exception:
            err_body = ""
        raise RuntimeError(
            f"vLLM HTTP {resp.status_code}: {resp.reason}. {err_body[:500] if err_body else ''}"
        )
    return resp.json()


def eval_one(stem: str, dim: str, distractors: list[str]) -> dict:
    """对一个视频的一个维度做 1-of-4 评测。"""
    global _effective_num_frames
    base = CHOICES_DIR / stem
    with _frames_lock:
        start_nf = _effective_num_frames

    # GT + 3个干扰，打乱（同一次请求的降帧重试保持同一选项顺序）
    options = [("GT.mp4", True)] + [(d, False) for d in distractors]
    random.shuffle(options)
    letters = [chr(65 + i) for i in range(len(options))]
    correct_letter = next(letters[i] for i, (_, is_gt) in enumerate(options) if is_gt)
    option_map = {letters[i]: Path(rel_path).stem for i, (rel_path, _) in enumerate(options)}

    def build_content_for_frames(nf: int) -> list[dict]:
        before_b64 = sample_and_encode(base / "before.mp4", nf)
        after_b64 = sample_and_encode(base / "after.mp4", nf)

        content: list[dict] = [{"type": "text", "text": build_prompt()}]
        content.append({"type": "text", "text": "[BEGINNING]"})
        for b in before_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

        content.append({"type": "text", "text": "[END]"})
        for b in after_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})

        for i, (rel_path, _) in enumerate(options):
            full_path = base / rel_path
            opt_b64 = sample_and_encode(full_path, nf)
            content.append({"type": "text", "text": f"[Candidate {letters[i]}]"})
            for b in opt_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
        return content

    frame_candidates = [nf for nf in FRAME_RETRY_SCHEDULE if nf <= start_nf]
    if not frame_candidates:
        frame_candidates = [start_nf]
    if frame_candidates[0] != start_nf:
        frame_candidates.insert(0, start_nf)

    last_error: Exception | None = None
    # 全局降帧重试：触发后会更新 _effective_num_frames，影响后续请求。
    for idx, nf in enumerate(frame_candidates):
        content = build_content_for_frames(nf)

        # 每个 nf 下最多重试 3 次（网络抖动等）
        for attempt in range(3):
            try:
                resp_json = _vllm_chat(content)
                text = _extract_response_text(resp_json)
                if not text:
                    raise ValueError("Empty response from vLLM")
                if _is_html_response(text):
                    raise ValueError("vLLM returned HTML instead of model output")
                answer, reason = parse_eval_response(text, letters)
                if not _is_valid_answer(answer):
                    raise ValueError(
                        "Parse failed: response must contain JSON with a single A/B/C/D answer."
                    )
                correct = answer == correct_letter
                return {
                    "stem": stem,
                    "dim": dim,
                    "correct": correct,
                    "answer": answer,
                    "expected": correct_letter,
                    "reason": reason,
                    "option_map": option_map,
                }
            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                # model_length 报错：进入下一档帧数，并全局下调，影响后续请求。
                if _is_model_length_error(err_str):
                    next_nf = frame_candidates[idx + 1] if idx + 1 < len(frame_candidates) else nf
                    with _frames_lock:
                        if next_nf < _effective_num_frames:
                            _effective_num_frames = next_nf
                    log.info(
                        f"model_length on {stem} {dim}, retry with fewer frames "
                        f"(nf={nf} -> {next_nf}), global_num_frames={_effective_num_frames}"
                    )
                    break

                if attempt < 2:
                    time.sleep((attempt + 1) * 5)
                else:
                    log.error(f"vLLM error on {stem} {dim} (nf={nf}): {e}")
                    return {
                        "stem": stem,
                        "dim": dim,
                        "correct": False,
                        "answer": None,
                        "expected": correct_letter,
                        "reason": None,
                        "option_map": option_map,
                        "error": str(e)[:200],
                    }

    final_err = str(last_error)[:200] if last_error else "Model length retries exhausted"
    log.error(f"vLLM model_length retries exhausted on {stem} {dim}: {final_err}")
    return {
        "stem": stem,
        "dim": dim,
        "correct": False,
        "answer": None,
        "expected": correct_letter,
        "reason": None,
        "option_map": option_map,
        "error": final_err,
    }


# ==================== Main ====================

def run(targets: list[str] | None = None, output_name: str = ""):
    """targets: 指定题目 stem 列表, None 则全部评测"""
    global _effective_num_frames
    _effective_num_frames = NUM_FRAMES  # 每次 run 开始时重置，本轮任务共享

    if not CHOICES_DIR.exists():
        log.error(f"Choices directory not found: {CHOICES_DIR}")
        print(f"[eval_vllm] 错误: 题目目录不存在: {CHOICES_DIR}")
        print(
            "[eval_vllm] 请创建该目录并放入题目（每题为子目录且含 GT.mp4），"
            "或设置环境变量 CHOICES_DIR，例如: CHOICES_DIR=Videos-LVD2M/choices python eval_vllm.py"
        )
        return

    all_stems = sorted(
        p.name for p in CHOICES_DIR.iterdir() if p.is_dir() and (p / "GT.mp4").exists()
    )

    if targets:
        missing = [s for s in targets if s not in all_stems]
        stems = [s for s in targets if s in all_stems]
    else:
        stems = all_stems

    log.info(f"vLLM model: {EVAL_MODEL}")
    log.info(f"vLLM base_url: {VLLM_BASE_URL}")
    print(f"[eval_vllm] Base URL: {VLLM_BASE_URL}")
    log.info(f"Num videos: {len(stems)}, total tasks ≈ {len(stems) * 3}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = _resolve_results_path(output_name)

    # 断点续做
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        removed_invalid = 0
        compacted_existing = 0
        for stem in list(all_results.keys()):
            stem_results = all_results.get(stem)
            if not isinstance(stem_results, dict):
                del all_results[stem]
                continue
            for dim in list(stem_results.keys()):
                entry = stem_results.get(dim)
                if not isinstance(entry, dict):
                    del stem_results[dim]
                    removed_invalid += 1
                    continue

                saved_answer = entry.get("answer")
                answer = saved_answer.strip().upper() if isinstance(saved_answer, str) else None
                if answer not in ANSWER_LETTERS:
                    del stem_results[dim]
                    removed_invalid += 1
                    continue
                if entry.get("answer") != answer:
                    entry["answer"] = answer
                    compacted_existing += 1

                compact_entry = {
                    "correct": entry.get("correct"),
                    "answer": entry.get("answer"),
                    "expected": entry.get("expected"),
                    "reason": entry.get("reason"),
                    "option_map": entry.get("option_map"),
                }
                if entry != compact_entry:
                    stem_results[dim] = compact_entry
                    compacted_existing += 1
            if not stem_results:
                del all_results[stem]

        if removed_invalid or compacted_existing:
            log.info(
                f"Resume cleanup: removed {removed_invalid} invalid entries, "
                f"compacted {compacted_existing} result entries"
            )
            print(
                f"[eval_vllm] 断点续做: 清理了 {removed_invalid} 条无效记录，"
                f"压缩了 {compacted_existing} 条结果字段，将继续评测剩余任务"
            )
            try:
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                log.error(f"Failed to rewrite cleaned results to {results_path}: {e}")

        done_count = sum(len(v) for v in all_results.values())
        done_stems = sorted(all_results.keys())
        last_stem = done_stems[-1] if done_stems else None
        log.info(f"结果文件: {results_path}, 已写入 {len(done_stems)} 个题目共 {done_count} 条结果")
        if last_stem is not None:
            log.info(f"最后有记录的题目: {last_stem} (维度: {list(all_results[last_stem].keys())})")
        print(
            f"[eval_vllm] 断点续做: 结果文件 {results_path.name}, "
            f"已完成 {done_count} 题, 最后题目 {last_stem or '-'}, 将跳过已测题目继续评测"
        )
    else:
        all_results = {}
        print(f"[eval_vllm] 结果文件不存在，将从头评测并保存到 {results_path}")

    # 构建任务：只包含尚未完成的 (stem, dim)
    tasks: list[tuple[str, str, list[str]]] = []
    if TASK_ORDER == "by_dim":
        for dim, distractors in DIMENSIONS.items():
            for stem in stems:
                if not (CHOICES_DIR / stem / "GT.mp4").exists():
                    continue
                if stem in all_results and dim in all_results[stem]:
                    continue
                tasks.append((stem, dim, distractors))
    else:
        for stem in stems:
            if not (CHOICES_DIR / stem / "GT.mp4").exists():
                continue
            for dim, distractors in DIMENSIONS.items():
                if stem in all_results and dim in all_results[stem]:
                    continue
                tasks.append((stem, dim, distractors))

    already_done = sum(len(v) for v in all_results.values())
    log.info(f"Tasks to run: {len(tasks)}, already done: {already_done}")

    if not tasks:
        log.info("No tasks to run.")
        print(
            f"[eval_vllm] 没有待评测任务（共 {len(stems)} 个题目，已完成 {already_done} 条）。"
            "若题目数为 0，请检查 CHOICES_DIR 下是否有含 GT.mp4 的子目录。"
        )
        return

    max_workers = min(NUM_WORKERS, max(1, os.cpu_count() or 1))
    future_to_task: dict = {}
    saved_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for stem, dim, distractors in tasks:
            fut = executor.submit(eval_one, stem, dim, distractors)
            future_to_task[fut] = (stem, dim)

        pbar = tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Evaluating (vLLM)")
        for fut in pbar:
            stem, dim = future_to_task[fut]
            try:
                result = fut.result()
            except Exception as e:
                log.error(f"Unexpected error in worker for {stem} {dim}: {e}")
                continue

            with _frames_lock:
                global_nf = _effective_num_frames
            pbar.set_postfix_str(f"global_nf={global_nf}", refresh=False)

            if "error" in result:
                err_msg = result["error"]
                log.error(f"vLLM error on {stem} {dim}: {err_msg}")
                print(f"[eval_vllm] ERROR: {stem} {dim} vLLM error: {err_msg}")
                if stem not in all_results:
                    all_results[stem] = {}
                all_results[stem][dim] = {
                    "correct": False,
                    "answer": result.get("answer"),
                    "expected": result.get("expected"),
                    "reason": result.get("reason"),
                    "option_map": result.get("option_map"),
                }
                try:
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    log.error(f"Failed to save results to {results_path}: {e}")
                continue

            answer = result.get("answer")
            if not _is_valid_answer(answer):
                err_msg = (
                    f"Invalid final answer on {stem} {dim}: {answer!r}. "
                    "Expected one of A/B/C/D after all retries."
                )
                log.error(f"Skip saving result with invalid answer on {stem} {dim}: {err_msg}")
                print(
                    f"[eval_vllm] WARN: {stem} {dim} answer无效({answer!r})，"
                    "不写入JSON，后续可续跑"
                )
                continue

            entry = {
                "correct": result.get("correct"),
                "answer": result.get("answer"),
                "expected": result.get("expected"),
                "reason": result.get("reason"),
                "option_map": result.get("option_map"),
            }
            if stem not in all_results:
                all_results[stem] = {}
            all_results[stem][dim] = entry
            saved_count += 1

            tag = "✓" if result["correct"] else "✗"
            tqdm.write(
                f"  {tag} {stem} {dim}  ans={result['answer']} "
                f"exp={result['expected']}  {(result.get('reason') or '')[:60]}"
            )

            try:
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                log.error(f"Failed to save results to {results_path}: {e}")

    log.info(f"Saved {saved_count}/{len(tasks)} tasks to {results_path}")

    # 汇总
    for dim in DIMENSIONS:
        entries = [v[dim] for v in all_results.values() if dim in v]
        c = sum(1 for e in entries if e["correct"])
        t = len(entries)
        log.info(f"{dim}: {c}/{t} = {c/t:.2%}" if t else f"{dim}: no data")

    all_entries = [e for v in all_results.values() for e in v.values()]
    total = len(all_entries)
    correct = sum(1 for e in all_entries if e["correct"])
    log.info(f"Overall: {correct}/{total} = {correct/total:.2%}" if total else "Overall: no data")


if __name__ == "__main__":
    args = _parse_args()
    run(args.targets or None, args.output_name)
