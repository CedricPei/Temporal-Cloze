"""Temporal Cloze 评测脚本（vLLM 推理服务版）

用法:
  python eval_vllm.py                                    # 评测全部题目
  python eval_vllm.py xMVtFHXrMR4.15_3                   # 评测指定题目
  python eval_vllm.py xMVtFHXrMR4.15_3 2Tz4_p9U56w.13_1  # 评测多个题目

请确保已启动 vLLM 服务（OpenAI 兼容 API），并加载支持视觉的模型，例如:

  python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-VL-7B-Instruct --port 8000 \\
    --max-model-len 16384 --limit-mm-per-prompt '{"image": 96}'

多图评测时必须在启动 vLLM 时加上 --limit-mm-per-prompt：默认每请求仅允许 1 张图，
本脚本每次请求会发送 6*NUM_FRAMES 张图（默认 96 张），故需至少 --limit-mm-per-prompt '{"image": 96}'。

然后设置环境变量（可选）:
  VLLM_BASE_URL=http://127.0.0.1:8000
  VLLM_MODEL=Qwen/Qwen2-VL-7B-Instruct
"""

import base64
import json
import logging
import os
import random
import re
import sys
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

os.environ["CHOICES_DIR"] = "/users/henry/Temporal-Cloze/video-cloze_360p/choices"
os.environ["EVAL_RESULTS_DIR"] = "/users/henry/Temporal-Cloze/video-cloze/eval_results_newwithNothink"
# ==================== 基本配置 ====================
NUM_FRAMES = 16
NUM_WORKERS = 16
FRAME_RETRY_SCHEDULE = [16, 12, 4, 2, 1]
# vLLM OpenAI 兼容接口地址与模型名
VLLM_BASE_URL = os.environ.setdefault("VLLM_BASE_URL", "http://127.0.0.1:8002").rstrip("/")
EVAL_MODEL = os.environ.setdefault("VLLM_MODEL", "lmms-lab/LLaVA-Critic-R1-7B").strip()


def _resolve_path(env_key: str, default_rel: str) -> Path:
    val = os.environ.get(env_key, "").strip()
    if not val:
        return ROOT / default_rel
    p = Path(val)
    return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()


CHOICES_DIR = _resolve_path("CHOICES_DIR", "choices")
RESULTS_DIR = _resolve_path("EVAL_RESULTS_DIR", "eval_results_")
LOG_DIR = ROOT / "logs"

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


# region agent log
def _agent_debug_log(hypothesis_id: str, message: str, data: dict | None = None, run_id: str = "initial") -> None:
    """
    轻量调试日志：按 NDJSON 追加到 /home/wenqi/Temporal-Cloze/.cursor/debug.log。
    避免引入第三方，仅用于本次 Debug，会在确认修复后移除。
    """
    try:
        payload = {
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "location": "eval_vllm.py",
            "message": message,
            "data": data or {},
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        debug_path = Path("/home/wenqi/Temporal-Cloze/.cursor/debug.log")
        with debug_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # 调试日志失败不能影响正常评测流程
        pass
# endregion


# ==================== 帧采样与编码 ====================

def sample_and_encode(video_path: Path, num_frames: int | None = None) -> list[str]:
    """均匀采样 num_frames 帧并编码为 base64 JPEG，默认用全局 NUM_FRAMES"""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    nf = num_frames if num_frames is not None else NUM_FRAMES
    n = min(nf, total)
    indices = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]

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


def _is_html_response(raw: str) -> bool:
    """接口偶尔可能返回 HTML 错误页，这种要忽略。"""
    s = (raw or "").strip()
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
    ]
    return any(k in s for k in keywords)


def _normalize_answer_value(answer: object) -> str | None:
    """规范化 answer：仅允许单个 A/B/C/D，支持 <A>，不接受多选。"""
    if not isinstance(answer, str):
        return None

    s = answer.strip().upper()
    if not s:
        return None

    # 支持 <A> 这类输出
    if s.startswith("<") and s.endswith(">") and len(s) >= 3:
        s = s[1:-1].strip()

    # 多选（如 <A, B> / A,B / A/B）统一视为非法
    if len(re.findall(r"[ABCD]", s)) != 1:
        return None

    return s if re.fullmatch(r"[ABCD]", s) else None


def parse_eval_response(raw: str, letters: list[str]) -> tuple[str | None, str]:
    """严格解析：raw 必须是 JSON，且 answer 为单个合法选项。"""
    raw = (raw or "").strip()
    if not raw:
        return None, ""
    if _is_html_response(raw):
        return None, ""

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, ""

    if not isinstance(parsed, dict):
        return None, ""
    if "answer" not in parsed or "reason" not in parsed:
        return None, ""

    answer = _normalize_answer_value(parsed.get("answer"))
    if answer not in letters:
        return None, ""

    reason = parsed.get("reason")
    if not isinstance(reason, str):
        return None, ""

    return answer, reason


def _is_valid_answer(answer: object) -> bool:
    return isinstance(answer, str) and answer.strip().upper() in VALID_ANSWERS


# ==================== 单题评测（调用 vLLM OpenAI 兼容 API） ====================

def _vllm_chat(content: list[dict]) -> dict:
    """调用 vLLM 的 OpenAI 兼容 /v1/chat/completions 接口。"""
    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": EVAL_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2048,
        "temperature": 0,
    }
    # region agent log
    _agent_debug_log(
        hypothesis_id="H9",
        message="Preparing vLLM HTTP request",
        data={"base_url": VLLM_BASE_URL, "url": url, "model": EVAL_MODEL},
        run_id="initial",
    )
    # endregion
    resp = requests.post(url, json=payload, timeout=600)
    # region agent log
    _agent_debug_log(
        hypothesis_id="H10",
        message="vLLM HTTP response received",
        data={"status_code": resp.status_code, "ok": resp.ok},
        run_id="initial",
    )
    # endregion
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

    # region agent log
    _agent_debug_log(
        hypothesis_id="H5",
        message="eval_one start",
        data={"stem": stem, "dim": dim, "nf": start_nf, "base": str(base)},
        run_id="initial",
    )
    # endregion

    # GT + 3个干扰，打乱（同一次请求的降帧重试保持同一选项顺序）
    options = [("GT.mp4", True)] + [(d, False) for d in distractors]
    random.shuffle(options)
    letters = [chr(65 + i) for i in range(len(options))]
    correct_letter = next(letters[i] for i, (_, is_gt) in enumerate(options) if is_gt)

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
                # region agent log
                _agent_debug_log(
                    hypothesis_id="H6",
                    message="Calling _vllm_chat",
                    data={"stem": stem, "dim": dim, "attempt": attempt + 1, "num_images": len(content), "nf": nf},
                    run_id="initial",
                )
                # endregion
                resp_json = _vllm_chat(content)
                raw = _extract_response_text(resp_json)
                if not raw:
                    raise ValueError("Empty response from vLLM")
                if _is_html_response(raw):
                    raise ValueError("vLLM returned HTML instead of model output")
                answer, reason = parse_eval_response(raw, letters)
                if not _is_valid_answer(answer):
                    raise ValueError(
                        "Strict parse failed: raw must be JSON, "
                        "and answer must be a single A/B/C/D option."
                    )
                correct = answer == correct_letter
                option_map = {letters[i]: Path(rel_path).stem for i, (rel_path, _) in enumerate(options)}
                # region agent log
                _agent_debug_log(
                    hypothesis_id="H7",
                    message="eval_one success",
                    data={
                        "stem": stem,
                        "dim": dim,
                        "attempt": attempt + 1,
                        "answer": answer,
                        "correct": correct,
                        "nf": nf,
                    },
                    run_id="initial",
                )
                # endregion
                return {
                    "stem": stem,
                    "dim": dim,
                    "used_num_frames": nf,
                    "correct": correct,
                    "answer": answer,
                    "expected": correct_letter,
                    "reason": reason,
                    "raw": raw,
                    "option_map": option_map,
                }
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # region agent log
                _agent_debug_log(
                    hypothesis_id="H8",
                    message="eval_one exception before decision",
                    data={"stem": stem, "dim": dim, "attempt": attempt + 1, "error": str(e)[:200], "nf": nf},
                    run_id="initial",
                )
                # endregion

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
                        "used_num_frames": nf,
                        "correct": False,
                        "answer": None,
                        "expected": correct_letter,
                        "reason": None,
                        "error": str(e)[:200],
                    }

    final_err = str(last_error)[:200] if last_error else "Model length retries exhausted"
    log.error(f"vLLM model_length retries exhausted on {stem} {dim}: {final_err}")
    return {
        "stem": stem,
        "dim": dim,
        "used_num_frames": frame_candidates[-1] if frame_candidates else start_nf,
        "correct": False,
        "answer": None,
        "expected": correct_letter,
        "reason": None,
        "error": final_err,
    }


# ==================== Main ====================

def run(targets: list[str] | None = None):
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
        # region agent log
        _agent_debug_log(
            hypothesis_id="H1",
            message="CHOICES_DIR does not exist",
            data={"CHOICES_DIR": str(CHOICES_DIR), "targets": targets or [], "reason": "missing_choices_dir"},
            run_id="initial",
        )
        # endregion
        return

    all_stems = sorted(
        p.name for p in CHOICES_DIR.iterdir() if p.is_dir() and (p / "GT.mp4").exists()
    )

    # region agent log
    _agent_debug_log(
        hypothesis_id="H2",
        message="After building all_stems",
        data={
            "CHOICES_DIR": str(CHOICES_DIR),
            "num_all_stems": len(all_stems),
            "sample_all_stems": all_stems[:10],
            "targets": targets or [],
        },
        run_id="initial",
    )
    # endregion

    if targets:
        missing = [s for s in targets if s not in all_stems]
        stems = [s for s in targets if s in all_stems]
        # region agent log
        _agent_debug_log(
            hypothesis_id="H3",
            message="After filtering stems by targets",
            data={
                "targets": targets,
                "stems": stems,
                "missing": missing,
                "num_all_stems": len(all_stems),
            },
            run_id="initial",
        )
        # endregion
    else:
        stems = all_stems

    log.info(f"vLLM model: {EVAL_MODEL}")
    log.info(f"Num videos: {len(stems)}, total tasks ≈ {len(stems) * 3}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{MODEL_TAG}.json"

    # 断点续做
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        removed_invalid = 0
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

                answer = _normalize_answer_value(entry.get("answer"))
                raw = entry.get("raw")
                parsed_answer, _ = parse_eval_response(
                    raw if isinstance(raw, str) else "",
                    ANSWER_LETTERS,
                )
                if not _is_valid_answer(answer) or parsed_answer is None or parsed_answer != answer:
                    del stem_results[dim]
                    removed_invalid += 1
            if not stem_results:
                del all_results[stem]

        if removed_invalid:
            log.info(
                f"Removed {removed_invalid} entries with invalid answer/raw format from resume file"
            )
            print(
                f"[eval_vllm] 断点续做: 清理了 {removed_invalid} 条 answer 或 raw 格式不严格的记录，将重新评测这些任务"
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
    for stem in stems:
        if not (CHOICES_DIR / stem / "GT.mp4").exists():
            continue
        for dim, distractors in DIMENSIONS.items():
            if stem in all_results and dim in all_results[stem]:
                continue
            tasks.append((stem, dim, distractors))

    already_done = sum(len(v) for v in all_results.values())
    # region agent log
    _agent_debug_log(
        hypothesis_id="H4",
        message="After building tasks",
        data={
            "num_stems": len(stems),
            "stems": stems,
            "num_tasks": len(tasks),
            "already_done": already_done,
            "results_path": str(results_path),
        },
        run_id="initial",
    )
    # endregion
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

            used_nf = result.get("used_num_frames")
            with _frames_lock:
                global_nf = _effective_num_frames
            pbar.set_postfix_str(f"nf={used_nf}, global_nf={global_nf}", refresh=False)

            answer = result.get("answer")
            if not _is_valid_answer(answer) and "error" not in result:
                result["error"] = (
                    f"Invalid final answer on {stem} {dim}: {answer!r}. "
                    "Expected one of A/B/C/D after all retries."
                )

            if not _is_valid_answer(answer):
                err_msg = result.get("error") or (
                    f"Invalid final answer on {stem} {dim}: {answer!r}. "
                    "Expected one of A/B/C/D."
                )
                raw_detail = result.get("raw") or result.get("reason") or result.get("error") or ""
                raw_preview = str(raw_detail).replace("\n", "\\n")
                if len(raw_preview) > 240:
                    raw_preview = raw_preview[:240] + "..."
                log.error(f"Skip saving result with invalid answer on {stem} {dim}: {err_msg}")
                print(
                    f"[eval_vllm] WARN: {stem} {dim} answer无效({answer!r})，"
                    f"raw={raw_preview!r}，不写入JSON，后续可续跑"
                )
                continue

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
                    "error": err_msg,
                }
                try:
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    log.error(f"Failed to save results to {results_path}: {e}")
                continue

            entry = {k: v for k, v in result.items() if k not in ("stem", "dim")}
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
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    run(targets)
