"""Temporal Cloze 评测脚本（Ollama 本地模型版）

用法:
  python eval_ollama.py                                    # 评测全部题目
  python eval_ollama.py xMVtFHXrMR4.15_3                   # 评测指定题目
  python eval_ollama.py xMVtFHXrMR4.15_3 2Tz4_p9U56w.13_1  # 评测多个题目

请确保你已经在本机启动了对应的 Ollama 模型，例如:

  CUDA_VISIBLE_DEVICES=0 ollama serve
  ollama pull qwen2.5-vl

然后在本脚本中把 EVAL_MODEL 换成你实际使用的模型名称。
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


# ==================== 基本配置 ====================

# Ollama HTTP 接口地址与模型名
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
# 修改为你在本地已 pull 且可用的模型名，例如 "qwen2.5-vl"、"llava" 等
EVAL_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:27b")


def _resolve_path(env_key: str, default_rel: str) -> Path:
    val = os.environ.get(env_key, "").strip()
    if not val:
        return ROOT / default_rel
    p = Path(val)
    return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()


CHOICES_DIR = _resolve_path("CHOICES_DIR", "choices")
RESULTS_DIR = _resolve_path("EVAL_RESULTS_DIR", "eval_results")
LOG_DIR = ROOT / "logs"

# 配置
NUM_FRAMES = 16
NUM_WORKERS = 1
# 全局有效帧数：遇 413 或 Too many images 时下调，所有请求共享（线程安全）
_effective_num_frames = NUM_FRAMES
_frames_lock = threading.Lock()
# 结果文件名与日志使用模型名
MODEL_TAG = f"ollama-{EVAL_MODEL.replace('/', '_')}"

# S / A / C 三个维度的选项
DIMENSIONS = {
    "S": ["S/Rand1.mp4", "S/Rand2.mp4", "S/Rand3.mp4"],
    "A": ["A/Early.mp4", "A/Late.mp4", "A/Wide.mp4"],
    "C": ["C/Reverse.mp4", "C/Shuffle.mp4", "C/Loop.mp4"],
}

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
            "location": "eval_ollama.py",
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
    """从 Ollama /v1/chat/completions JSON 中提取文本内容。"""
    if not resp_json:
        raise ValueError("Empty response from Ollama")
    choices = resp_json.get("choices") or []
    if not choices:
        raise ValueError("Response has no choices")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    reasoning = msg.get("reasoning")

    # 优先使用 content；若为空则回退到 reasoning（兼容 Ollama 把文本放在 reasoning 里的情况）
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
    elif isinstance(content, list):
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

    # content 为空时，尝试使用 reasoning
    if isinstance(reasoning, str):
        text = reasoning.strip()
        if text:
            return text

    raise ValueError("Message has neither usable content nor reasoning")


def _is_html_response(raw: str) -> bool:
    """接口偶尔可能返回 HTML 错误页，这种要忽略。"""
    s = (raw or "").strip()
    if not s or len(s) < 50:
        return False
    s_lower = s[:500].lower()
    return s_lower.startswith("<!doctype") or s_lower.startswith("<html")


def parse_eval_response(raw: str, letters: list[str]) -> tuple[str | None, str]:
    """从模型返回文本中尽量解析出 answer 和 reason。letters 为可选字母如 ['A','B','C','D']。"""
    raw = (raw or "").strip()
    if not raw:
        return None, ""
    if _is_html_response(raw):
        return None, ""

    # 1) 去掉 markdown 代码块包裹
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # 2) 尝试直接解析 JSON
    try:
        parsed = json.loads(raw)
        answer = (parsed.get("answer") or "").strip().upper()
        reason = parsed.get("reason") or ""
        if answer in letters:
            return answer, reason
    except json.JSONDecodeError:
        pass

    # 3) 尝试从首尾截取 {...} 再解析（前后可能有说明文字）
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            answer = (parsed.get("answer") or "").strip().upper()
            reason = parsed.get("reason") or ""
            if answer in letters:
                return answer, reason
        except json.JSONDecodeError:
            pass

    # 4) 解析失败：从 raw 中抽取 answer（第一个出现的可选字母）和 reason 字段值
    answer: str | None = None
    for c in raw.upper():
        if c in letters:
            answer = c
            break
    reason = raw
    m = re.search(r'"reason"\s*:\s*"', raw)
    if m:
        start = m.end()
        i = start
        while i < len(raw):
            if raw[i] == "\\" and i + 1 < len(raw):
                i += 2
                continue
            if raw[i] == '"':
                reason = raw[start:i].replace("\\n", "\n").replace('\\"', '"')
                break
            i += 1
        else:
            reason = raw[start:].replace("\\n", "\n").replace('\\"', '"')
    return answer, reason


# ==================== 单题评测（调用 Ollama） ====================

def _ollama_chat(content: list[dict]) -> dict:
    """调用 Ollama 的 /v1/chat/completions 接口。"""
    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    payload = {
        "model": EVAL_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2048,
        "temperature": 0,
    }
    # region agent log
    _agent_debug_log(
        hypothesis_id="H9",
        message="Preparing Ollama HTTP request",
        data={"base_url": OLLAMA_BASE_URL, "url": url, "model": EVAL_MODEL},
        run_id="initial",
    )
    # endregion
    resp = requests.post(url, json=payload, timeout=600)
    # region agent log
    _agent_debug_log(
        hypothesis_id="H10",
        message="Ollama HTTP response received",
        data={"status_code": resp.status_code, "ok": resp.ok},
        run_id="initial",
    )
    # endregion
    resp.raise_for_status()
    return resp.json()


def eval_one(stem: str, dim: str, distractors: list[str]) -> dict:
    """对一个视频的一个维度做 1-of-4 评测。使用全局 _effective_num_frames（遇图像过多报错时共享下调）。"""
    global _effective_num_frames
    base = CHOICES_DIR / stem
    with _frames_lock:
        nf = _effective_num_frames

    # region agent log
    _agent_debug_log(
        hypothesis_id="H5",
        message="eval_one start",
        data={"stem": stem, "dim": dim, "nf": nf, "base": str(base)},
        run_id="initial",
    )
    # endregion

    before_b64 = sample_and_encode(base / "before.mp4", nf)
    after_b64 = sample_and_encode(base / "after.mp4", nf)

    # GT + 3个干扰，打乱
    options = [("GT.mp4", True)] + [(d, False) for d in distractors]
    random.shuffle(options)
    letters = [chr(65 + i) for i in range(len(options))]
    correct_letter = next(letters[i] for i, (_, is_gt) in enumerate(options) if is_gt)

    # 构建 content（OpenAI/Ollama 兼容的多模态格式）
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

    # 调用 Ollama，最多重试 3 次
    for attempt in range(3):
        try:
            # region agent log
            _agent_debug_log(
                hypothesis_id="H6",
                message="Calling _ollama_chat",
                data={"stem": stem, "dim": dim, "attempt": attempt + 1, "num_images": len(content)},
                run_id="initial",
            )
            # endregion
            resp_json = _ollama_chat(content)
            raw = _extract_response_text(resp_json)
            if not raw:
                raise ValueError("Empty response from Ollama")
            if _is_html_response(raw):
                raise ValueError("Ollama returned HTML instead of model output")
            answer, reason = parse_eval_response(raw, letters)
            correct = answer == correct_letter
            # region agent log
            _agent_debug_log(
                hypothesis_id="H7",
                message="eval_one success",
                data={"stem": stem, "dim": dim, "attempt": attempt + 1, "answer": answer, "correct": correct},
                run_id="initial",
            )
            # endregion
            return {
                "stem": stem,
                "dim": dim,
                "correct": correct,
                "answer": answer,
                "expected": correct_letter,
                "reason": reason,
            }
        except Exception as e:
            err_str = str(e).lower()
            # region agent log
            _agent_debug_log(
                hypothesis_id="H8",
                message="eval_one exception before decision",
                data={"stem": stem, "dim": dim, "attempt": attempt + 1, "error": str(e)[:200]},
                run_id="initial",
            )
            # endregion
            # Too many images 等错误：全局减帧再重试
            if ("too many images" in err_str or "413" in err_str or "entity too large" in err_str) and nf > 5:
                with _frames_lock:
                    _effective_num_frames = min(_effective_num_frames, max(5, nf - 4))
                    new_nf = _effective_num_frames
                log.info(f"Ollama image limit on {stem} {dim}, set global num_frames={new_nf}, retry")
                return eval_one(stem, dim, distractors)
            if attempt < 2:
                wait = (attempt + 1) * 5
                time.sleep(wait)
            else:
                log.error(f"Ollama error on {stem} {dim}: {e}")
                # region agent log
                _agent_debug_log(
                    hypothesis_id="H9",
                    message="eval_one failed after retries",
                    data={"stem": stem, "dim": dim, "error": str(e)[:200]},
                    run_id="initial",
                )
                # endregion
                return {
                    "stem": stem,
                    "dim": dim,
                    "correct": False,
                    "answer": None,
                    "expected": correct_letter,
                    "reason": None,
                    "error": str(e)[:200],
                }


# ==================== Main ====================

def run(targets: list[str] | None = None):
    """targets: 指定题目 stem 列表, None 则全部评测"""
    global _effective_num_frames
    _effective_num_frames = NUM_FRAMES  # 每次 run 开始时重置，本轮任务共享

    if not CHOICES_DIR.exists():
        log.error(f"Choices directory not found: {CHOICES_DIR}")
        print(f"[eval_ollama] 错误: 题目目录不存在: {CHOICES_DIR}")
        print(
            "[eval_ollama] 请创建该目录并放入题目（每题为子目录且含 GT.mp4），"
            "或设置环境变量 CHOICES_DIR，例如: CHOICES_DIR=Videos-LVD2M/choices python eval_ollama.py"
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

    log.info(f"Ollama model: {EVAL_MODEL}")
    log.info(f"Num videos: {len(stems)}, total tasks ≈ {len(stems) * 3}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{MODEL_TAG}.json"

    # 断点续做
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        done_count = sum(len(v) for v in all_results.values())
        done_stems = sorted(all_results.keys())
        last_stem = done_stems[-1] if done_stems else None
        log.info(f"结果文件: {results_path}, 已写入 {len(done_stems)} 个题目共 {done_count} 条结果")
        if last_stem is not None:
            log.info(f"最后有记录的题目: {last_stem} (维度: {list(all_results[last_stem].keys())})")
        print(
            f"[eval_ollama] 断点续做: 结果文件 {results_path.name}, "
            f"已完成 {done_count} 题, 最后题目 {last_stem or '-'}, 将跳过已测题目继续评测"
        )
    else:
        all_results = {}
        print(f"[eval_ollama] 结果文件不存在，将从头评测并保存到 {results_path}")

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
            f"[eval_ollama] 没有待评测任务（共 {len(stems)} 个题目，已完成 {already_done} 条）。"
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

        for fut in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Evaluating (Ollama)"):
            stem, dim = future_to_task[fut]
            try:
                result = fut.result()
            except Exception as e:
                log.error(f"Unexpected error in worker for {stem} {dim}: {e}")
                continue

            if "error" in result:
                err_msg = result["error"]
                log.error(f"Ollama error on {stem} {dim}: {err_msg}")
                print(f"[eval_ollama] ERROR: {stem} {dim} Ollama error: {err_msg}")
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

