"""Temporal Cloze evaluator for Video-LLaVA-7B-hf via Transformers.

Usage:
  python eval_video_llava_hf.py
  python eval_video_llava_hf.py xMVtFHXrMR4.15_3
  python eval_video_llava_hf.py xMVtFHXrMR4.15_3 2Tz4_p9U56w.13_1

Optional environment variables:
  CHOICES_DIR=/users/henry/Temporal-Cloze/video-cloze/choices
  EVAL_RESULTS_DIR=/users/henry/Temporal-Cloze/video-cloze/eval_results
  HF_MODEL=LanguageBind/Video-LLaVA-7B-hf
  HF_DEVICE_MAP=auto
  HF_DTYPE=float16
  HF_TRUST_REMOTE_CODE=1
  NUM_FRAMES=8
  NUM_WORKERS=4
  BATCH_SIZE=1
  MAX_NEW_TOKENS=128
  REPETITION_PENALTY=1.05
  MAX_REASON_CHARS=500
  RESET_RESULTS=0
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
os.environ.setdefault("CHOICES_DIR", "/users/henry/Temporal-Cloze/video-cloze/choices")
os.environ.setdefault("EVAL_RESULTS_DIR", "/users/henry/Temporal-Cloze/video-cloze/eval_results")


def _resolve_path(env_key: str, default_rel: str) -> Path:
    val = os.environ.get(env_key, "").strip()
    if not val:
        return ROOT / default_rel
    p = Path(val)
    return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _resolve_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    out = dtype_map.get(key, torch.float16)
    if not torch.cuda.is_available() and out in {torch.float16, torch.bfloat16}:
        return torch.float32
    return out


# -------------------- Config --------------------

NUM_FRAMES = _env_int("NUM_FRAMES", 8)
NUM_WORKERS = _env_int("NUM_WORKERS", 4)
BATCH_SIZE = _env_int("BATCH_SIZE", 1)
MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 128)
REPETITION_PENALTY = _env_float("REPETITION_PENALTY", 1.05)
MAX_REASON_CHARS = _env_int("MAX_REASON_CHARS", 500)
RESET_RESULTS = _env_bool("RESET_RESULTS", False)

HF_MODEL = os.environ.get("HF_MODEL", "LanguageBind/Video-LLaVA-7B-hf").strip()
HF_DEVICE_MAP = os.environ.get("HF_DEVICE_MAP", "auto").strip()
HF_DTYPE = _resolve_dtype(os.environ.get("HF_DTYPE", "float16"))
HF_TRUST_REMOTE_CODE = _env_bool("HF_TRUST_REMOTE_CODE", True)

CHOICES_DIR = _resolve_path("CHOICES_DIR", "video-cloze/choices")
RESULTS_DIR = _resolve_path("EVAL_RESULTS_DIR", "video-cloze/eval_results")
LOG_DIR = ROOT / "logs"
MODEL_TAG = f"hf-{HF_MODEL.replace('/', '_')}"

DIMENSIONS = {
    "S": ["S/Rand1.mp4", "S/Rand2.mp4", "S/Rand3.mp4"],
    "A": ["A/Early.mp4", "A/Late.mp4", "A/Wide.mp4"],
    "C": ["C/Reverse.mp4", "C/Shuffle.mp4", "C/Loop.mp4"],
}
_VALID_ANSWERS = {"A", "B", "C", "D"}

# Shared effective frame count (lowered on OOM), process-wide.
_effective_num_frames = NUM_FRAMES
_frames_lock = threading.Lock()

# Shared model objects.
_load_lock = threading.Lock()
_infer_lock = threading.Lock()
_processor: VideoLlavaProcessor | None = None
_model: VideoLlavaForConditionalGeneration | None = None
_input_device = torch.device("cpu")

LOG_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
_fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", datefmt="%H:%M:%S")
_fh = logging.FileHandler(LOG_DIR / f"eval_{MODEL_TAG}.log", encoding="utf-8")
_fh.setFormatter(_fmt)
log.addHandler(_fh)


# -------------------- Model --------------------

def _pick_input_device(model: VideoLlavaForConditionalGeneration) -> torch.device:
    if hasattr(model, "hf_device_map"):
        dev_map = getattr(model, "hf_device_map")
        if isinstance(dev_map, dict):
            for dev in dev_map.values():
                if isinstance(dev, str) and dev not in {"cpu", "disk"}:
                    return torch.device(dev)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_model_and_processor() -> tuple[VideoLlavaForConditionalGeneration, VideoLlavaProcessor]:
    global _model, _processor, _input_device
    if _model is not None and _processor is not None:
        return _model, _processor

    with _load_lock:
        if _model is not None and _processor is not None:
            return _model, _processor

        log.info(
            "Loading model=%s device_map=%s dtype=%s trust_remote_code=%s",
            HF_MODEL,
            HF_DEVICE_MAP,
            HF_DTYPE,
            HF_TRUST_REMOTE_CODE,
        )
        processor = VideoLlavaProcessor.from_pretrained(
            HF_MODEL,
            trust_remote_code=HF_TRUST_REMOTE_CODE,
        )
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            HF_MODEL,
            trust_remote_code=HF_TRUST_REMOTE_CODE,
            torch_dtype=HF_DTYPE,
            device_map=HF_DEVICE_MAP if HF_DEVICE_MAP else "auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

        _processor = processor
        _model = model
        _input_device = _pick_input_device(model)
        log.info("Model loaded. Input device=%s", _input_device)

    return _model, _processor


# -------------------- Data --------------------

def sample_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    n = min(num_frames, total)
    indices = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]

    out: list[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append(Image.fromarray(rgb))
    cap.release()
    return out


# -------------------- Prompt & Parse --------------------

def build_prompt() -> str:
    return """You are given:
- BEGINNING: the first part of a video (frames in temporal order).
- END: the last part of the same video (frames in temporal order).
The middle segment between BEGINNING and END was removed.

You will then see four candidate middle segments, labeled A, B, C, D.
Each candidate is a short clip; exactly one is the true middle.

Task: choose the correct middle segment.

Output STRICT JSON only: {"answer":"<A|B|C|D>","reason":"<<=20 words>"}"""


def _is_html_response(raw: str) -> bool:
    s = (raw or "").strip()
    if not s or len(s) < 50:
        return False
    s_lower = s[:500].lower()
    return s_lower.startswith("<!doctype") or s_lower.startswith("<html")


def parse_eval_response(raw: str, letters: list[str]) -> tuple[str | None, str]:
    raw = (raw or "").strip()
    if not raw or _is_html_response(raw):
        return None, ""

    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
        answer = (parsed.get("answer") or "").strip().upper()
        reason = parsed.get("reason") or ""
        if answer in letters:
            return answer, reason
    except json.JSONDecodeError:
        pass

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

    def _extract_reason_fallback(text: str) -> str:
        m_reason = re.search(
            r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"',
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m_reason:
            return ""
        return (
            m_reason.group(1)
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .strip()
        )

    # Strict fallback: only accept explicit answer patterns.
    m = re.search(r'"answer"\s*:\s*"?\s*([A-D])\s*"?', raw, flags=re.IGNORECASE)
    if m:
        answer = m.group(1).upper()
        if answer in letters:
            return answer, _extract_reason_fallback(raw)

    m = re.search(r"\b(?:answer|option|choice)\s*[:=]\s*([A-D])\b", raw, flags=re.IGNORECASE)
    if m:
        answer = m.group(1).upper()
        if answer in letters:
            return answer, _extract_reason_fallback(raw)

    m = re.fullmatch(r"\s*([A-D])\s*", raw, flags=re.IGNORECASE)
    if m:
        answer = m.group(1).upper()
        if answer in letters:
            return answer, ""

    # Do not guess by scanning random chars in malformed output.
    return None, ""


def parse_answer_only_response(raw: str, letters: list[str]) -> str | None:
    text = (raw or "").strip().upper()
    if not text or _is_html_response(text):
        return None

    m = re.search(r"\b([A-D])\b", text)
    if m and m.group(1) in letters:
        return m.group(1)
    if len(text) == 1 and text in letters:
        return text
    return None


def _build_content_and_images(
    before_frames: list[Image.Image],
    after_frames: list[Image.Image],
    options: list[tuple[str, bool]],
    letters: list[str],
    base_dir: Path,
    nf: int,
) -> tuple[list[dict[str, str]], list[Image.Image], dict[str, str]]:
    content: list[dict[str, str]] = [{"type": "text", "text": build_prompt()}]
    images: list[Image.Image] = []
    option_map: dict[str, str] = {}

    content.append({"type": "text", "text": "[BEGINNING]"})
    for img in before_frames:
        content.append({"type": "image"})
        images.append(img)

    content.append({"type": "text", "text": "[END]"})
    for img in after_frames:
        content.append({"type": "image"})
        images.append(img)

    for i, (rel_path, _) in enumerate(options):
        letter = letters[i]
        content.append({"type": "text", "text": f"[Candidate {letter}]"})
        option_map[letter] = Path(rel_path).stem
        opt_frames = sample_frames(base_dir / rel_path, nf)
        for img in opt_frames:
            content.append({"type": "image"})
            images.append(img)

    content.append({"type": "text", "text": 'Return JSON only, with keys "answer" and "reason".'})
    return content, images, option_map


def _fallback_prompt_from_content(content: list[dict[str, str]]) -> str:
    # Fallback for processors that do not support multimodal chat template.
    lines: list[str] = []
    for item in content:
        if item.get("type") == "text":
            lines.append(item.get("text", ""))
        elif item.get("type") == "image":
            lines.append("<image>")
    return f"USER:\n{'\n'.join(lines)}\nASSISTANT:\n"


def _content_to_prompt(content: list[dict[str, str]]) -> str:
    # Video-LLaVA checkpoints typically do not provide a chat template.
    # Use explicit USER/ASSISTANT prompt format directly to avoid noisy fallback warnings.
    return _fallback_prompt_from_content(content)


# -------------------- Inference --------------------

def _prepare_model_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    model_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            model_inputs[key] = value.to(_input_device)
        else:
            model_inputs[key] = value
    return model_inputs


def _generation_kwargs(processor: VideoLlavaProcessor) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "use_cache": True,
        "repetition_penalty": REPETITION_PENALTY,
    }
    if getattr(processor, "tokenizer", None) is not None:
        eos_id = getattr(processor.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            kwargs["pad_token_id"] = eos_id
    return kwargs


def _decode_outputs(
    processor: VideoLlavaProcessor,
    output_ids: torch.Tensor,
    model_inputs: dict[str, Any],
) -> list[str]:
    if "attention_mask" in model_inputs:
        input_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
    else:
        input_lengths = [model_inputs["input_ids"].shape[-1]] * output_ids.shape[0]

    outputs: list[str] = []
    for idx, input_len in enumerate(input_lengths):
        gen_ids = output_ids[idx, int(input_len) :]
        text = processor.batch_decode(
            gen_ids.unsqueeze(0),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        outputs.append(text)
    return outputs


def _hf_chat_single(prompt: str, images: list[Image.Image]) -> str:
    model, processor = _get_model_and_processor()
    raw_inputs = processor(text=prompt, images=images, return_tensors="pt")
    model_inputs = _prepare_model_inputs(raw_inputs)
    kwargs = _generation_kwargs(processor)

    with _infer_lock:
        with torch.inference_mode():
            output_ids = model.generate(**model_inputs, **kwargs)
    return _decode_outputs(processor, output_ids, model_inputs)[0]


def _hf_chat_single_short_retry(prompt: str, images: list[Image.Image]) -> str:
    model, processor = _get_model_and_processor()
    retry_prompt = (
        f"{prompt}\n"
        "Return exactly one JSON object with keys answer and reason. "
        "answer must be one of A,B,C,D."
    )
    raw_inputs = processor(text=retry_prompt, images=images, return_tensors="pt")
    model_inputs = _prepare_model_inputs(raw_inputs)
    kwargs = _generation_kwargs(processor)
    kwargs["max_new_tokens"] = min(64, MAX_NEW_TOKENS)

    with _infer_lock:
        with torch.inference_mode():
            output_ids = model.generate(**model_inputs, **kwargs)
    return _decode_outputs(processor, output_ids, model_inputs)[0]


def _hf_chat_single_answer_only(prompt: str, images: list[Image.Image], letters: list[str]) -> str:
    model, processor = _get_model_and_processor()
    answer_prompt = (
        f"{prompt}\n"
        f"Output only one capital letter from {','.join(letters)}.\n"
        "Answer:"
    )
    raw_inputs = processor(text=answer_prompt, images=images, return_tensors="pt")
    model_inputs = _prepare_model_inputs(raw_inputs)
    kwargs = _generation_kwargs(processor)
    kwargs["max_new_tokens"] = min(4, MAX_NEW_TOKENS)
    kwargs["repetition_penalty"] = 1.0

    with _infer_lock:
        with torch.inference_mode():
            output_ids = model.generate(**model_inputs, **kwargs)
    return _decode_outputs(processor, output_ids, model_inputs)[0]


def _hf_pick_answer_from_logits(prompt: str, images: list[Image.Image], letters: list[str]) -> str | None:
    model, processor = _get_model_and_processor()
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None

    choice_prompt = (
        f"{prompt}\n"
        f"Choose one letter from {','.join(letters)}.\n"
        "Answer:"
    )
    raw_inputs = processor(text=choice_prompt, images=images, return_tensors="pt")
    model_inputs = _prepare_model_inputs(raw_inputs)

    def _candidate_token_ids(letter: str) -> list[int]:
        cands = set()
        for text in (letter, f" {letter}", f"\n{letter}", f'"{letter}"'):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                cands.add(int(ids[0]))
        return list(cands)

    with _infer_lock:
        with torch.inference_mode():
            outputs = model(**model_inputs)
            next_token_logits = outputs.logits[0, -1, :]

    scores: dict[str, float] = {}
    for letter in letters:
        token_ids = _candidate_token_ids(letter)
        if not token_ids:
            continue
        token_tensor = torch.tensor(token_ids, device=next_token_logits.device, dtype=torch.long)
        scores[letter] = float(next_token_logits.index_select(0, token_tensor).max().item())

    if not scores:
        return None
    return max(scores.items(), key=lambda x: x[1])[0]


def _hf_chat_batch(prompts: list[str], image_batches: list[list[Image.Image]]) -> list[str]:
    model, processor = _get_model_and_processor()
    raw_inputs = processor(
        text=prompts,
        images=image_batches,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = _prepare_model_inputs(raw_inputs)
    kwargs = _generation_kwargs(processor)

    with _infer_lock:
        with torch.inference_mode():
            output_ids = model.generate(**model_inputs, **kwargs)
    return _decode_outputs(processor, output_ids, model_inputs)


# -------------------- Eval --------------------

def _prepare_case(stem: str, dim: str, distractors: list[str], nf: int) -> dict[str, Any]:
    base = CHOICES_DIR / stem
    before_frames = sample_frames(base / "before.mp4", nf)
    after_frames = sample_frames(base / "after.mp4", nf)

    options = [("GT.mp4", True)] + [(d, False) for d in distractors]
    random.shuffle(options)
    letters = [chr(65 + i) for i in range(len(options))]
    correct_letter = next(letters[i] for i, (_, is_gt) in enumerate(options) if is_gt)

    content, images, option_map = _build_content_and_images(
        before_frames=before_frames,
        after_frames=after_frames,
        options=options,
        letters=letters,
        base_dir=base,
        nf=nf,
    )
    prompt = _content_to_prompt(content)
    return {
        "stem": stem,
        "dim": dim,
        "letters": letters,
        "correct_letter": correct_letter,
        "prompt": prompt,
        "images": images,
        "option_map": option_map,
    }


def _prepare_batch_cases(batch_tasks: list[tuple[str, str, list[str]]], nf: int) -> list[dict[str, Any]]:
    workers = max(1, min(NUM_WORKERS, len(batch_tasks)))
    if workers == 1:
        return [_prepare_case(stem, dim, distractors, nf) for stem, dim, distractors in batch_tasks]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_prepare_case, stem, dim, distractors, nf)
            for stem, dim, distractors in batch_tasks
        ]
        return [f.result() for f in futures]


def _reduce_frames_on_oom(nf: int) -> bool:
    global _effective_num_frames
    if nf <= 4:
        return False
    with _frames_lock:
        _effective_num_frames = min(_effective_num_frames, max(4, nf - 2))
        new_nf = _effective_num_frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("OOM detected, reduce global num_frames from %s to %s", nf, new_nf)
    return True


def _batch_results_from_cases(cases: list[dict[str, Any]], raws: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case, raw in zip(cases, raws):
        answer, reason = parse_eval_response(raw, case["letters"])
        if isinstance(reason, str) and len(reason) > MAX_REASON_CHARS:
            reason = reason[:MAX_REASON_CHARS] + "...[truncated]"
        results.append(
            {
                "stem": case["stem"],
                "dim": case["dim"],
                "correct": answer == case["correct_letter"],
                "answer": answer,
                "expected": case["correct_letter"],
                "reason": reason,
                "option_map": case["option_map"],
            }
        )
    return results


def _infer_batch_with_retry(cases: list[dict[str, Any]], nf: int) -> tuple[list[dict[str, Any]], bool]:
    prompts = [case["prompt"] for case in cases]
    image_batches = [case["images"] for case in cases]

    # Try true batched generate first.
    for attempt in range(2):
        try:
            raws = _hf_chat_batch(prompts, image_batches)
            results = _batch_results_from_cases(cases, raws)
            for i, result in enumerate(results):
                if result.get("answer") is not None:
                    continue
                try:
                    answer_only_raw = _hf_chat_single_answer_only(
                        cases[i]["prompt"],
                        cases[i]["images"],
                        cases[i]["letters"],
                    )
                    answer = parse_answer_only_response(answer_only_raw, cases[i]["letters"])
                    if answer is not None:
                        result["answer"] = answer
                        result["reason"] = ""
                        result["correct"] = answer == cases[i]["correct_letter"]
                        continue
                    retry_raw = _hf_chat_single_short_retry(cases[i]["prompt"], cases[i]["images"])
                    answer, reason = parse_eval_response(retry_raw, cases[i]["letters"])
                    if isinstance(reason, str) and len(reason) > MAX_REASON_CHARS:
                        reason = reason[:MAX_REASON_CHARS] + "...[truncated]"
                    result["answer"] = answer
                    result["reason"] = reason
                    result["correct"] = answer == cases[i]["correct_letter"]
                except Exception:
                    pass
            return results, False
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if _reduce_frames_on_oom(nf):
                    return [], True
            if attempt == 0:
                time.sleep(2)
                continue
            break
        except Exception:
            if attempt == 0:
                time.sleep(2)
                continue
            break

    # Fallback: per-sample inference to keep robustness if processor batching fails.
    results: list[dict[str, Any]] = []
    for case in cases:
        for attempt in range(2):
            try:
                raw = _hf_chat_single(case["prompt"], case["images"])
                answer, reason = parse_eval_response(raw, case["letters"])
                if answer is None:
                    answer_only_raw = _hf_chat_single_answer_only(
                        case["prompt"],
                        case["images"],
                        case["letters"],
                    )
                    answer = parse_answer_only_response(answer_only_raw, case["letters"])
                    if answer is not None:
                        reason = ""
                if answer is None:
                    retry_raw = _hf_chat_single_short_retry(case["prompt"], case["images"])
                    answer, reason = parse_eval_response(retry_raw, case["letters"])
                if isinstance(reason, str) and len(reason) > MAX_REASON_CHARS:
                    reason = reason[:MAX_REASON_CHARS] + "...[truncated]"
                results.append(
                    {
                        "stem": case["stem"],
                        "dim": case["dim"],
                        "correct": answer == case["correct_letter"],
                        "answer": answer,
                        "expected": case["correct_letter"],
                        "reason": reason,
                        "option_map": case["option_map"],
                    }
                )
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and _reduce_frames_on_oom(nf):
                    return [], True
                if attempt == 0:
                    time.sleep(2)
                    continue
                results.append(
                    {
                        "stem": case["stem"],
                        "dim": case["dim"],
                        "correct": False,
                        "answer": None,
                        "expected": case["correct_letter"],
                        "reason": None,
                        "error": str(e)[:300],
                    }
                )
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 0:
                    time.sleep(2)
                    continue
                results.append(
                    {
                        "stem": case["stem"],
                        "dim": case["dim"],
                        "correct": False,
                        "answer": None,
                        "expected": case["correct_letter"],
                        "reason": None,
                        "error": str(e)[:300],
                    }
                )
                break
    return results, False


def run(targets: list[str] | None = None) -> None:
    global _effective_num_frames
    _effective_num_frames = NUM_FRAMES

    if not CHOICES_DIR.exists():
        print(f"[eval_video_llava_hf] Missing CHOICES_DIR: {CHOICES_DIR}")
        return

    all_stems = sorted(
        p.name for p in CHOICES_DIR.iterdir() if p.is_dir() and (p / "GT.mp4").exists()
    )
    stems = [s for s in (targets or all_stems) if s in all_stems]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{MODEL_TAG}.json"
    if results_path.exists() and not RESET_RESULTS:
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"[eval_video_llava_hf] Resume from {results_path.name}")
    else:
        all_results = {}
        if results_path.exists() and RESET_RESULTS:
            print(f"[eval_video_llava_hf] RESET_RESULTS=1, overwrite {results_path.name}")
        else:
            print(f"[eval_video_llava_hf] Start new run -> {results_path.name}")

    tasks: list[tuple[str, str, list[str]]] = []
    for stem in stems:
        for dim, distractors in DIMENSIONS.items():
            cached = all_results.get(stem, {}).get(dim)
            if isinstance(cached, dict) and cached.get("answer") in _VALID_ANSWERS:
                continue
            if cached is not None:
                log.info("Re-run invalid cached result: stem=%s dim=%s answer=%s", stem, dim, cached.get("answer"))
            tasks.append((stem, dim, distractors))

    if not tasks:
        print("[eval_video_llava_hf] No tasks to run.")
        return

    _get_model_and_processor()
    batch_size = max(1, BATCH_SIZE)
    print(
        f"[eval_video_llava_hf] tasks={len(tasks)} batch_size={batch_size} "
        f"num_workers={NUM_WORKERS} num_frames={NUM_FRAMES}"
    )

    index = 0
    with tqdm(total=len(tasks), desc="Evaluating (HF)") as pbar:
        while index < len(tasks):
            current_batch = tasks[index : index + batch_size]
            with _frames_lock:
                nf = _effective_num_frames
            try:
                cases = _prepare_batch_cases(current_batch, nf)
            except Exception as e:  # noqa: BLE001
                for stem, dim, _ in current_batch:
                    log.error("Prepare failed on %s %s: %s", stem, dim, e)
                    pbar.update(1)
                index += len(current_batch)
                continue

            batch_results, should_retry = _infer_batch_with_retry(cases, nf)
            if should_retry:
                # Retry same batch with reduced frame count.
                continue

            for result in batch_results:
                stem = result["stem"]
                dim = result["dim"]

                if "error" in result:
                    log.error("HF error on %s %s: %s", stem, dim, result["error"])
                    print(f"[eval_video_llava_hf] ERROR: {stem} {dim}: {result['error']}")
                    pbar.update(1)
                    continue

                entry = {k: v for k, v in result.items() if k not in ("stem", "dim")}
                if stem not in all_results:
                    all_results[stem] = {}
                all_results[stem][dim] = entry

                tag = "OK" if result["correct"] else "X"
                tqdm.write(
                    f"  {tag} {stem} {dim} ans={result['answer']} exp={result['expected']} "
                    f"{(result.get('reason') or '')[:60]}"
                )
                pbar.update(1)

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            index += len(current_batch)

    for dim in DIMENSIONS:
        entries = [v[dim] for v in all_results.values() if dim in v]
        c = sum(1 for e in entries if e["correct"])
        t = len(entries)
        if t:
            print(f"{dim}: {c}/{t} = {c / t:.2%}")
        else:
            print(f"{dim}: no data")

    all_entries = [e for v in all_results.values() for e in v.values()]
    total = len(all_entries)
    correct = sum(1 for e in all_entries if e["correct"])
    if total:
        print(f"Overall: {correct}/{total} = {correct / total:.2%}")
    else:
        print("Overall: no data")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    run(targets)
