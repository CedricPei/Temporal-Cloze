"""DINOv2 edge-matching baseline for TempCloze.

This is a non-Video-LLM, training-free baseline. It uses frozen DINOv2 image
features on the same 16 bin-centered sampled frames used by the main
evaluation scripts, then scores each candidate by the cosine similarity of its
two temporal boundary transitions.

Outputs a compact JSON file with per-question edge scores, linearly normalized
probabilities, ground-truth probability, and margin summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).parent
DEFAULT_CHOICES_DIR = ROOT / "choices"
DEFAULT_RESULTS_DIR = ROOT / "eval_results" / "baselines"

NUM_FRAMES = 16
MAX_HEIGHT = 360
EDGE_FRAMES = 4
PRIMARY_VARIANT = f"edge{EDGE_FRAMES}_linear_probability"
VALID_DIMS = ("S", "A", "C")
DISPLAY_DIMS = {"S": "Semantic", "A": "Alignment", "C": "Progression"}
OPTION_LETTERS = ("A", "B", "C", "D")
QUESTION_FIELDS = (
    "video_id",
    "dimension",
    "scores",
    "probabilities",
    "ground_truth_index",
    "p_gt",
    "predicted_index",
    "correct",
    "gt_margin",
)

DIMENSIONS = {
    "S": ["S/Rand1.mp4", "S/Rand2.mp4", "S/Rand3.mp4"],
    "A": ["A/Early.mp4", "A/Late.mp4", "A/Wide.mp4"],
    "C": ["C/Reverse.mp4", "C/Shuffle.mp4", "C/Loop.mp4"],
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _jsonable_float(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def _normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec))
    if denom <= 0 or not math.isfinite(denom):
        return vec.astype(np.float32, copy=False)
    return (vec / denom).astype(np.float32, copy=False)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)
    return float(np.dot(a, b))


def bin_centered_indices(total_frames: int, num_frames: int = NUM_FRAMES) -> list[int]:
    if total_frames <= 2:
        return []
    n = min(num_frames, total_frames - 2)
    return [1 + int((i + 0.5) * (total_frames - 2) / n) for i in range(n)]


def edge_indices_from_sampled(
    sampled_indices: list[int],
    role: str,
    max_edge_frames: int = EDGE_FRAMES,
) -> list[int]:
    if role == "before":
        return sampled_indices[-max_edge_frames:]
    if role == "after":
        return sampled_indices[:max_edge_frames]
    if role == "candidate":
        selected: list[int] = []
        seen: set[int] = set()
        n = len(sampled_indices)
        positions = list(range(min(max_edge_frames, n)))
        positions.extend(range(max(0, n - max_edge_frames), n))
        for pos in positions:
            idx = sampled_indices[pos]
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(idx)
        return selected
    raise ValueError(f"Unknown clip role: {role}")


def read_video_indices(video_path: Path, indices: list[int], max_height: int = MAX_HEIGHT) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if h > max_height:
            scale = max_height / h
            frame = cv2.resize(frame, (int(w * scale), max_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def sample_frames(
    video_path: Path,
    num_frames: int = NUM_FRAMES,
    max_height: int = MAX_HEIGHT,
) -> list[np.ndarray]:
    """Sample frames with the same bin-centered indices as the main eval."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return read_video_indices(video_path, bin_centered_indices(total, num_frames), max_height)


def sample_edge_frames(
    video_path: Path,
    role: str,
    num_frames: int = NUM_FRAMES,
    max_height: int = MAX_HEIGHT,
    edge_frames: int = EDGE_FRAMES,
) -> list[np.ndarray]:
    """Read only the edge frames selected from the main 16 sampled indices."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    sampled = bin_centered_indices(total, num_frames)
    return read_video_indices(video_path, edge_indices_from_sampled(sampled, role, edge_frames), max_height)


class DinoEncoder:
    """Frozen DINOv2 frame encoder backed by HuggingFace Transformers."""

    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "DINOv2 baseline requires working PyTorch, Transformers, and Pillow. "
                "Install/fix them before running the baseline."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.torch = torch
        self.image_cls = Image
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)

    def encode_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("Cannot encode an empty frame list")

        outputs: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(frames), self.batch_size):
                batch = frames[start : start + self.batch_size]
                images = [self.image_cls.fromarray(frame) for frame in batch]
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                if hasattr(self.model, "get_image_features"):
                    feats = self.model.get_image_features(**inputs)
                else:
                    model_out = self.model(**inputs)
                    feats = getattr(model_out, "pooler_output", None)
                    if feats is None:
                        feats = model_out.last_hidden_state[:, 0]
                feats = self.torch.nn.functional.normalize(feats.float(), dim=-1)
                outputs.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(outputs, axis=0)


@dataclass
class Candidate:
    letter: str
    rel_path: str
    is_gt: bool

    @property
    def clip_name(self) -> str:
        return Path(self.rel_path).stem


def build_candidates(stem: str, dim: str, seed: int) -> tuple[list[Candidate], str]:
    options = [("GT.mp4", True)] + [(rel, False) for rel in DIMENSIONS[dim]]
    rng = random.Random(f"{seed}:{stem}:{dim}")
    rng.shuffle(options)
    candidates = [
        Candidate(chr(65 + idx), rel_path, is_gt)
        for idx, (rel_path, is_gt) in enumerate(options)
    ]
    expected = next(c.letter for c in candidates if c.is_gt)
    return candidates, expected


def score_candidate(
    before_feats: np.ndarray,
    candidate_feats: np.ndarray,
    after_feats: np.ndarray,
) -> dict[str, float | None]:
    for name, features, min_len in (
        ("before", before_feats, EDGE_FRAMES),
        ("candidate", candidate_feats, EDGE_FRAMES * 2),
        ("after", after_feats, EDGE_FRAMES),
    ):
        if features.ndim != 2 or features.shape[0] < min_len:
            raise ValueError(
                f"Invalid {name} feature shape for edge-{EDGE_FRAMES}: {features.shape}"
            )

    before_tail = before_feats[-EDGE_FRAMES:]
    candidate_head = candidate_feats[:EDGE_FRAMES]
    candidate_tail = candidate_feats[-EDGE_FRAMES:]
    after_head = after_feats[:EDGE_FRAMES]

    before_tail_avg = before_tail.mean(axis=0)
    candidate_head_avg = candidate_head.mean(axis=0)
    candidate_tail_avg = candidate_tail.mean(axis=0)
    after_head_avg = after_head.mean(axis=0)

    left_cosine = cosine(before_tail_avg, candidate_head_avg)
    right_cosine = cosine(candidate_tail_avg, after_head_avg)
    raw_score = (left_cosine + right_cosine) / 2.0
    return {
        "left_boundary_cosine": _jsonable_float(left_cosine),
        "right_boundary_cosine": _jsonable_float(right_cosine),
        "raw_score": _jsonable_float(raw_score),
    }


def linear_probability(scores: list[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid score vector: {scores}")

    weights = values.copy()
    if float(np.min(weights)) < 0:
        weights = weights - float(np.min(weights))
    denom = weights.sum()
    if denom <= 0 or not math.isfinite(float(denom)):
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return weights / denom


def evaluate_probability_question(
    video_id: str,
    dim: str,
    expected_letter: str,
    candidate_scores: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    letters = list(OPTION_LETTERS)
    if sorted(candidate_scores) != letters:
        raise ValueError(f"Expected candidate letters {letters}, got {sorted(candidate_scores)}")
    if expected_letter not in letters:
        raise ValueError(f"Unknown ground-truth letter: {expected_letter}")

    scores = [candidate_scores[letter].get("raw_score") for letter in letters]
    if any(score is None or not math.isfinite(float(score)) for score in scores):
        raise ValueError(f"Invalid candidate scores: {scores}")

    score_values = [float(score) for score in scores]
    probabilities = linear_probability(score_values)
    ground_truth_index = letters.index(expected_letter)
    predicted_index = int(np.argmax(probabilities))
    score_gt = score_values[ground_truth_index]
    max_score_wrong = max(score for idx, score in enumerate(score_values) if idx != ground_truth_index)
    p_gt = float(probabilities[ground_truth_index])

    return {
        "video_id": video_id,
        "dimension": dim,
        "scores": [_jsonable_float(score) for score in score_values],
        "probabilities": [_jsonable_float(float(prob)) for prob in probabilities],
        "ground_truth_index": ground_truth_index,
        "p_gt": _jsonable_float(p_gt),
        "predicted_index": predicted_index,
        "correct": predicted_index == ground_truth_index,
        "gt_margin": _jsonable_float(score_gt - max_score_wrong),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def is_question_record(value: Any) -> bool:
    return isinstance(value, dict) and all(field in value for field in QUESTION_FIELDS)


def compact_question_record(value: Any) -> dict[str, Any] | None:
    if is_question_record(value):
        return {field: value[field] for field in QUESTION_FIELDS}
    if isinstance(value, dict) and is_question_record(value.get(PRIMARY_VARIANT)):
        question = value[PRIMARY_VARIANT]
        return {field: question[field] for field in QUESTION_FIELDS}
    return None


def compact_results(results: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(results, dict):
        return {}
    compacted: dict[str, dict[str, Any]] = {}
    for stem, stem_results in results.items():
        if not isinstance(stem_results, dict):
            continue
        compact_stem: dict[str, Any] = {}
        for dim in VALID_DIMS:
            question = compact_question_record(stem_results.get(dim))
            if question is not None:
                compact_stem[dim] = question
        if compact_stem:
            compacted[str(stem)] = compact_stem
    return compacted


def load_existing(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        return {}, {}, []
    data = json.loads(path.read_text(encoding="utf-8"))
    results = compact_results(data.get("results", {}))
    skipped = data.get("skipped", [])
    if not isinstance(skipped, list):
        skipped = []
    return data, results, skipped


def summarize(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, list[float]]] = {
        DISPLAY_DIMS[dim]: {
            "p_gt": [],
            "gt_margin": [],
        }
        for dim in VALID_DIMS
    }
    buckets["Overall"] = {
        "p_gt": [],
        "gt_margin": [],
    }

    for stem_results in results.values():
        if not isinstance(stem_results, dict):
            continue
        for dim in VALID_DIMS:
            question = compact_question_record(stem_results.get(dim))
            if question is None:
                continue
            p_gt = question.get("p_gt")
            gt_margin = question.get("gt_margin")
            if p_gt is None or gt_margin is None:
                continue
            p_gt_float = float(p_gt)
            gt_margin_float = float(gt_margin)
            for bucket_name in (DISPLAY_DIMS[dim], "Overall"):
                buckets[bucket_name]["p_gt"].append(p_gt_float)
                buckets[bucket_name]["gt_margin"].append(gt_margin_float)

    summary: dict[str, Any] = {}
    for name in ("Semantic", "Alignment", "Progression", "Overall"):
        bucket = buckets[name]
        summary[name] = {
            "mean_gt_probability": (
                float(np.mean(bucket["p_gt"])) if bucket["p_gt"] else None
            ),
            "mean_gt_margin": (
                float(np.mean(bucket["gt_margin"])) if bucket["gt_margin"] else None
            ),
        }
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = []
    lines.append(f"\n{PRIMARY_VARIANT.upper()}")
    for name in ("Semantic", "Alignment", "Progression", "Overall"):
        metrics = summary.get(name, {})
        mean_p = metrics.get("mean_gt_probability")
        margin = metrics.get("mean_gt_margin")
        mean_p_text = "N/A" if mean_p is None else f"{mean_p:.4f}"
        margin_text = "N/A" if margin is None else f"{margin:.4f}"
        lines.append(
            f"  {name}: Mean GT Prob={mean_p_text}; "
            f"Mean GT Margin={margin_text}"
        )
    return "\n".join(lines)


def process_stem(
    stem: str,
    choices_dir: Path,
    encoder: DinoEncoder,
    seed: int,
    num_frames: int,
    edge_frames: int,
    strict: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = choices_dir / stem
    feature_cache: dict[str, np.ndarray] = {}
    skipped: list[dict[str, Any]] = []

    stem_result: dict[str, Any] = {}
    dim_setup: dict[str, tuple[list[Candidate], str]] = {}
    for dim in VALID_DIMS:
        candidates, expected = build_candidates(stem, dim, seed)
        needed = ["before.mp4", "after.mp4"] + [c.rel_path for c in candidates]
        missing = [rel for rel in needed if not (base / rel).exists()]
        if missing:
            item = {"stem": stem, "dim": dim, "reason": "missing_files", "missing": missing}
            if strict:
                raise FileNotFoundError(json.dumps(item, ensure_ascii=False))
            skipped.append(item)
            continue
        dim_setup[dim] = (candidates, expected)

    rels_to_encode = sorted(
        {
            rel
            for candidates, _ in dim_setup.values()
            for rel in (["before.mp4", "after.mp4"] + [c.rel_path for c in candidates])
        }
    )
    if not rels_to_encode:
        return stem_result, skipped

    rel_slices: dict[str, tuple[int, int]] = {}
    all_frames: list[np.ndarray] = []
    bad_rels: dict[str, str] = {}
    for rel in rels_to_encode:
        video_path = base / rel
        if rel == "before.mp4":
            frames = sample_edge_frames(video_path, "before", num_frames=num_frames, edge_frames=edge_frames)
        elif rel == "after.mp4":
            frames = sample_edge_frames(video_path, "after", num_frames=num_frames, edge_frames=edge_frames)
        else:
            frames = sample_edge_frames(video_path, "candidate", num_frames=num_frames, edge_frames=edge_frames)
        if not frames:
            bad_rels[rel] = f"No sampled frames from {video_path}"
            continue
        start = len(all_frames)
        all_frames.extend(frames)
        rel_slices[rel] = (start, len(all_frames))

    if bad_rels:
        for dim, (candidates, _) in list(dim_setup.items()):
            needed = ["before.mp4", "after.mp4"] + [c.rel_path for c in candidates]
            bad = {rel: bad_rels[rel] for rel in needed if rel in bad_rels}
            if bad:
                item = {"stem": stem, "dim": dim, "reason": "empty_sampled_frames", "details": bad}
                if strict:
                    raise ValueError(json.dumps(item, ensure_ascii=False))
                skipped.append(item)
                del dim_setup[dim]

    if not dim_setup:
        return stem_result, skipped

    encoded = encoder.encode_frames(all_frames)
    for rel, (start, end) in rel_slices.items():
        feature_cache[rel] = encoded[start:end]

    def get_features(rel_path: str) -> np.ndarray:
        return feature_cache[rel_path]

    for dim, (candidates, expected) in dim_setup.items():
        try:
            before_feats = get_features("before.mp4")
            after_feats = get_features("after.mp4")
            candidate_scores: dict[str, dict[str, float | None]] = {}
            for candidate in candidates:
                candidate_feats = get_features(candidate.rel_path)
                candidate_scores[candidate.letter] = score_candidate(
                    before_feats,
                    candidate_feats,
                    after_feats,
                )
            stem_result[dim] = evaluate_probability_question(
                video_id=stem,
                dim=dim,
                expected_letter=expected,
                candidate_scores=candidate_scores,
            )
        except Exception as exc:
            item = {"stem": stem, "dim": dim, "reason": type(exc).__name__, "message": str(exc)}
            if strict:
                raise
            skipped.append(item)

    return stem_result, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv2 edge-matching baseline for TempCloze")
    parser.add_argument("--choices-dir", type=Path, default=DEFAULT_CHOICES_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default="facebook/dinov2-base")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--targets-file", type=Path, default=None, help="newline-delimited stem list")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="fail instead of skipping incomplete tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = (
            DEFAULT_RESULTS_DIR
            / f"{PRIMARY_VARIANT}_{_safe_name(args.model)}.json"
        )

    if args.overwrite and output.exists():
        output.unlink()

    _, results, skipped = load_existing(output)
    completed_before = sum(len(v) for v in results.values() if isinstance(v, dict))

    stems = sorted(
        p.name for p in args.choices_dir.iterdir()
        if p.is_dir() and (p / "GT.mp4").exists()
    )
    targets: list[str] = []
    if args.targets:
        targets.extend(args.targets)
    if args.targets_file:
        targets.extend(
            line.strip()
            for line in args.targets_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if targets:
        wanted = set(targets)
        stems = [stem for stem in stems if stem in wanted]
    if args.limit is not None:
        stems = stems[: args.limit]

    if not stems:
        raise SystemExit(f"No stems found in {args.choices_dir}")

    encoder = DinoEncoder(args.model, args.device, args.batch_size)

    processed = 0
    for stem in tqdm(stems, desc="Edge matching"):
        existing = results.get(stem)
        if (
            isinstance(existing, dict)
            and all(is_question_record(existing.get(dim)) for dim in VALID_DIMS)
        ):
            continue

        stem_result, stem_skipped = process_stem(
            stem=stem,
            choices_dir=args.choices_dir,
            encoder=encoder,
            seed=args.seed,
            num_frames=args.num_frames,
            edge_frames=EDGE_FRAMES,
            strict=args.strict,
        )
        if stem_result:
            results[stem] = stem_result
        skipped.extend(stem_skipped)
        processed += 1

        if processed % max(1, args.save_every) == 0:
            payload = {
                "summary": summarize(results),
                "results": results,
                "skipped": skipped,
            }
            atomic_write_json(output, payload)

    summary = summarize(results)
    payload = {"summary": summary, "results": results, "skipped": skipped}
    atomic_write_json(output, payload)

    completed_after = sum(len(v) for v in results.values() if isinstance(v, dict))
    print(f"Saved: {output}")
    print(f"Completed entries: {completed_before} -> {completed_after}")
    print(f"Skipped tasks: {len(skipped)}")
    print(format_summary(summary))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
