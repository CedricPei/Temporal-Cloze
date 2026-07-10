"""Streamlit app for collecting human Temporal Cloze dataset quality ratings.

Run:
  streamlit run human_quality_eval_app.py

Expected package layout:
  human_quality_eval_app.py
  choices/
    set_1/
    set_2/
    ...

Optional environment variables:
  HUMAN_QUALITY_EVAL_CHOICES_DIR=./choices_human
  HUMAN_EVAL_CHOICES_DIR=./choices_human
  HUMAN_QUALITY_EVAL_RESULTS_DIR=./human_quality_eval_results
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

import cv2
import streamlit as st


ROOT = Path(__file__).resolve().parent


def resolve_config_path(value: str, default: Path) -> Path:
    path = Path(value).expanduser() if value.strip() else default
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


DEFAULT_CHOICES_ROOT = ROOT / "choices" if (ROOT / "choices").exists() else ROOT / "choices_human"
CHOICES_ROOT = resolve_config_path(
    os.getenv(
        "HUMAN_QUALITY_EVAL_CHOICES_DIR",
        os.getenv("HUMAN_EVAL_CHOICES_DIR", ""),
    ),
    DEFAULT_CHOICES_ROOT,
)
RESULTS_DIR = resolve_config_path(
    os.getenv("HUMAN_QUALITY_EVAL_RESULTS_DIR", ""),
    ROOT / "human_quality_eval_results",
)
DEFAULT_DISPLAY_MODE = os.getenv("HUMAN_QUALITY_EVAL_DISPLAY_MODE", "video").strip().lower()
NUM_FRAMES = int(os.getenv("HUMAN_QUALITY_EVAL_NUM_FRAMES", "16"))
MAX_HEIGHT = int(os.getenv("HUMAN_QUALITY_EVAL_MAX_HEIGHT", "360"))
SCHEDULE_SEED = os.getenv("HUMAN_QUALITY_EVAL_SCHEDULE_SEED", "temporal-cloze-quality-eval-v1")
SCHEDULE_POLICY = "single_folder_quality_rating_v1"
RUNTIME_CHOICES_DIR = "_runtime_choices_dir"
RUNTIME_RESULT_PATH = "_runtime_result_path"
DISPLAY_MODES = ("frames", "video")

REQUIRED_FILES = [
    "before.mp4",
    "GT.mp4",
    "after.mp4",
]

PROMPT = """You are given the full Temporal Cloze sample in order:
- BEGINNING: the first part of a video.
- ANSWER: the candidate middle segment to judge.
- ENDING: the last part of the same video.

The downstream task asks a model to identify the true middle segment between BEGINNING and ENDING. Your rating should reflect whether this sample is a reliable and meaningful Temporal Cloze question.

Please consider these quality aspects:
- Scene transition: whether BEGINNING, ANSWER, and ENDING appear to belong to a coherent visual scene or a justifiable scene transition.
- Subject consistency: whether the main person/object/environment remains consistent across the three segments.
- Action consistency: whether the motion or activity in ANSWER reasonably continues from BEGINNING and leads toward ENDING.
- Logical consistency: whether the full sequence makes temporal and causal sense as a middle-completion example.

Rate the overall data quality from 1 to 5:
1 = Unusable: major inconsistencies make ANSWER clearly unreliable as the middle segment.
2 = Poor: several quality issues make the middle relation hard to trust.
3 = Borderline: partially coherent, but at least one important aspect is ambiguous or weak.
4 = Good: mostly coherent and suitable for evaluation, with only minor uncertainty.
5 = Excellent: clearly coherent across the listed aspects and strongly suitable as a Temporal Cloze sample.
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_id(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._-") or "participant"


def stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def folder_seed(choice_dir: Path) -> int:
    return stable_seed(f"{SCHEDULE_SEED}:{choice_dir.name}")


@st.cache_data(show_spinner=False)
def discover_stems(choices_dir: str) -> list[str]:
    root = Path(choices_dir)
    if not root.exists():
        return []

    stems: list[str] = []
    for item in root.iterdir():
        if item.is_dir() and all((item / rel).exists() for rel in REQUIRED_FILES):
            stems.append(item.name)
    return sorted(stems)


@st.cache_data(show_spinner=False)
def discover_choice_sets(choices_root: str) -> list[str]:
    root = Path(choices_root)
    if not root.exists():
        return []

    direct_stems = discover_stems(str(root))
    if direct_stems:
        return [str(root)]

    sets: list[str] = []
    for item in sorted(root.iterdir()):
        if item.is_dir() and discover_stems(str(item)):
            sets.append(str(item))
    return sets


def build_schedule(stems: list[str], seed: int) -> list[dict]:
    tasks = [{"stem": stem} for stem in stems]
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks


def result_path_for(choice_dir: Path) -> Path:
    return RESULTS_DIR / f"{dataset_id_for(choice_dir)}.json"


def dataset_id_for(choice_dir: Path) -> str:
    if choice_dir.name in {"choices", "choices_human"}:
        return safe_id(choice_dir.parent.name)
    return safe_id(choice_dir.name)


def write_result(data: dict) -> None:
    result_path = Path(data[RUNTIME_RESULT_PATH])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    persisted = {key: value for key, value in data.items() if not key.startswith("_runtime_")}
    result_path.write_text(json.dumps(persisted, indent=2, ensure_ascii=False), encoding="utf-8")


def attach_runtime_paths(data: dict, choice_dir: Path, result_path: Path) -> None:
    # Legacy result files stored machine-specific paths. Keep paths only in
    # memory so newly written results remain portable across machines.
    data.pop("choices_dir", None)
    data.pop("result_path", None)
    data[RUNTIME_CHOICES_DIR] = str(choice_dir)
    data[RUNTIME_RESULT_PATH] = str(result_path)


def normalize_responses(data: dict) -> None:
    responses = data.get("responses")
    if isinstance(responses, list):
        data["responses"] = {
            str(entry.get("task_index")): entry
            for entry in responses
            if isinstance(entry, dict) and entry.get("task_index") is not None
        }
    elif not isinstance(responses, dict):
        data["responses"] = {}


def run_matches_config(data: dict, choice_dir: Path) -> bool:
    return (
        data.get("schedule_policy") == SCHEDULE_POLICY
        and data.get("dataset_id") == dataset_id_for(choice_dir)
        and data.get("seed") == folder_seed(choice_dir)
    )


def backup_incompatible_result(path: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.stem}.legacy-{stamp}{path.suffix}")
    path.replace(backup_path)


def load_or_create_run(choice_dir: Path) -> dict:
    choice_dir = choice_dir.expanduser().resolve()
    result_path = result_path_for(choice_dir)
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        normalize_responses(existing)
        if run_matches_config(existing, choice_dir):
            attach_runtime_paths(existing, choice_dir, result_path)
            write_result(existing)
            return existing
        backup_incompatible_result(result_path)

    stems = discover_stems(str(choice_dir))
    seed = folder_seed(choice_dir)
    data = {
        "dataset_id": dataset_id_for(choice_dir),
        "schedule_policy": SCHEDULE_POLICY,
        "seed": seed,
        "started_at_utc": now_iso(),
        "completed_at_utc": None,
        "stems": stems,
        "schedule": build_schedule(stems, seed),
        "responses": {},
    }
    attach_runtime_paths(data, choice_dir, result_path)
    write_result(data)
    return data


def is_answered(data: dict, idx: int) -> bool:
    return str(idx) in data.get("responses", {})


def get_response(data: dict, idx: int) -> dict | None:
    value = data.get("responses", {}).get(str(idx))
    return value if isinstance(value, dict) else None


def first_unanswered_index(data: dict) -> int:
    for idx in range(len(data["schedule"])):
        if not is_answered(data, idx):
            return idx
    return 0


def next_index_after(data: dict, idx: int) -> int:
    total = len(data["schedule"])
    for offset in range(1, total + 1):
        candidate = (idx + offset) % total
        if not is_answered(data, candidate):
            return candidate
    return min(idx + 1, total - 1)


def set_run(data: dict) -> None:
    st.session_state.run_data = data
    st.session_state.current_index = first_unanswered_index(data)
    st.session_state.task_started_at = time.time()


def init_ui_state() -> None:
    if st.session_state.get("display_mode") not in DISPLAY_MODES:
        st.session_state.display_mode = DEFAULT_DISPLAY_MODE if DEFAULT_DISPLAY_MODE in DISPLAY_MODES else "frames"
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0


def current_task_index() -> int | None:
    data = st.session_state.get("run_data")
    if not data:
        return None
    idx = int(st.session_state.get("current_index", 0))
    return idx if 0 <= idx < len(data["schedule"]) else None


def go_to_task(idx: int) -> None:
    data = st.session_state.run_data
    if not 0 <= idx < len(data["schedule"]):
        return
    st.session_state.current_index = idx
    st.session_state.task_started_at = time.time()


def rate_current(rating: int) -> None:
    data = st.session_state.run_data
    idx = current_task_index()
    if idx is None:
        return

    task = data["schedule"][idx]
    previous = get_response(data, idx)
    data["responses"][str(idx)] = {
        "task_index": idx,
        "stem": task["stem"],
        "rating": rating,
        "display_mode": st.session_state.get("display_mode", DEFAULT_DISPLAY_MODE),
        "elapsed_sec": round(time.time() - st.session_state.task_started_at, 3),
        "answered_at_utc": now_iso(),
        "redo_count": int(previous.get("redo_count", 0)) + 1 if previous else 0,
    }

    if len(data["responses"]) >= len(data["schedule"]):
        data["completed_at_utc"] = now_iso()
    else:
        data["completed_at_utc"] = None

    write_result(data)
    st.session_state.run_data = data
    st.session_state.current_index = next_index_after(data, idx)
    st.session_state.task_started_at = time.time()
    st.rerun()


@st.cache_data(show_spinner=False, max_entries=128)
def sample_frames(video_path: str, num_frames: int, max_height: int) -> list:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    n = min(num_frames, total)
    indices = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if h > max_height:
            scale = max_height / h
            frame = cv2.resize(frame, (int(w * scale), max_height))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def render_clip(path: Path, display_mode: str, frames_per_row: int = 4) -> None:
    if not path.exists():
        st.error(f"Missing clip: {path.name}")
        return

    if display_mode == "video":
        st.video(str(path))
        return

    frames = sample_frames(str(path), NUM_FRAMES, MAX_HEIGHT)
    if not frames:
        st.error(f"Cannot read clip: {path.name}")
        return

    for start in range(0, len(frames), frames_per_row):
        cols = st.columns(frames_per_row)
        for offset, frame in enumerate(frames[start : start + frames_per_row]):
            frame_idx = start + offset + 1
            cols[offset].image(frame, caption=f"{frame_idx}", use_container_width=True)


def frame_data_uri(frame) -> str | None:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    if not ok:
        return None
    b64 = base64.b64encode(encoded).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def render_frame_clip_columns(base: Path, clip_specs: tuple[tuple[str, str], ...]) -> None:
    sampled_clips: list[tuple[str, str, Path, list]] = []
    frame_aspect_ratio = "16 / 9"
    for title, filename in clip_specs:
        path = base / filename
        frames = sample_frames(str(path), NUM_FRAMES, MAX_HEIGHT) if path.exists() else []
        if frames and frame_aspect_ratio == "16 / 9":
            h, w = frames[0].shape[:2]
            if h > 0 and w > 0:
                frame_aspect_ratio = f"{w} / {h}"
        sampled_clips.append((title, filename, path, frames))

    sections: list[str] = []
    for title, _filename, path, frames in sampled_clips:
        if not path.exists():
            body = f'<div class="clip-frame-error">Missing clip: {path.name}</div>'
        elif not frames:
            body = f'<div class="clip-frame-error">Cannot read clip: {path.name}</div>'
        else:
            items = []
            for frame_idx, frame in enumerate(frames, start=1):
                uri = frame_data_uri(frame)
                if uri is None:
                    continue
                items.append(
                    '<figure class="clip-frame-item">'
                    '<div class="clip-frame-box">'
                    f'<img src="{uri}" alt="{title} frame {frame_idx}">'
                    "</div>"
                    f"<figcaption>{frame_idx}</figcaption>"
                    "</figure>"
                )
            body = f'<div class="clip-frame-grid">{"".join(items)}</div>'

        sections.append(
            '<section class="clip-frame-column">'
            f"<h3>{title}</h3>"
            f"{body}"
            "</section>"
        )

    html_parts: list[str] = []
    for idx, section in enumerate(sections):
        if idx > 0:
            html_parts.append('<div class="clip-frame-separator" aria-hidden="true"></div>')
        html_parts.append(section)

    st.markdown(
        (
            f'<div class="clip-frame-layout" style="--clip-frame-aspect-ratio: {frame_aspect_ratio};">'
            f'{"".join(html_parts)}</div>'
        ),
        unsafe_allow_html=True,
    )


def render_task(data: dict) -> None:
    idx = current_task_index()
    total = len(data["schedule"])

    if total == 0 or idx is None:
        st.error("No valid tasks were found in the selected choices folder.")
        return

    task = data["schedule"][idx]
    base = Path(data[RUNTIME_CHOICES_DIR]) / task["stem"]
    display_mode = st.session_state.get("display_mode", DEFAULT_DISPLAY_MODE)
    existing = get_response(data, idx)

    st.progress(len(data["responses"]) / total, text=f"Rated {len(data['responses'])} of {total}")
    st.markdown(f"### Sample {idx + 1} of {total}")
    st.markdown(PROMPT)

    clip_specs = (
        ("BEGINNING", "before.mp4"),
        ("ANSWER", "GT.mp4"),
        ("ENDING", "after.mp4"),
    )
    if display_mode == "frames":
        render_frame_clip_columns(base, clip_specs)
    else:
        content_cols = st.columns(3)
        for col, (title, filename) in zip(content_cols, clip_specs):
            with col:
                st.subheader(title)
                render_clip(base / filename, display_mode, frames_per_row=2)

    st.subheader("Quality rating")
    if existing:
        st.info(f"Current saved rating: {existing['rating']}")

    cols = st.columns(5)
    for rating, col in enumerate(cols, start=1):
        is_selected = existing and existing.get("rating") == rating
        col.button(
            str(rating),
            key=f"rate-{idx}-{rating}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
            on_click=rate_current,
            args=(rating,),
        )


def render_start() -> None:
    choice_sets = [Path(p) for p in discover_choice_sets(str(CHOICES_ROOT))]

    st.title("Temporal Cloze Data Quality Evaluation")
    st.caption(f"Choices root: {CHOICES_ROOT}")

    if not CHOICES_ROOT.exists():
        st.error("The choices root directory does not exist.")
        return
    if not choice_sets:
        st.error("No valid choice folders were found.")
        return

    labels = [p.name for p in choice_sets]
    selected_label = st.selectbox("Evaluation folder", labels)
    choice_dir = choice_sets[labels.index(selected_label)]
    stems = discover_stems(str(choice_dir))

    st.write(f"Valid samples: {len(stems)}")
    st.write("Each sample shows BEGINNING, ANSWER, and ENDING in order.")

    if st.button("Start / Resume", use_container_width=True):
        data = load_or_create_run(choice_dir)
        set_run(data)
        st.rerun()


def render_question_nav(data: dict) -> None:
    st.markdown("Sample List")
    total = len(data["schedule"])
    current = current_task_index()
    for start in range(0, total, 5):
        cols = st.columns(5)
        for offset, col in enumerate(cols):
            idx = start + offset
            if idx >= total:
                continue
            answered = is_answered(data, idx)
            label = str(idx + 1)
            if idx == current:
                col.markdown(
                    f'<div class="current-task-cell">{label}</div>',
                    unsafe_allow_html=True,
                )
                continue
            if answered:
                label = f"✓ {label}"
            col.button(
                label,
                key=f"nav-{idx}",
                type="primary" if answered else "secondary",
                use_container_width=True,
                on_click=go_to_task,
                args=(idx,),
            )


def main() -> None:
    st.set_page_config(page_title="Temporal Cloze Quality Eval", layout="wide")
    init_ui_state()
    st.markdown(
        """
        <style>
        div.stButton > button[kind="primary"] {
            background-color: #198754;
            border-color: #198754;
            color: white;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #157347;
            border-color: #157347;
            color: white;
        }
        .current-task-cell {
            align-items: center;
            background-color: #3f444a;
            border: 1px solid #3f444a;
            border-radius: 0.5rem;
            color: white;
            display: flex;
            font-size: 0.9rem;
            font-weight: 600;
            justify-content: center;
            min-height: 2.45rem;
            width: 100%;
        }
        .clip-frame-layout {
            align-items: stretch;
            display: grid;
            gap: 1rem;
            grid-template-columns: minmax(0, 1fr) 2px minmax(0, 1fr) 2px minmax(0, 1fr);
            margin-top: 0.75rem;
        }
        .clip-frame-column {
            min-width: 0;
        }
        .clip-frame-separator {
            border-left: 2px dashed #6b7280;
            height: 100%;
            min-height: 100%;
            width: 2px;
        }
        .clip-frame-column h3 {
            font-size: 1.25rem;
            font-weight: 600;
            line-height: 1.2;
            margin: 0 0 0.75rem;
        }
        .clip-frame-grid {
            display: grid;
            gap: 0.5rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .clip-frame-item {
            margin: 0;
            min-width: 0;
        }
        .clip-frame-box {
            align-items: center;
            aspect-ratio: var(--clip-frame-aspect-ratio);
            background: #111827;
            box-sizing: border-box;
            display: flex;
            justify-content: center;
            overflow: hidden;
            width: 100%;
        }
        .clip-frame-item img {
            display: block;
            height: 100%;
            object-fit: contain;
            width: 100%;
        }
        .clip-frame-item figcaption {
            color: #6b7280;
            font-size: 0.75rem;
            line-height: 1.4;
            text-align: center;
        }
        .clip-frame-error {
            color: #b42318;
            font-size: 0.95rem;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "run_data" not in st.session_state:
        render_start()
        return

    with st.sidebar:
        data = st.session_state.run_data
        done = len(data["responses"])
        total = len(data["schedule"])
        st.write(f"Folder: `{data['dataset_id']}`")
        st.write(f"Progress: {done}/{total}")
        st.radio(
            "Display",
            DISPLAY_MODES,
            format_func=lambda value: "16 frames" if value == "frames" else "Video",
            key="display_mode",
        )
        st.caption("Rated samples are green. Click any number to review or redo.")
        st.divider()
        render_question_nav(data)

    render_task(st.session_state.run_data)


if __name__ == "__main__":
    main()
