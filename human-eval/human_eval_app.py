"""Streamlit app for collecting human Temporal Cloze baselines.

Run:
  streamlit run human_eval_app.py

Expected package layout:
  human_eval_app.py
  choices_human/
    set_1/
    set_2/
    ...

Optional environment variables:
  HUMAN_EVAL_CHOICES_DIR=./choices_human
  HUMAN_EVAL_RESULTS_DIR=./human_eval_results
"""

from __future__ import annotations

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


def resolve_config_path(env_key: str, default: Path) -> Path:
    value = os.getenv(env_key, "").strip()
    path = Path(value).expanduser() if value else default
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


CHOICES_ROOT = resolve_config_path("HUMAN_EVAL_CHOICES_DIR", ROOT / "choices_human")
RESULTS_DIR = resolve_config_path("HUMAN_EVAL_RESULTS_DIR", ROOT / "human_eval_results")
DEFAULT_DISPLAY_MODE = os.getenv("HUMAN_EVAL_DISPLAY_MODE", "frames").strip().lower()
NUM_FRAMES = int(os.getenv("HUMAN_EVAL_NUM_FRAMES", "16"))
MAX_HEIGHT = int(os.getenv("HUMAN_EVAL_MAX_HEIGHT", "360"))
SCHEDULE_SEED = os.getenv("HUMAN_EVAL_SCHEDULE_SEED", "temporal-cloze-human-eval-folder-v1")
SCHEDULE_POLICY = "single_folder_navigation_v1"
RUNTIME_CHOICES_DIR = "_runtime_choices_dir"
RUNTIME_RESULT_PATH = "_runtime_result_path"
DISPLAY_MODES = ("frames", "video")
ANSWER_FLOWS = ("immediate", "confirm")

DIMENSIONS = {
    "S": ["S/Rand1.mp4", "S/Rand2.mp4", "S/Rand3.mp4"],
    "A": ["A/Early.mp4", "A/Late.mp4", "A/Wide.mp4"],
    "C": ["C/Reverse.mp4", "C/Shuffle.mp4", "C/Loop.mp4"],
}

REQUIRED_FILES = [
    "before.mp4",
    "after.mp4",
    "GT.mp4",
    *[rel for distractors in DIMENSIONS.values() for rel in distractors],
]

PROMPT = """You are given:
- BEGINNING: the first part of a video.
- END: the last part of the same video.

The middle segment between BEGINNING and END was removed. You will then see four candidate middle segments, labeled A, B, C, D. Exactly one is the true middle that connects BEGINNING to END.

Task: Which candidate is the correct middle? Choose one of A, B, C, D."""


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
        if not item.is_dir():
            continue
        if all((item / rel).exists() for rel in REQUIRED_FILES):
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
    tasks = [{"stem": stem, "dim": dim} for stem in stems for dim in DIMENSIONS]
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks


def build_options(stem: str, dim: str, seed: int) -> tuple[list[dict], str, dict[str, str]]:
    options = [{"rel_path": "GT.mp4", "is_gt": True}]
    options.extend({"rel_path": rel, "is_gt": False} for rel in DIMENSIONS[dim])

    rng = random.Random(f"{seed}:{stem}:{dim}:options")
    rng.shuffle(options)

    letters = [chr(65 + i) for i in range(len(options))]
    expected = next(letters[i] for i, opt in enumerate(options) if opt["is_gt"])
    option_map = {letters[i]: Path(opt["rel_path"]).stem for i, opt in enumerate(options)}
    rendered = [
        {"letter": letters[i], "rel_path": opt["rel_path"], "clip_name": option_map[letters[i]]}
        for i, opt in enumerate(options)
    ]
    return rendered, expected, option_map


def result_path_for(choice_dir: Path) -> Path:
    return RESULTS_DIR / f"{safe_id(choice_dir.name)}.json"


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
        and data.get("dataset_id") == choice_dir.name
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
        "dataset_id": choice_dir.name,
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
    st.session_state.pending_answer = None
    st.session_state.pending_task_index = None


def init_ui_state() -> None:
    if st.session_state.get("display_mode") not in DISPLAY_MODES:
        st.session_state.display_mode = DEFAULT_DISPLAY_MODE if DEFAULT_DISPLAY_MODE in DISPLAY_MODES else "frames"
    if st.session_state.get("answer_flow") not in ANSWER_FLOWS:
        st.session_state.answer_flow = "confirm"
    if "pending_answer" not in st.session_state:
        st.session_state.pending_answer = None
    if "pending_task_index" not in st.session_state:
        st.session_state.pending_task_index = None
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
    existing = get_response(data, idx)
    st.session_state.pending_answer = existing.get("answer") if existing else None
    st.session_state.pending_task_index = idx if existing else None
    st.session_state.task_started_at = time.time()


def select_current(letter: str) -> None:
    idx = current_task_index()
    if idx is None:
        return
    st.session_state.pending_answer = letter
    st.session_state.pending_task_index = idx


def answer_current(letter: str) -> None:
    data = st.session_state.run_data
    idx = current_task_index()
    if idx is None:
        return

    task = data["schedule"][idx]
    _, expected, option_map = build_options(task["stem"], task["dim"], data["seed"])
    previous = get_response(data, idx)

    data["responses"][str(idx)] = {
        "task_index": idx,
        "stem": task["stem"],
        "dim": task["dim"],
        "answer": letter,
        "expected": expected,
        "correct": letter == expected,
        "option_map": option_map,
        "display_mode": st.session_state.get("display_mode", DEFAULT_DISPLAY_MODE),
        "answer_flow": st.session_state.get("answer_flow", "confirm"),
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
    st.session_state.pending_answer = None
    st.session_state.pending_task_index = None
    st.session_state.current_index = next_index_after(data, idx)
    st.session_state.task_started_at = time.time()
    st.rerun()


def submit_pending_answer() -> None:
    idx = current_task_index()
    if idx is None:
        return
    if st.session_state.get("pending_task_index") != idx:
        return
    letter = st.session_state.get("pending_answer")
    if letter:
        answer_current(letter)


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


def render_clip(path: Path, display_mode: str) -> None:
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

    for start in range(0, len(frames), 4):
        cols = st.columns(4)
        for offset, frame in enumerate(frames[start : start + 4]):
            frame_idx = start + offset + 1
            cols[offset].image(frame, caption=f"{frame_idx}", use_container_width=True)


def render_task(data: dict) -> None:
    idx = current_task_index()
    total = len(data["schedule"])

    if total == 0 or idx is None:
        st.error("No valid tasks were found in the selected choices folder.")
        return

    task = data["schedule"][idx]
    base = Path(data[RUNTIME_CHOICES_DIR]) / task["stem"]
    options, _, _ = build_options(task["stem"], task["dim"], data["seed"])
    display_mode = st.session_state.get("display_mode", DEFAULT_DISPLAY_MODE)
    answer_flow = st.session_state.get("answer_flow", "confirm")
    existing = get_response(data, idx)

    st.progress(len(data["responses"]) / total, text=f"Answered {len(data['responses'])} of {total}")
    st.markdown(f"### Question {idx + 1} of {total}")
    st.markdown(PROMPT)

    left, right = st.columns(2)
    with left:
        st.subheader("BEGINNING")
        render_clip(base / "before.mp4", display_mode)
    with right:
        st.subheader("END")
        render_clip(base / "after.mp4", display_mode)

    st.divider()
    st.subheader("Candidates")
    if display_mode == "video":
        candidate_slots = [col for _ in range(2) for col in st.columns(2)]
    else:
        candidate_slots = [st.container() for _ in options]

    selected = (
        st.session_state.get("pending_answer")
        if st.session_state.get("pending_task_index") == idx
        else existing.get("answer") if existing else None
    )
    for slot, option in zip(candidate_slots, options):
        with slot:
            st.markdown(f"#### Candidate {option['letter']}")
            render_clip(base / option["rel_path"], display_mode)
            is_selected = selected == option["letter"]
            st.button(
                f"{'Selected' if is_selected else 'Choose'} {option['letter']}",
                key=f"choose-{idx}-{option['letter']}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
                on_click=select_current if answer_flow == "confirm" else answer_current,
                args=(option["letter"],),
            )
            if display_mode != "video":
                st.divider()


def render_start() -> None:
    choice_sets = [Path(p) for p in discover_choice_sets(str(CHOICES_ROOT))]

    st.title("Temporal Cloze Human Evaluation")
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

    st.write(f"Valid stems: {len(stems)}")
    st.write(f"Sub-tasks: {len(stems) * len(DIMENSIONS)}")

    if st.button("Start / Resume", use_container_width=True):
        data = load_or_create_run(choice_dir)
        set_run(data)
        st.rerun()


def render_question_nav(data: dict) -> None:
    st.markdown("Question List")
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
            elif answered:
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
    st.set_page_config(page_title="Temporal Cloze Human Eval", layout="wide")
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
        st.radio(
            "Submit mode",
            ANSWER_FLOWS,
            format_func=lambda value: "Choose submits" if value == "immediate" else "Select then submit",
            key="answer_flow",
        )
        idx = current_task_index()
        if st.session_state.answer_flow == "confirm" and idx is not None:
            selected = (
                st.session_state.pending_answer
                if st.session_state.pending_task_index == idx
                else get_response(data, idx).get("answer") if get_response(data, idx) else None
            )
            st.button(
                "Submit",
                type="primary",
                disabled=selected is None,
                use_container_width=True,
                on_click=submit_pending_answer,
            )
        st.caption("Answered questions are green. Click any number to review or redo.")
        st.divider()
        render_question_nav(data)

    render_task(st.session_state.run_data)


if __name__ == "__main__":
    main()
