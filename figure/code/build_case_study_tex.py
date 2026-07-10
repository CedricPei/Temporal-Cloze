#!/usr/bin/env python3
"""Build a visual LaTeX case study for selected Temporal-Cloze samples.

The layout mirrors the human-eval UI:
  - shared BEGINNING/END first+last frames at the top of each sample;
  - three subtask columns (S/A/P);
  - four candidate clips per subtask in a 2x2 grid;
  - two model rationale+answer rows below each candidate grid.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
CHOICES_DIR = REPO_ROOT / "TempCloze" / "choices"
OUT_DIR = REPO_ROOT / "figure" / "pics"
FRAME_DIR = OUT_DIR / "case_study_frames"
OUT_PATH = OUT_DIR / "case_study_qwen_seed.tex"

MODEL_PATHS = {
    "Qwen3.5-397B-A17B": REPO_ROOT / "TempCloze" / "eval_results" / "closed" / "eval_results" / "qwen3.5-397b-a17b.json",
    "Seed1.8-T": REPO_ROOT / "TempCloze" / "eval_results" / "closed" / "eval_results" / "seed-1-8-thinking.json",
}

MODEL_HEIGHTS = {
    "Qwen3.5-397B-A17B": r"\caseQwenH",
    "Seed1.8-T": r"\caseSeedH",
}

CASE_STEMS = [
    "DAY1_A2_ALICE_17130000",
    "HwAPGFLvoHg.60_0",
]

DIM_LABEL = {"S": "S", "A": "A", "C": "P"}
DIM_TITLE = {
    "S": "Semantic",
    "A": "Alignment",
    "C": "Progression",
}
OPTION_REL_PATHS = {
    "S": [
        ("True middle", "GT.mp4"),
        ("Rand1", "S/Rand1.mp4"),
        ("Rand2", "S/Rand2.mp4"),
        ("Rand3", "S/Rand3.mp4"),
    ],
    "A": [
        ("True middle", "GT.mp4"),
        ("Advanced", "A/Early.mp4"),
        ("Deferred", "A/Late.mp4"),
        ("Expanded", "A/Wide.mp4"),
    ],
    "C": [
        ("True middle", "GT.mp4"),
        ("Reverse", "C/Reverse.mp4"),
        ("Reorder", "C/Shuffle.mp4"),
        ("Repeat", "C/Loop.mp4"),
    ],
}
CLIP_DISPLAY = {
    "GT": "True middle",
    "Early": "Advanced",
    "Late": "Deferred",
    "Wide": "Expanded",
    "Shuffle": "Reorder",
    "Loop": "Repeat",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def tex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")


def display_clip_name(name: Any) -> str:
    return CLIP_DISPLAY.get(str(name), str(name))


def source_lookup() -> dict[str, str]:
    lookup = {}
    for source_dir in sorted((REPO_ROOT / "output").iterdir()):
        meta_path = source_dir / "meta.json"
        if not source_dir.is_dir() or not meta_path.exists():
            continue
        for filename in load_json(meta_path):
            lookup[Path(filename).stem] = source_dir.name
    return lookup


def all_model_paths() -> list[Path]:
    roots = [
        REPO_ROOT / "TempCloze" / "eval_results" / "closed" / "eval_results",
        REPO_ROOT / "TempCloze" / "eval_results" / "open" / "eval_results",
    ]
    return [path for root in roots for path in sorted(root.glob("*.json"))]


def sample_accuracy(stem: str) -> float:
    correct = 0
    total = 0
    for path in all_model_paths():
        data = load_json(path)
        entries = data.get(stem, {})
        if not isinstance(entries, dict):
            continue
        for dim in ("S", "A", "C"):
            entry = entries.get(dim)
            if isinstance(entry, dict) and isinstance(entry.get("correct"), bool):
                total += 1
                correct += int(entry["correct"])
    return correct / total if total else 0.0


def save_frame_pdf(frame: Any, out_path: Path, max_height: int = 720) -> None:
    h, w = frame.shape[:2]
    if h > max_height:
        scale = max_height / h
        frame = cv2.resize(frame, (int(round(w * scale)), max_height))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(out_path, "PDF", resolution=1600.0)


def frame_pair(video_path: Path, key: str, max_height: int = 720) -> tuple[str, str]:
    first = FRAME_DIR / f"{key}_first.pdf"
    last = FRAME_DIR / f"{key}_last.pdf"
    if first.exists() and last.exists():
        return first.relative_to(OUT_DIR).as_posix(), last.relative_to(OUT_DIR).as_posix()

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot read frames from {video_path}")

    for idx, out_path in [(0, first), (max(total - 1, 0), last)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Cannot read frame {idx} from {video_path}")
        save_frame_pdf(frame, out_path, max_height=max_height)

    cap.release()
    return first.relative_to(OUT_DIR).as_posix(), last.relative_to(OUT_DIR).as_posix()


def clip_pair_tex(label: str, first: str, last: str, width: str = r"0.47\linewidth") -> str:
    return (
        r"\caseClip{"
        + tex_escape(label)
        + "}{"
        + first
        + "}{"
        + last
        + "}{"
        + width
        + "}"
    )


def option_grid_tex(stem: str, dim: str) -> str:
    base = CHOICES_DIR / stem
    rendered = []
    for label, rel_path in OPTION_REL_PATHS[dim]:
        key = safe_name(f"{stem}_{dim}_{Path(rel_path).with_suffix('').as_posix()}")
        first, last = frame_pair(base / rel_path, key)
        rendered.append(clip_pair_tex(label, first, last, width=r"0.47\linewidth"))
    return (
        r"{\setlength{\tabcolsep}{0pt}\begin{tabular}{@{}c@{\hspace{0.02\linewidth}}c@{}}"
        + "\n"
        + rendered[0]
        + " & "
        + rendered[1]
        + r"\\[0.45em]"
        + "\n"
        + rendered[2]
        + " & "
        + rendered[3]
        + "\n"
        + r"\end{tabular}}"
    )


def answer_summary(entry: dict[str, Any]) -> str:
    answer = entry.get("answer")
    expected = entry.get("expected")
    option_map = entry.get("option_map", {})
    selected_clip = display_clip_name(option_map.get(answer, "?"))
    expected_clip = display_clip_name(option_map.get(expected, "?"))
    status = r"\textcolor{caseGreen}{Correct}" if entry.get("correct") else r"\textcolor{caseRed}{Wrong}"
    return (
        rf"\textbf{{Ans.}} {tex_escape(answer)} ({tex_escape(selected_clip)}); "
        rf"\textbf{{GT}} {tex_escape(expected)} ({tex_escape(expected_clip)}); {status}"
    )


def model_rows_tex(stem: str, dim: str, models: dict[str, dict[str, Any]]) -> str:
    rows = []
    for model_name, data in models.items():
        entry = data[stem][dim]
        rows.append(
            r"\caseReason{"
            + MODEL_HEIGHTS[model_name]
            + "}{"
            + tex_escape(model_name)
            + "}{"
            + answer_summary(entry)
            + "}{"
            + tex_escape(entry.get("reason", ""))
            + "}"
        )
    return "\n".join(rows)


def measure_answer_heights_tex(stem: str, models: dict[str, dict[str, Any]]) -> str:
    rows = [r"\caseQwenH=0pt", r"\caseSeedH=0pt"]
    for model_name, data in models.items():
        height_macro = MODEL_HEIGHTS[model_name]
        for dim in ("S", "A", "C"):
            entry = data[stem][dim]
            rows.append(
                r"\caseMeasureReason{"
                + height_macro
                + "}{"
                + tex_escape(model_name)
                + "}{"
                + answer_summary(entry)
                + "}{"
                + tex_escape(entry.get("reason", ""))
                + "}"
            )
    rows.extend([
        r"\advance\caseQwenH by 0.25em",
        r"\advance\caseSeedH by 0.25em",
    ])
    return "\n".join(rows)


def top_context_tex(stem: str) -> str:
    base = CHOICES_DIR / stem
    b_first, b_last = frame_pair(base / "before.mp4", safe_name(f"{stem}_before"))
    e_first, e_last = frame_pair(base / "after.mp4", safe_name(f"{stem}_after"))
    return "\n".join([
        r"\begin{center}",
        clip_pair_tex("BEGINNING: first / last", b_first, b_last, width=r"0.36\linewidth"),
        r"\hspace{1.0em}",
        clip_pair_tex("END: first / last", e_first, e_last, width=r"0.36\linewidth"),
        r"\end{center}",
    ])


def question_column_tex(stem: str, dim: str, models: dict[str, dict[str, Any]]) -> str:
    title = f"Question {DIM_LABEL[dim]}: {DIM_TITLE[dim]}"
    return "\n".join([
        r"\begin{minipage}[t]{0.318\linewidth}",
        r"\centering",
        r"\caseSubTitle{" + tex_escape(title) + "}",
        r"\casePrompt{Choose the middle clip that connects the shared BEGINNING and END.}",
        option_grid_tex(stem, dim),
        r"\vspace{0.35em}",
        model_rows_tex(stem, dim, models),
        r"\end{minipage}",
    ])


def sample_tex(stem: str, models: dict[str, dict[str, Any]], sources: dict[str, str]) -> str:
    source = sources.get(stem, "unknown")
    label = "fig:case-study-" + safe_name(f"{source}-{stem}").lower().replace("_", "-")
    columns = [
        question_column_tex(stem, dim, models)
        for dim in ("S", "A", "C")
    ]
    return "\n".join([
        r"\begin{figure*}[t]",
        r"\centering",
        r"\caseTitle{"
        + tex_escape(stem)
        + r"}{"
        + tex_escape(source)
        + r"}",
        measure_answer_heights_tex(stem, models),
        top_context_tex(stem),
        r"\vspace{0.4em}",
        columns[0],
        r"\hfill",
        columns[1],
        r"\hfill",
        columns[2],
        r"\caption{Qualitative case study on "
        + tex_escape(source)
        + r" sample \texttt{"
        + tex_escape(stem)
        + r"}. Each subtask shows the shared context, candidate middle clips, model answers, and model reasons.}",
        r"\label{" + label + r"}",
        r"\end{figure*}",
    ])


def preamble_macros() -> str:
    return r"""
% Auto-generated by figure/code/build_case_study_tex.py
% Required packages: graphicx, xcolor, array.
\definecolor{caseBlue}{HTML}{DDECF9}
\definecolor{caseGreen}{HTML}{198754}
\definecolor{caseRed}{HTML}{B42318}
\definecolor{caseGray}{HTML}{F3F4F6}
\newsavebox{\caseMeasureBox}
\newlength{\caseReasonWidth}
\newdimen\caseQwenH
\newdimen\caseSeedH
\newcommand{\caseTitle}[2]{%
  \noindent{\Large\bfseries #2 \texttt{#1}}%
  \par\vspace{0.45em}\hrule\vspace{0.55em}
}
\newcommand{\caseSubTitle}[1]{%
  {\bfseries #1}\par\vspace{0.2em}
}
\newcommand{\casePrompt}[1]{%
  {\footnotesize\emph{#1}}\par\vspace{0.35em}
}
\newcommand{\caseClip}[4]{%
  \begin{minipage}[t]{#4}
    \centering
    {\scriptsize\bfseries #1}\par\vspace{0.12em}
    \includegraphics[width=0.49\linewidth]{#2}\hfill
    \includegraphics[width=0.49\linewidth]{#3}\\[-0.1em]
    {\tiny first\hfill last}
  \end{minipage}
}
\newcommand{\caseReasonBody}[3]{%
  \raggedright
  \emergencystretch=1em
  {\scriptsize\bfseries #1}\quad {\scriptsize #2}\par
  {\scriptsize\emph{Reason:} #3}
}
\newcommand{\caseMeasureReason}[4]{%
  \begingroup
  \setlength{\fboxsep}{1.5pt}%
  \setlength{\caseReasonWidth}{0.31005\linewidth}%
  \sbox{\caseMeasureBox}{%
    \begin{minipage}{\dimexpr\caseReasonWidth-2\fboxsep-2\fboxrule\relax}
      \caseReasonBody{#2}{#3}{#4}
    \end{minipage}%
  }%
  \ifdim\dimexpr\ht\caseMeasureBox+\dp\caseMeasureBox\relax>#1
    \global#1=\dimexpr\ht\caseMeasureBox+\dp\caseMeasureBox\relax
  \fi
  \endgroup
}
\newcommand{\caseReason}[4]{%
  \vspace{0.28em}
  \begingroup
  \setlength{\fboxsep}{1.5pt}%
  \setlength{\caseReasonWidth}{0.975\linewidth}%
  \noindent\makebox[\linewidth][c]{%
  \fcolorbox{black!15}{caseGray}{%
    \begin{minipage}[t][#1][t]{\dimexpr\caseReasonWidth-2\fboxsep-2\fboxrule\relax}
      \caseReasonBody{#2}{#3}{#4}
      \vfill
    \end{minipage}
  }}%
  \endgroup\par
}
"""


def build_tex() -> str:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    models = {name: load_json(path) for name, path in MODEL_PATHS.items()}
    sources = source_lookup()
    sections = [sample_tex(stem, models, sources) for stem in CASE_STEMS]
    return preamble_macros() + "\n" + "\n\n".join(sections) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_tex(), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
