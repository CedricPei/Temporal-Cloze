# TempCloze: Can Video-LLMs Identify the Missing Middle?

[![Paper](https://img.shields.io/badge/arXiv-2609.01515-b31b1b.svg)](https://arxiv.org/abs/2609.01515)
[![Venue](https://img.shields.io/badge/EMNLP_2026-Findings-4b44ce.svg)](https://arxiv.org/abs/2609.01515)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://www.python.org/)

Official code, metadata, and evaluation toolkit for **TempCloze**, a video cloze benchmark for measuring visual temporal reasoning in Video-LLMs.

Given the **beginning** and **ending** of a video, a model must select the true missing middle from four video candidates. Because the inputs and answers are visual, the task reduces shortcuts from textual option wording, answer correlations, and language priors.

> **Paper:** [TempCloze: Can Video-LLMs Identify the Missing Middle?](https://arxiv.org/abs/2609.01515)<br>
> **Venue:** Findings of EMNLP 2026

## Benchmark at a glance

- **1,521 videos** from seven public sources, with an emphasis on long-take and egocentric footage.
- **Three temporal dimensions** per video: Semantic, Alignment, and Progression.
- **Four-way video multiple choice:** one ground-truth middle and three same-source distractors.
- **31 evaluated Video-LLMs:** 10 proprietary and 21 open-source models.
- **Default evaluation:** 16 uniformly sampled frames per clip, or 96 frames for each six-clip question.

Each source video is split into a beginning \(B\), missing middle \(M\), and ending \(E\). The candidate set for each dimension contains \(M\) and three distractors:

| Dimension | What it tests | Distractors in this repository |
| --- | --- | --- |
| Semantic (S) | **What** event belongs in the gap | Three non-overlapping intervals from the same source video (`Rand1`–`Rand3`) |
| Alignment (A) | **When** the event should occur | Shifted earlier (`Early`), shifted later (`Late`), and expanded (`Wide`) intervals |
| Progression (P) | **How** the event unfolds | Reversed (`Reverse`), reordered (`Shuffle`), and repeated (`Loop`) clips |

The progression data is stored under `C/` in the current file layout for backward compatibility; evaluation reports display this dimension as **P**.

### Dataset composition

| Source | Videos |
| --- | ---: |
| LVD-2M | 515 |
| EgoLife | 437 |
| MiraData | 198 |
| FAVOR-Bench | 145 |
| CaReBench | 94 |
| Video-TT | 89 |
| Daily-Omni | 43 |
| **Total** | **1,521** |

### Main finding

Alignment is the main bottleneck. The proprietary-model average falls from 70.73% on Semantic and 67.72% on Progression to 48.13% on Alignment; the open-source average falls to 26.54% on Alignment. Humans reach 97.00% mean accuracy under the paper's strict evaluation protocol.

| System | S | A | P | Mean |
| --- | ---: | ---: | ---: | ---: |
| Seed1.8 (thinking) | 95.07 | 76.92 | 93.75 | **88.58** |
| Qwen3.5-397B-A17B | 75.94 | 50.62 | 78.24 | **68.27** |
| Human baseline | 96.00 | 98.00 | 97.00 | **97.00** |
| Random | 25.00 | 25.00 | 25.00 | **25.00** |

See the [paper](https://arxiv.org/abs/2609.01515) for the complete leaderboard, cumulative accuracy, error attribution, and behavioral sensitivity analyses.

## Repository contents

```text
Temporal-Cloze/
├── TempCloze/              # Main, mixed, hard-subset, and baseline evaluation
│   ├── eval.py             # OpenAI-compatible hosted API evaluation
│   ├── eval_vllm.py        # Local vLLM evaluation
│   ├── eval_results/       # Analysis scripts and released reports
│   ├── mixed/              # TempCloze-Mixed (300 videos)
│   └── subset/             # TempCloze-Hard and sensitivity experiments
├── src/                    # Source-specific filtering/download and generation code
├── output/                 # Released filtering decisions and gap metadata
├── human-eval/             # Streamlit human-baseline and quality-review interfaces
├── figure/                 # Scripts and assets used for paper figures
└── requirements.txt
```

Raw videos and generated `choices/` directories are not committed to Git. The repository contains the construction code, the retained gap metadata, analysis subsets, evaluation utilities, and released model results. Obtain each source dataset under its original terms before reconstructing video questions.

## Installation

### 1. Clone and create an environment

```bash
git clone https://github.com/CedricPei/Temporal-Cloze.git
cd Temporal-Cloze

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10 or newer is recommended. Benchmark construction also requires `ffmpeg` and `ffprobe` on `PATH`:

```bash
ffmpeg -version
ffprobe -version
```

The source-specific download utilities may additionally require `yt-dlp`, Node.js, source metadata files, or access credentials. Local open-source evaluation requires a separate vLLM environment compatible with the selected model and accelerator.

### 2. Configure an OpenAI-compatible API (when needed)

Create `.env` in the repository root or under `TempCloze/`:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
```

Do not commit API keys. `.env` files are ignored by Git.

## Data preparation

### Expected question layout

Evaluation scripts expect one directory per video:

```text
choices/<video_id>/
├── before.mp4
├── GT.mp4
├── after.mp4
├── S/
│   ├── Rand1.mp4
│   ├── Rand2.mp4
│   └── Rand3.mp4
├── A/
│   ├── Early.mp4
│   ├── Late.mp4
│   └── Wide.mp4
└── C/                       # Progression (P) in the paper
    ├── Reverse.mp4
    ├── Shuffle.mp4
    └── Loop.mp4
```

### Reconstruct questions from source videos

The released files at `output/<source>/meta.json` map source video filenames to the retained `gap_start` and `gap_end` boundaries. Supported source keys are:

```text
care  dailyomni  egolife  favor  lvd  mira  tt
```

For one source at a time:

1. Obtain the corresponding videos from the original dataset.
2. Preserve the filenames used by `output/<source>/meta.json`.
3. Put the videos in `src/downloaded/`.
4. Generate the question clips:

```bash
python src/generate.py lvd
```

Generated questions are written to the repository-root `choices/` directory. Repeat with another source key after preparing that source's videos.

> [!WARNING]
> After a question is generated successfully, `src/generate.py` deletes its processed source video from `src/downloaded/`. Keep a separate copy of the original data if you need it later.

The scripts under `src/<source>/download_*.py` document the filtering and download pipeline used for each source. Some of their input metadata is distributed by the original datasets and is intentionally not included here.

## Evaluation

### Local open-source models with vLLM

Start an OpenAI-compatible vLLM server with a vision-language model. A request contains up to 96 images at the default 16 frames per clip, so raise the multimodal image limit accordingly:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8002 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 96}'
```

In another shell, run the full benchmark:

```bash
CHOICES_DIR=../choices \
VLLM_BASE_URL=http://127.0.0.1:8002 \
VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
EVAL_NUM_FRAMES=16 \
python TempCloze/eval_vllm.py --output-name qwen2.5-vl-7b
```

Evaluate selected video IDs by passing them as positional arguments:

```bash
CHOICES_DIR=../choices \
VLLM_BASE_URL=http://127.0.0.1:8002 \
VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
python TempCloze/eval_vllm.py <video_id_1> <video_id_2>
```

Useful environment variables are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHOICES_DIR` | `TempCloze/choices` | Question directory; paths are resolved relative to `TempCloze/` |
| `EVAL_RESULTS_DIR` | `TempCloze/eval_results/customization_exp` | Result directory |
| `VLLM_BASE_URL` | `http://127.0.0.1:8002` | vLLM server URL |
| `VLLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Served model identifier |
| `EVAL_NUM_FRAMES` | `16` | Frames sampled from each clip |
| `EVAL_NUM_WORKERS` | `16` | Concurrent evaluation workers |
| `EVAL_TASK_ORDER` | `by_stem` | Task ordering; use `by_dim` for dimension-first evaluation |

Results are saved incrementally, and rerunning the same output resumes unfinished questions.

### Reproduce released result summaries

The repository includes the per-model results used by the paper. Recompute the main and mixed-set summaries with:

```bash
python TempCloze/eval_results/analyze.py all
python TempCloze/mixed/analyze_mixed.py
```

### Hosted OpenAI-compatible APIs

`TempCloze/eval.py` evaluates the three dimensions through an OpenAI-compatible Chat Completions endpoint. Set `EVAL_MODEL` near the top of the script to the endpoint's model name, expose the generated questions at `TempCloze/choices`, and run:

```bash
ln -s ../choices TempCloze/choices
python TempCloze/eval.py video-cloze
python TempCloze/eval.py video-cloze 200 --seed 42
```

The first command evaluates all videos. The second evaluates a reproducible 200-video sample. Existing results are reused for checkpoint-style continuation.

### Additional experiments

| Task | Entry point |
| --- | --- |
| TempCloze-Mixed | `TempCloze/mixed/eval_mixed.py` |
| Context direction, frame density, permutation, and test-time scaling | `TempCloze/subset/` |
| Exact boundary-frame sanity check | `TempCloze/boundary_sanity_check.py` |
| DINOv2 edge-matching baseline | `TempCloze/edge_matching_baseline.py` |
| Single-frame Video-LLM baseline | `TempCloze/single_frame_baseline_api.py` |
| Human baseline interface | `human-eval/human_eval_app.py` |
| Dataset-quality review interface | `human-eval/human_quality_eval_app.py` |

The DINOv2 baseline additionally requires PyTorch and Transformers. Run the human interface with Streamlit, pointing it to a prepared choice-set directory:

```bash
HUMAN_EVAL_CHOICES_DIR=../choices_human \
streamlit run human-eval/human_eval_app.py
```

## Citation

If TempCloze is useful in your research, please cite:

```bibtex
@article{pei2026tempcloze,
  title   = {TempCloze: Can Video-LLMs Identify the Missing Middle?},
  author  = {Pei, Wenqi and Zhao, Henry Hengyuan and Liu, Yilai and Meng, Jiahao and Chen, Han and Wang, Ziyu and Du, Hongyang},
  journal = {arXiv preprint arXiv:2609.01515},
  year    = {2026}
}
```

## License and data terms

The paper is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No repository-level software license is included in the current release. Videos remain subject to the licenses and terms of their original source datasets; this repository does not redistribute the raw videos.

## Acknowledgements

TempCloze is built from CaReBench, Daily-Omni, EgoLife, FAVOR-Bench, LVD-2M, MiraData, and Video Thinking Test. Please also cite the corresponding source datasets when using reconstructed benchmark videos.
