# TempCloze: Can Video-LLMs Identify the Missing Middle?

[![Paper](https://img.shields.io/badge/arXiv-2609.01515-b31b1b.svg)](https://arxiv.org/abs/2609.01515)
[![Venue](https://img.shields.io/badge/EMNLP_2026-Findings-4b44ce.svg)](https://arxiv.org/abs/2609.01515)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://www.python.org/)

Official code, metadata, and evaluation toolkit for **TempCloze**, a video cloze benchmark for measuring visual temporal reasoning in Video-LLMs.

Given the **beginning** and **ending** of a video, a model must select the true missing middle from four video candidates. Because the inputs and answers are visual, the task reduces shortcuts from textual option wording, answer correlations, and language priors.

> **Paper:** [TempCloze: Can Video-LLMs Identify the Missing Middle?](https://arxiv.org/abs/2609.01515)<br>
> **Venue:** Findings of EMNLP 2026

<p align="center">
  <img src="https://arxiv.org/html/2609.01515v1/main-figure.png" alt="Overview of the TempCloze benchmark" width="900">
</p>
<p align="center"><em>TempCloze asks a model to recover the missing middle using Semantic, Alignment, and Progression candidates.</em></p>

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

## Quick start

TempCloze requires Python 3.10 or newer. Benchmark construction also requires `ffmpeg` and `ffprobe` on `PATH`.

```bash
git clone https://github.com/CedricPei/Temporal-Cloze.git
cd Temporal-Cloze

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For hosted-model evaluation, create `.env` in the repository root or under `TempCloze/`:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
```

## Data preparation

Raw videos are not included. To reconstruct a source split:

1. Obtain the corresponding videos from the original dataset.
2. Preserve the filenames listed in `output/<source>/meta.json` and place the videos in `src/downloaded/`.
3. Run the generator with one of `care`, `dailyomni`, `egolife`, `favor`, `lvd`, `mira`, or `tt`:

```bash
python src/generate.py lvd
```

Questions are written to `choices/<video_id>/`, containing `before.mp4`, `GT.mp4`, `after.mp4`, and the `S/`, `A/`, and `C/` distractor directories. Here `C/` stores the Progression (P) dimension.

> [!WARNING]
> After a question is generated successfully, `src/generate.py` deletes its processed source video from `src/downloaded/`. Keep a separate copy of the original data if you need it later.

## Evaluation

### Local open-source models with vLLM

Install vLLM in an environment compatible with your model and accelerator, then start its OpenAI-compatible server. The default setup uses 16 frames per clip, or up to 96 images per request.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8002 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 96}'
```

Run the benchmark in another shell:

```bash
CHOICES_DIR=../choices \
VLLM_BASE_URL=http://127.0.0.1:8002 \
VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
EVAL_NUM_FRAMES=16 \
python TempCloze/eval_vllm.py --output-name qwen2.5-vl-7b
```

Results are saved incrementally, and rerunning the same output resumes unfinished questions.

### Hosted OpenAI-compatible APIs

Set `EVAL_MODEL` near the top of `TempCloze/eval.py`, expose the generated questions at `TempCloze/choices`, and run:

```bash
ln -s ../choices TempCloze/choices
python TempCloze/eval.py video-cloze
```

To evaluate a reproducible subset, provide its size and seed:

```bash
python TempCloze/eval.py video-cloze 200 --seed 42
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

## Acknowledgements

TempCloze is built from CaReBench, Daily-Omni, EgoLife, FAVOR-Bench, LVD-2M, MiraData, and Video Thinking Test. Please also cite the corresponding source datasets when using reconstructed benchmark videos.
