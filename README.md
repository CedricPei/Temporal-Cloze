# TempCloze: Can Video-LLMs Identify the Missing Middle?

[![Paper](https://img.shields.io/badge/arXiv-2609.01515-b31b1b.svg)](https://arxiv.org/abs/2609.01515)
[![Venue](https://img.shields.io/badge/EMNLP_2026-Findings-4b44ce.svg)](https://arxiv.org/abs/2609.01515)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://www.python.org/)

Official code, metadata, and evaluation toolkit for **TempCloze**, a video cloze benchmark for measuring visual temporal reasoning in Video-LLMs.

Given the **beginning** and **ending** of a video, a model must select the true missing middle from four video candidates. Because the inputs and answers are visual, the task reduces shortcuts from textual option wording, answer correlations, and language priors.

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

The main results are reported below. S, A, and P denote Semantic, Alignment, and Progression accuracy; cumulative accuracy measures the fraction of videos with at least one, at least two, or all three dimensions correct. All values are percentages.

| Model | Think | S | A | P | Mean | ≥1 | ≥2 | 3/3 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *Proprietary models* | | | | | | | | |
| Seed1.8 | ✓ | 95.07 | 76.92 | 93.75 | 88.58 | 99.54 | 95.40 | 70.81 |
| Qwen3.5-Plus | ✓ | 90.72 | 76.32 | 90.20 | 85.74 | 98.75 | 91.25 | 67.22 |
| Seed1.8 | ✗ | 96.25 | 61.93 | 92.11 | 83.43 | 99.35 | 94.02 | 56.94 |
| Gemini2.5-Pro | ✓ | 90.33 | 66.67 | 67.78 | 74.92 | 97.44 | 83.49 | 43.95 |
| Gemini2.5-Flash | ✗ | 82.18 | 59.83 | 74.82 | 72.28 | 94.61 | 78.17 | 44.05 |
| GPT5.4 | ✗ | 68.11 | 37.41 | 65.15 | 56.89 | 89.35 | 60.16 | 21.17 |
| Claude4.6-Sonnet | ✗ | 55.69 | 28.80 | 70.94 | 51.81 | 86.12 | 52.66 | 16.63 |
| Claude4.6-Opus | ✓ | 49.47 | 39.34 | 62.89 | 50.57 | 81.97 | 50.85 | 18.88 |
| Gemini3-Flash | ✗ | 61.08 | 40.37 | 48.13 | 49.86 | 84.09 | 49.57 | 15.91 |
| Seed1.6 | ✗ | 64.59 | 17.83 | 55.40 | 45.93 | 81.65 | 46.68 | 9.38 |
| Grok4.1 | ✗ | 24.52 | 24.06 | 23.73 | 24.11 | 57.53 | 14.07 | 0.72 |
| **Proprietary average** | — | **70.73** | **48.13** | **67.72** | **62.19** | **88.22** | **65.12** | **33.24** |
| *Open-source models* | | | | | | | | |
| Qwen3.5-397B-A17B | ✓ | 75.94 | 50.62 | 78.24 | 68.27 | 94.54 | 74.49 | 35.77 |
| Qwen3.5-35B-A3B | ✓ | 75.10 | 51.55 | 69.96 | 65.54 | 92.33 | 70.84 | 33.80 |
| KimiK2.5 | ✗ | 71.27 | 41.95 | 65.02 | 59.41 | 89.94 | 62.79 | 25.51 |
| Qwen3VL-32B-I | ✗ | 51.35 | 10.91 | 63.25 | 41.84 | 78.70 | 41.36 | 5.46 |
| InternVL3.5-38B | ✗ | 37.81 | 32.63 | 46.23 | 38.96 | 70.43 | 36.54 | 11.35 |
| Qwen3VL-8B-I | ✗ | 31.14 | 15.06 | 45.46 | 30.55 | 64.49 | 23.71 | 3.49 |
| InternVL3.5-8B | ✗ | 26.30 | 28.34 | 29.52 | 28.05 | 60.15 | 20.18 | 3.81 |
| InternVL3-38B | ✗ | 30.31 | 20.12 | 32.81 | 27.74 | 59.04 | 21.24 | 2.96 |
| KimiVL-A3B-T | ✓ | 30.17 | 25.52 | 24.98 | 26.91 | 62.17 | 16.23 | 2.17 |
| Qwen3VL-32B-T | ✓ | 25.05 | 25.38 | 25.72 | 25.38 | 58.76 | 15.40 | 1.91 |
| LLaVA-CriticR1-7B | ✓ | 23.81 | 26.90 | 25.36 | 25.36 | 58.22 | 16.17 | 1.73 |
| Qwen2.5VL-7B-I | ✗ | 27.22 | 26.22 | 24.17 | 25.87 | 57.93 | 17.70 | 2.11 |
| GLM4.6V-Flash | ✗ | 21.78 | 14.71 | 39.58 | 25.36 | 55.64 | 17.36 | 3.04 |
| Qwen3VL-4B-T | ✓ | 27.81 | 24.85 | 22.95 | 25.20 | 58.38 | 15.58 | 1.64 |
| Qwen3VL-8B-T | ✓ | 24.65 | 25.38 | 24.59 | 24.87 | 57.53 | 15.45 | 1.64 |
| ThinkLiteVL-7B | ✓ | 24.36 | 25.79 | 23.58 | 24.58 | 57.80 | 14.60 | 1.33 |
| MiMoVL-7B-RL | ✗ | 20.84 | 23.73 | 28.80 | 24.46 | 53.91 | 16.70 | 2.76 |
| Qwen3VL-4B-I | ✗ | 20.38 | 19.72 | 32.48 | 24.19 | 54.30 | 15.51 | 2.76 |
| KimiVL-A3B-I | ✗ | 21.09 | 26.78 | 23.13 | 23.67 | 52.23 | 15.60 | 2.75 |
| MiMoVL-7B-SFT | ✗ | 23.41 | 20.51 | 25.77 | 23.23 | 52.86 | 14.60 | 2.24 |
| Molmo2-8B | ✗ | 24.19 | 20.70 | 24.73 | 23.21 | 53.50 | 14.25 | 1.88 |
| **Open-source average** | — | **34.00** | **26.54** | **36.97** | **32.51** | **63.95** | **26.49** | **7.15** |
| **Human baseline** | — | **96.00** | **98.00** | **97.00** | **97.00** | **100.00** | **99.00** | **92.00** |
| Random | — | 25.00 | 25.00 | 25.00 | 25.00 | 57.81 | 15.62 | 1.56 |

### Behavioral analysis

We study four representative models on TempCloze-Mixed and TempCloze-Hard, analyzing candidate-order permutations, context direction, visible span, frame sampling density, and test-time scaling. The results show that model choices can be unstable under candidate reordering, models rely more on beginning context than ending context, denser or longer visual inputs can dilute decisive boundary cues, and test-time scaling brings model-dependent gains without removing Alignment as the main bottleneck.

## Evaluation

TempCloze requires Python 3.10 or newer. Install the dependencies and make sure the prepared benchmark is available at `choices/` in the repository root.

```bash
git clone https://github.com/CedricPei/Temporal-Cloze.git
cd Temporal-Cloze

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Local open-source models with vLLM

Install vLLM in an environment compatible with your model and accelerator, then start its OpenAI-compatible server. The default evaluation samples 16 frames per clip, or up to 96 images per request.

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

Create `.env` in the repository root or under `TempCloze/`:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
```

Set `EVAL_MODEL` near the top of `TempCloze/eval.py`, expose the benchmark at `TempCloze/choices`, and run:

```bash
ln -s ../choices TempCloze/choices
python TempCloze/eval.py video-cloze
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

TempCloze is built from CaReBench, Daily-Omni, EgoLife, FAVOR-Bench, LVD-2M, MiraData, and Video Thinking Test.
