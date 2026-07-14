# When Multiple Scripts Matter: Evaluating ASR in Clinical Settings

[![arXiv](https://img.shields.io/badge/arXiv-2606.17826-b31b1b.svg)](https://arxiv.org/abs/2606.17826)

[Jean Seo](mailto:jean.seo@di.ku.dk)<sup>1,2,\*</sup>,
[Minkyu Kim](mailto:minkyu.kim@aitrics.com)<sup>1,\*</sup>,
Jeonguk Lee<sup>1</sup>,
Jisoo Jung<sup>1</sup>,
Wooseok Han<sup>1</sup>,
Eunho Yang<sup>1,3,†</sup>

<sup>1</sup> AITRICS &nbsp;&nbsp; <sup>2</sup> University of Copenhagen &nbsp;&nbsp; <sup>3</sup> KAIST
<br><sup>\*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Corresponding author

Official code and dataset for the **MultiClin** benchmark, introduced in
*"When Multiple Scripts Matter: Evaluating ASR in Clinical Settings"* (Interspeech 2026).

## Updates

- **2026-07** — Added a `--suppress_hallucinations` flag to `eval_whisper.py`. After the
  camera-ready deadline, a per-utterance error analysis showed that Whisper-family models
  are prone to repetition-loop hallucinations on long audio (the previous segment is fed
  back as the decoding prompt, so a repeated sentence reinforces itself), which inflates
  insertion errors. The flag decodes with anti-hallucination constraints
  (`condition_on_previous_text=False`, repetition penalties, hallucination silence
  skipping) and raises Whisper-family scores substantially — e.g., Whisper large-v3 on
  MultiClin-ko improves from 86.2% to 90.4% (1 − CER). The **default decoding is
  unchanged** and reproduces the numbers reported in the paper.

## Overview

Automatic speech recognition (ASR) in non-English clinical settings is challenged by
**multiscript variability**: a single spoken term can correspond to multiple valid
orthographic forms (e.g., the conventional English spelling of a medical term vs. its
phonetic rendering in the local script). Conventional string-based metrics such as WER
assume a single canonical transcript, so they systematically *underestimate* model
performance when the output is phonetically and semantically correct but orthographically
divergent.

**MultiClin** is a clinical ASR benchmark. The paper presents an in-depth Korean case
study; this release additionally provides **Japanese (`ja`), Chinese (`zh`), and Arabic
(`ar`)** alongside Korean (`ko`). Each reference transcript tags script-switching entities
(`MEDICAL`, `NUMBER`, `UNIT`) with both their original and local-script variants. We pair
this with a **dynamic, multiscript-aware evaluation protocol** (Algorithm 1 in the paper)
that, for each tagged entity, locally resolves which variant best matches the ASR
hypothesis before computing CER/WER. Across diverse models, this consistently reveals
higher performance than traditional single-label string matching.

The dataset is synthesized from publicly available doctor–patient dialogues
(ACI-Bench, PriMock57, MTS-Dialog) to comply with HIPAA restrictions on releasing
real clinical audio.

## Repository contents

| File | Description |
|------|-------------|
| `utils.py` | Shared utilities: argument parsing, text normalization, dataset download, and the dynamic multiscript reference resolution (`clean_tags_by_priority`). |
| `eval_whisper.py` | Run inference with Whisper models (via `faster-whisper`). |
| `eval_qwen3asr.py` | Run inference with Qwen3-ASR models. |
| `eval_gemini.py` | Run inference with Gemini multimodal models. |
| `calculate_score.py` | Compute CER/WER over all script-evaluation modes from an inference result CSV. |

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the exact environment used for the paper and pulls the
CUDA 12.8 PyTorch build via an `--extra-index-url`, so a single command installs
everything in the correct order. For a different CUDA/CPU build, edit the index URL
and the `+cu128` torch suffixes in `requirements.txt`.

You only need the backend(s) for the model(s) you intend to run: `faster-whisper`
(eval_whisper.py), `qwen-asr` (eval_qwen3asr.py), or `google-genai` (eval_gemini.py).

## Dataset

Each language is released as a separate zip. The dataset for the selected `--lang` is
downloaded and extracted automatically on the first run of any script via
[`gdown`](https://github.com/wkentaro/gdown). After extraction the per-language layout is:

```
multiscript_dataset/lang_<lang>/
├── labels.csv      # src, multiscript (tagged reference), TTS (spoken form)
└── audios/         # 316 synthesized WAV files (16 kHz), named <src>.wav
```

The reference column used for evaluation is `multiscript`, where each tagged entity is
written as `<TAG>original,local</TAG>`, e.g. `<MEDICAL>brace,브레이스</MEDICAL>` (ko) or
`<MEDICAL>brace,ブレース</MEDICAL>` (ja). The `src` identifiers are shared across languages.

> **Note:** The per-language Google Drive links live in `DATASET_ZIP_URLS` in
> [`utils.py`](utils.py). Alternatively, place a dataset manually at
> `./multiscript_dataset/lang_<lang>` (with `labels.csv` and `audios/`) to skip the download.

## Usage

Evaluation is a two-step process: **(1)** run a model to produce an inference result CSV,
then **(2)** score it across all script-evaluation modes.

### 1. Run inference

Pass `--lang` to choose the language (`ko`, `ja`, `zh`, `ar`; default `ko`). Each script
writes `results/result_<model>_<lang>_Med-<m>_Num-<n>_Unit-<u>.csv`, containing the `src`,
`hypothesis`, and resolved `best_reference` for every dialogue.

```bash
# Whisper (faster-whisper) — Japanese
python eval_whisper.py --lang ja --model large-v3 --device cuda

# Qwen3-ASR — Chinese
python eval_qwen3asr.py --lang zh --model Qwen3-ASR-0.6B --device cuda

# Gemini — Arabic
python eval_gemini.py --lang ar --model gemini-2.5-flash --api_key $GEMINI_API_KEY
```

Common options (see `utils.py` for the full list):

| Argument | Default | Description |
|----------|---------|-------------|
| `--lang` | `ko` | Target language: `ko`, `ja`, `zh`, or `ar`. |
| `--data_root` | `./multiscript_dataset/lang_<lang>` | Dataset directory (auto-downloaded if missing). |
| `--output_dir` | `./results` | Where result CSVs are written. |
| `--medical` / `--number` / `--unit` | `both` | Per-tag evaluation mode: `original`, `target`, or `both`. |

### 2. Compute scores

`calculate_score.py` recomputes CER/WER for every `(medical × number × unit)` combination
of `{original, target, both}` from a single result CSV (the hypotheses are mode-independent):

```bash
python calculate_score.py --lang ja --model large-v3
```

This prints a CER/WER table where each row corresponds to one evaluation mode, reproducing
the per-mode results reported in the paper. Transitioning from strict single-label
matching (`original`) to the multiscript-aware criterion (`both`) yields consistent error
reductions across all models.

## Evaluation protocol

For each script-switching entity in the reference, the protocol extracts a 50-character
window from the ASR hypothesis (tracked by a moving cursor), applies Longest Common
Substring matching to handle temporal misalignment, and computes the *local* CER/WER within
the aligned bounds. The variant (original vs. local script) with the lower local error is
selected as the reference for that entity. See `clean_tags_by_priority` /
`calculate_local_metrics` in [`utils.py`](utils.py) and Algorithm 1 in the paper.

## Citation

```bibtex
@misc{seo2026multiplescriptsmatterevaluating,
      title={When Multiple Scripts Matter: Evaluating ASR in Clinical Settings}, 
      author={Jean Seo and Minkyu Kim and Jeonguk Lee and Jisoo Jung and Wooseok Han and Eunho Yang},
      year={2026},
      eprint={2606.17826},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.17826}, 
}
```
