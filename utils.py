import os
os.environ["HF_HOME"] = "./hf_model"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import re, json, time
import argparse
import difflib
import zipfile
import tempfile
import shutil
import pandas as pd
from jiwer import wer, cer, process_characters, process_words
import soundfile as sf
import librosa
import numpy as np
import random
from transformers import set_seed
from tqdm import tqdm
import torch


# MultiClin dataset release. The archive extracts to a directory containing
# `labels.csv` and an `audios/` folder of 316 WAV files.
DATASET_DIR = "./interspeech26_multiscript_dataset"
# Single-zip release on Google Drive. A standard sharing URL works (fuzzy match).
DATASET_ZIP_URL = "https://drive.google.com/file/d/1hpFHn60g6sK_Eu0B3cNyjo6LPjsmssbS/view?usp=sharing"


def ensure_dataset(data_root=DATASET_DIR, zip_url=DATASET_ZIP_URL):
    """Ensure the MultiClin dataset is available at `data_root`.

    If `labels.csv` and `audios/` are not already present, the dataset zip is
    downloaded from Google Drive with gdown and extracted. The archive may wrap
    its contents in a top-level folder or not; both layouts are handled.
    """
    labels_path = os.path.join(data_root, "labels.csv")
    audios_dir = os.path.join(data_root, "audios")
    if os.path.exists(labels_path) and os.path.isdir(audios_dir):
        return data_root

    if not zip_url or zip_url.startswith("<"):
        raise RuntimeError(
            "DATASET_ZIP_URL is not set in utils.py. Set it to the Google Drive "
            "share link of the dataset zip, or place the dataset manually at "
            f"{data_root} (with labels.csv and audios/)."
        )

    try:
        import gdown
    except ImportError:
        raise ImportError("gdown is required to download the dataset. Install it with: pip install gdown")

    zip_path = os.path.join(tempfile.gettempdir(), "multiclin_dataset.zip")
    print(f"Dataset not found at {data_root}. Downloading from Google Drive...")
    gdown.download(zip_url, zip_path, quiet=False, fuzzy=True)

    print("Extracting dataset...")
    extract_tmp = tempfile.mkdtemp(prefix="multiclin_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_tmp)

    # Locate the directory that directly contains labels.csv + audios/.
    src_dir = None
    for cur, dirs, files in os.walk(extract_tmp):
        if "labels.csv" in files and "audios" in dirs:
            src_dir = cur
            break
    if src_dir is None:
        shutil.rmtree(extract_tmp, ignore_errors=True)
        raise RuntimeError("Downloaded archive does not contain labels.csv and audios/.")

    parent = os.path.dirname(os.path.abspath(data_root)) or "."
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    shutil.move(src_dir, os.path.abspath(data_root))

    shutil.rmtree(extract_tmp, ignore_errors=True)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"Dataset ready at {data_root}")
    return data_root


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASR models on the MultiClin dataset")

    parser.add_argument("--lang", type=str, default="ko", choices=['ko'],
                        help="Target language (Korean)")

    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash",
        help=(
            "Model name or path. Valid values depend on the evaluation script: "
            "eval_gemini.py accepts any Gemini API model (e.g. gemini-2.5-flash, gemini-2.5-pro; "
            "see the available models at https://ai.google.dev/gemini-api/docs/models); "
            "eval_whisper.py accepts any faster-whisper model size or local path "
            "(e.g. large-v3, large-v3-turbo; see the supported list at "
            "https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/utils.py); "
            "eval_qwen3asr.py accepts Qwen3-ASR models (e.g. Qwen3-ASR-0.6B, Qwen3-ASR-1.7B; "
            "see the available models at https://github.com/QwenLM/Qwen3-ASR), "
            "with the 'Qwen/' prefix added automatically."
        )
    )
    parser.add_argument("--api_key", type=str, default="", help="Gemini API key")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--data_root", type=str, default=DATASET_DIR,
                        help="Dataset directory (contains labels.csv and audios/). "
                             "Downloaded automatically if missing.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")

    parser.add_argument("--medical", type=str, default="both", choices=["original", "target", "both"], help="Evaluation mode for MEDICAL tags")
    parser.add_argument("--number", type=str, default="both", choices=["original", "target", "both"], help="Evaluation mode for NUMBER tags")
    parser.add_argument("--unit", type=str, default="both", choices=["original", "target", "both"], help="Evaluation mode for UNIT tags")

    return parser.parse_args()


def normalize_text(text):
    text = str(text).replace("[doctor]", "").replace("[doctor2]", "").replace("[patient]", "").replace("[guest_family]", "").replace("[guest_family2]", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def calculate_local_metrics(target, window_text):
    """
    Locate the best-matching span of `target` inside `window_text` and return
    its ((CER distance, WER distance), match end offset).

    The end offset is used to advance the search cursor so that subsequent tags
    are matched against the remaining hypothesis.
    """
    if not target:
        return (0, 0), 0
    if not window_text:
        return (len(target), len(target.split())), 0

    # Find the longest common substring to anchor the match (handles temporal misalignment).
    matcher = difflib.SequenceMatcher(None, target, window_text)
    match = matcher.find_longest_match(0, len(target), 0, len(window_text))

    if match.size == 0:
        return (len(target), len(target.split())), 0

    start_idx = match.b
    end_idx = min(start_idx + len(target), len(window_text))
    hyp_substring = window_text[start_idx:end_idx]

    # Compute the local CER/WER within the aligned bounds only.
    out_cer = process_characters(target, hyp_substring)
    cer_dist = out_cer.substitutions + out_cer.deletions + out_cer.insertions

    out_wer = process_words(target, hyp_substring)
    wer_dist = out_wer.substitutions + out_wer.deletions + out_wer.insertions

    return (cer_dist, wer_dist), match.b + match.size


def clean_tags_by_priority(text, hypothesis, medical, number, unit):
    """
    Resolve each multiscript tag in `text` to a single reference form.

    For 'both' mode, the variant (original script vs. target/local script) with
    the lower local error against the hypothesis is selected, comparing CER
    first and using WER as a tie-breaker.
    """
    master_pattern = re.compile(r'<(MEDICAL|NUMBER|UNIT)>(.*?)</\1>')
    search_cursor = 0

    mode_map = {"MEDICAL": medical, "NUMBER": number, "UNIT": unit}

    def replacer(match):
        nonlocal search_cursor

        tag_type = match.group(1)
        content = match.group(2)

        parts = content.split(',')
        if len(parts) < 2:
            return content

        orig_part = parts[0].strip()
        target_part = parts[1].strip()

        mode = mode_map.get(tag_type, "both")

        if mode == 'original':
            return orig_part
        elif mode == 'target':
            return target_part

        # both mode: pick the lower-error variant.
        if search_cursor >= len(hypothesis):
            return orig_part

        window_size = 50
        search_window = hypothesis[search_cursor:search_cursor + window_size]

        metrics_orig, end_orig = calculate_local_metrics(orig_part, search_window)
        metrics_target, end_target = calculate_local_metrics(target_part, search_window)

        # Tuple comparison: (CER, WER).
        if metrics_target < metrics_orig:
            result = target_part
            offset = end_target
        else:
            result = orig_part
            offset = end_orig

        search_cursor += offset
        return result

    return master_pattern.sub(replacer, text)


def load_audio_safely(audio_path, target_sr=16000):
    # Read the file as float32.
    try:
        data, native_sr = sf.read(audio_path, dtype='float32')
    except Exception as e:
        print(f"Failed to read audio file {audio_path}: {e}")
        return None, None

    # Convert to mono by averaging channels.
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if the native sample rate differs from the target.
    if native_sr != target_sr:
        data = librosa.resample(y=data, orig_sr=native_sr, target_sr=target_sr)

    return data.astype(np.float32), target_sr
