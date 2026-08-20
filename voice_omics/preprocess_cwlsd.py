"""Preprocess the CWLSD external cohort.

Slices the raw CWLSD wavs into fixed 6-second windows (mirroring the ICHBI
``preprocessing.py`` slicing logic) and writes a slice-level manifest. CWLSD is
external-test only: ``fold = -1`` and ``split_role = 'external_test'`` for every
slice -- no folds are assigned and no parameters are learned here.

Filenames encode clinical info::

    <PREFIX><id>_<diagnosis>,<sound>,<location>,<age>,<sex>.wav
    e.g. BP100_N,N,P R M,70,F.wav

``PREFIX`` (BP/EP/DP/...) is the recording position (``recording_type``); the
numeric ``<id>`` identifies the PATIENT (same id under different prefixes == same
patient), so ``pid = int(<id>)``.

Run with ``python -m preprocess_cwlsd`` from inside the flat package.
"""

import os
import re

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from config import Config

RAW_DIR = Config.PROJECT_ROOT / "data" / "CWLSD"

# Manifest column order (exact).
COLUMNS = [
    "filename", "pid", "label_idx", "disease", "age", "sex",
    "crackles", "wheezes", "fold", "cohort", "split_role", "recording_type",
]


def parse_filename(filename):
    """Parse a CWLSD wav filename into clinical fields.

    Returns a dict with recording_type, pid, disease (3-class name), label_idx,
    age, sex, crackles, wheezes -- or ``None`` if the name is unparseable.
    """
    stem = filename[:-4] if filename.lower().endswith(".wav") else filename
    if "_" not in stem:
        return None
    head, rest = stem.split("_", 1)

    # head = PREFIX (leading letters) + id (trailing digits)
    m = re.match(r"^([A-Za-z]+)(\d+)$", head.strip())
    if not m:
        return None
    recording_type = m.group(1)
    pid = int(m.group(2))

    # rest = diagnosis, sound, location, age, sex  (be defensive)
    parts = [p.strip() for p in rest.split(",")]
    diagnosis = parts[0] if len(parts) > 0 else ""
    sound = parts[1] if len(parts) > 1 else ""
    age_raw = parts[3] if len(parts) > 3 else None
    sex_raw = parts[4] if len(parts) > 4 else None

    if not diagnosis:
        return None

    # 3-class label mapping.
    diag = diagnosis.lower().strip()
    if diag in ("n", "normal", "healthy"):
        disease, label_idx = "Healthy", 0
    elif diag == "copd":
        disease, label_idx = "COPD", 1
    else:
        disease, label_idx = "Other", 2

    s = sound.lower()
    crackles = 1 if "crep" in s else 0
    wheezes = 1 if "w" in s else 0

    try:
        age = int(float(age_raw))
    except (TypeError, ValueError):
        age = ""
    sex = sex_raw if sex_raw else ""

    return {
        "recording_type": recording_type,
        "pid": pid,
        "disease": disease,
        "label_idx": label_idx,
        "age": age,
        "sex": sex,
        "crackles": crackles,
        "wheezes": wheezes,
    }


def slice_audio(audio):
    """Yield (start_idx, slice) 6s windows at 50% overlap; pad short clips."""
    total = len(audio)
    if total < Config.TARGET_LEN:
        padded = np.pad(audio, (0, Config.TARGET_LEN - total), mode="constant")
        yield 0, padded
    else:
        for start_idx in range(0, total - Config.TARGET_LEN + 1, Config.STEP_SIZE):
            yield start_idx, audio[start_idx:start_idx + Config.TARGET_LEN]


def process():
    """Parse, slice, and write CWLSD slices; return the manifest DataFrame."""
    files = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".wav"))
    print(f"--- Slicing CWLSD ({len(files)} wav files found in {RAW_DIR}) ---")

    rows = []
    parsed = skipped = 0
    for filename in files:
        info = parse_filename(filename)
        if info is None:
            print(f"[skip] unparseable filename: {filename}")
            skipped += 1
            continue

        try:
            audio, _ = librosa.load(RAW_DIR / filename, sr=Config.SAMPLE_RATE, mono=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] failed to load {filename}: {exc}")
            skipped += 1
            continue

        parsed += 1
        stem = filename[:-4]
        for start_idx, audio_slice in slice_audio(audio):
            slice_name = f"{stem}_{start_idx}.wav"
            sf.write(Config.EXTERNAL_PROCESSED_DIR / slice_name, audio_slice, Config.SAMPLE_RATE)
            rows.append({
                "filename": slice_name,
                "pid": int(info["pid"]),
                "label_idx": info["label_idx"],
                "disease": info["disease"],
                "age": info["age"],
                "sex": info["sex"],
                "crackles": info["crackles"],
                "wheezes": info["wheezes"],
                "fold": -1,
                "cohort": "CWLSD",
                "split_role": "external_test",
                "recording_type": info["recording_type"],
            })

    manifest = pd.DataFrame(rows, columns=COLUMNS)
    print(f"\nParsed {parsed} files, skipped {skipped}, total {len(manifest)} slices.")
    if not manifest.empty:
        print(f"Class distribution: {manifest['disease'].value_counts().to_dict()}")
        print(f"recording_type distribution: {manifest['recording_type'].value_counts().to_dict()}")
    return manifest


def main():
    Config.set_seed(Config.SEED)
    Config.EXTERNAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Config.EXTERNAL_CSV.parent.mkdir(parents=True, exist_ok=True)

    manifest = process()

    if not (manifest["fold"] == -1).all() or not (manifest["split_role"] == "external_test").all():
        raise RuntimeError("CWLSD external-only policy was violated")

    manifest.to_csv(Config.EXTERNAL_CSV, index=False)
    print(f"\nSaved manifest to {Config.EXTERNAL_CSV}")


if __name__ == "__main__":
    main()
