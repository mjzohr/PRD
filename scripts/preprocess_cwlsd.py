"""Slice the CWLSD external cohort and build its manifest (external test only, fold -1)."""
import os
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from prd.config import Config

RAW_DIR = Config.PROJECT_ROOT / "data" / "CWLSD"

COLUMNS = [
    "filename", "pid", "label_idx", "disease", "age", "sex",
    "crackles", "wheezes", "fold", "cohort", "split_role", "recording_type",
]


def parse_filename(filename):
    # <PREFIX><id>_<diagnosis>,<sound>,<location>,<age>,<sex>.wav, e.g. BP100_N,N,P R M,70,F.wav.
    # The prefix is the recording position; the numeric id identifies the patient.
    stem = filename[:-4] if filename.lower().endswith(".wav") else filename
    if "_" not in stem:
        return None
    head, rest = stem.split("_", 1)

    m = re.match(r"^([A-Za-z]+)(\d+)$", head.strip())
    if not m:
        return None
    recording_type = m.group(1)
    pid = int(m.group(2))

    parts = [p.strip() for p in rest.split(",")]
    diagnosis = parts[0] if len(parts) > 0 else ""
    sound = parts[1] if len(parts) > 1 else ""
    age_raw = parts[3] if len(parts) > 3 else None
    sex_raw = parts[4] if len(parts) > 4 else None

    if not diagnosis:
        return None

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
    total = len(audio)
    if total < Config.TARGET_LEN:
        yield 0, np.pad(audio, (0, Config.TARGET_LEN - total), mode="constant")
    else:
        for start_idx in range(0, total - Config.TARGET_LEN + 1, Config.STEP_SIZE):
            yield start_idx, audio[start_idx:start_idx + Config.TARGET_LEN]


def process():
    files = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".wav"))
    print(f"slicing {len(files)} CWLSD files from {RAW_DIR}")

    rows = []
    parsed = skipped = 0
    for filename in files:
        info = parse_filename(filename)
        if info is None:
            print(f"skip, unparseable name: {filename}")
            skipped += 1
            continue

        try:
            audio, _ = librosa.load(RAW_DIR / filename, sr=Config.SAMPLE_RATE, mono=True)
        except Exception as exc:
            print(f"skip, failed to load {filename}: {exc}")
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
    print(f"parsed {parsed} files, skipped {skipped}, {len(manifest)} slices")
    if not manifest.empty:
        print(f"classes: {manifest['disease'].value_counts().to_dict()}")
        print(f"recording types: {manifest['recording_type'].value_counts().to_dict()}")
    return manifest


def main():
    Config.set_seed(Config.SEED)
    Config.EXTERNAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Config.EXTERNAL_CSV.parent.mkdir(parents=True, exist_ok=True)

    manifest = process()
    if not (manifest["fold"] == -1).all() or not (manifest["split_role"] == "external_test").all():
        raise RuntimeError("CWLSD must be external test only")

    manifest.to_csv(Config.EXTERNAL_CSV, index=False)
    print(f"wrote {Config.EXTERNAL_CSV}")


if __name__ == "__main__":
    main()
