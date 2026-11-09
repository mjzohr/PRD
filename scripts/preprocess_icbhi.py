"""Slice ICBHI recordings into fixed-length clips and assign patient-level folds."""
import os
import sys
import warnings
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from prd.config import Config


def load_metadata():
    diag = pd.read_csv(Config.DIAG_FILE, names=["pid", "disease"], header=None)
    diag["pid"] = pd.to_numeric(diag["pid"], errors="coerce")
    demo = pd.read_csv(Config.DEMO_FILE, sep="\t", names=["pid", "age", "sex", "bmi", "wt", "ht"])
    demo["pid"] = pd.to_numeric(demo["pid"], errors="coerce")

    meta = pd.merge(demo, diag, on="pid", how="inner").dropna(subset=["pid"])
    meta["target_class"] = meta["disease"].map(Config.CLASS_MAPPING)
    meta = meta.dropna(subset=["target_class"])
    class_to_idx = {name: idx for idx, name in enumerate(Config.CLASSES)}
    meta["label_idx"] = meta["target_class"].map(class_to_idx)

    print(f"{len(meta)} patients: {meta['target_class'].value_counts().to_dict()}")
    return meta


def annotation_overlap(start_sec, end_sec, events):
    if events.empty:
        return 0, 0
    hit = events[(events["start"] < end_sec) & (events["end"] > start_sec)]
    if hit.empty:
        return 0, 0
    return int(hit["c"].sum() > 0), int(hit["w"].sum() > 0)


def slice_audio(meta):
    files = sorted(f for f in os.listdir(Config.AUDIO_DIR) if f.endswith(".wav"))
    records = []

    for filename in tqdm(files, desc="slicing"):
        try:
            pid = int(filename.split("_")[0])
        except ValueError:
            continue
        if pid not in meta["pid"].values:
            continue

        patient = meta[meta["pid"] == pid].iloc[0]
        path = os.path.join(Config.AUDIO_DIR, filename)
        try:
            audio, _ = librosa.load(path, sr=Config.SAMPLE_RATE)
        except Exception:
            continue

        events = pd.DataFrame()
        txt_path = path.replace(".wav", ".txt")
        if os.path.exists(txt_path):
            try:
                events = pd.read_csv(txt_path, sep="\t", names=["start", "end", "c", "w"])
            except Exception:
                pass

        stem = filename.replace(".wav", "")

        def save(clip, start_idx, c, w):
            name = f"{stem}_{start_idx}.wav"
            sf.write(os.path.join(Config.PROCESSED_DIR, name), clip, Config.SAMPLE_RATE)
            records.append({
                "filename": name,
                "pid": pid,
                "crackles": c,
                "wheezes": w,
                "label_idx": patient["label_idx"],
                "disease": patient["target_class"],
            })

        if len(audio) < Config.TARGET_LEN:
            clip = np.pad(audio, (0, Config.TARGET_LEN - len(audio)), mode="constant")
            save(clip, 0, *annotation_overlap(0, Config.DURATION, events))
        else:
            for start in range(0, len(audio) - Config.TARGET_LEN + 1, Config.STEP_SIZE):
                end = start + Config.TARGET_LEN
                c, w = annotation_overlap(start / Config.SAMPLE_RATE, end / Config.SAMPLE_RATE, events)
                save(audio[start:end], start, c, w)

    return pd.DataFrame(records)


def assign_folds(records, meta):
    df = pd.merge(records, meta[["pid", "age", "sex"]], on="pid", how="left")
    patients = meta[["pid", "label_idx"]].drop_duplicates("pid").reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
    fold_by_pid = {}
    for fold, (_, idx) in enumerate(splitter.split(np.zeros(len(patients)), patients["label_idx"])):
        for pid in patients.iloc[idx]["pid"]:
            fold_by_pid[int(pid)] = fold
    df["fold"] = df["pid"].map(fold_by_pid).astype(int)
    df["cohort"] = "ICHBI"
    df["split_role"] = "development"
    if not (df.groupby("pid")["fold"].nunique() == 1).all():
        raise RuntimeError("patient leakage across ICBHI folds")

    df.to_csv(Config.FULL_CSV, index=False)
    print(f"wrote {Config.FULL_CSV}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    Config.set_seed(Config.SEED)
    Config.check_audio_paths()
    meta = load_metadata()
    assign_folds(slice_audio(meta), meta)
