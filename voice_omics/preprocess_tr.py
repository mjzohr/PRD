"""Slice the RespiratoryDatabase@TR cohort and build its manifest.

A second external cohort, complementary to CWLSD: it is COPD-focused and carries
GOLD severity grades (COPD0..COPD4), which enables two evaluations CWLSD cannot
support --
  * sharp binary transfer: COPD0 (healthy) vs COPD3/4 (severe), matching exactly
    the phenotype the privileged teachers were trained on;
  * severity ranking: does the model's COPD score track GOLD grade across all
    patients (Spearman), a far better-powered test than a 6-vs-24 AUROC.

Layout: RespiratoryDatabase@TR/<PID>_<L|R><1..6>.wav, 12 auscultation channels
per patient at 4 kHz. Labels.xlsx maps patient id -> COPD grade.

    python preprocess_tr.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from config import Config

TR_ROOT = Config.PROJECT_ROOT / "data" / "RespiratoryDatabase@TR"
AUDIO_DIR = TR_ROOT / "RespiratoryDatabase@TR"
LABELS_XLSX = TR_ROOT / "Labels.xlsx"
PROCESSED_DIR = Config.PROJECT_ROOT / "data" / "processed_audio" / "tr"
MANIFEST = Config.PROJECT_ROOT / "data" / "manifests" / "tr_external_slices.csv"
PID_OFFSET = 20000          # keeps patient ids distinct from other cohorts


def read_labels() -> dict[str, int]:
    """Parse Labels.xlsx without an Excel dependency -> {patient_id: gold}."""
    archive = zipfile.ZipFile(LABELS_XLSX)
    shared = re.findall(
        r"<t[^>]*>(.*?)</t>", archive.read("xl/sharedStrings.xml").decode("utf8", "ignore")
    )
    xml = archive.read("xl/worksheets/sheet1.xml").decode("utf8", "ignore")
    labels: dict[str, int] = {}
    for row in re.findall(r"<row[^>]*>(.*?)</row>", xml, re.S):
        values = []
        for cell in re.findall(r"<c[^>]*?(?:\s|>)(?:.*?)</c>|<c[^>]*/>", cell_block := row, re.S):
            match = re.search(r"<v>(.*?)</v>", cell)
            if match is None:
                values.append("")
                continue
            values.append(shared[int(match.group(1))] if 't="s"' in cell else match.group(1))
        if len(values) < 2:
            continue
        pid, diagnosis = str(values[0]).strip(), str(values[1]).strip().upper()
        grade = re.fullmatch(r"COPD([0-4])", diagnosis)
        if grade:
            labels[pid] = int(grade.group(1))
    return labels


def process() -> pd.DataFrame:
    labels = read_labels()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records, skipped = [], 0
    for path in sorted(AUDIO_DIR.glob("*.wav")):
        stem = path.stem                       # e.g. H002_L1
        parts = stem.split("_")
        if len(parts) != 2 or parts[0] not in labels:
            skipped += 1
            continue
        subject, channel = parts
        gold = labels[subject]
        # ICBHI label space: COPD0 -> Healthy, any COPD grade -> COPD.
        disease = "Healthy" if gold == 0 else "COPD"
        label_idx = Config.CLASSES.index(disease)
        pid = PID_OFFSET + int(re.sub(r"\D", "", subject))

        audio, _ = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
        total = len(audio)
        starts = ([0] if total < Config.TARGET_LEN
                  else list(range(0, total - Config.TARGET_LEN + 1, Config.STEP_SIZE)))
        for start in starts:
            chunk = audio[start:start + Config.TARGET_LEN]
            if len(chunk) < Config.TARGET_LEN:
                chunk = np.pad(chunk, (0, Config.TARGET_LEN - len(chunk)))
            name = f"{stem}_{start}.wav"
            sf.write(PROCESSED_DIR / name, chunk, Config.SAMPLE_RATE)
            records.append({
                "filename": name, "pid": pid, "subject": subject,
                "label_idx": label_idx, "disease": disease, "gold": gold,
                "channel": channel, "age": np.nan, "sex": "",
                "crackles": 0, "wheezes": 0, "fold": -1,
                "cohort": "TR", "split_role": "external_test",
                "recording_type": channel[0],      # L or R
            })
    frame = pd.DataFrame.from_records(records)
    print(f"subjects={frame['pid'].nunique()} files={len(set(f[:-4] for f in frame.filename))} "
          f"slices={len(frame)} skipped={skipped}")
    print("patients per grade:",
          frame.drop_duplicates("pid")["gold"].value_counts().sort_index().to_dict())
    print("patients per class:",
          frame.drop_duplicates("pid")["disease"].value_counts().to_dict())
    return frame


if __name__ == "__main__":
    Config.set_seed(Config.SEED)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    table = process()
    assert (table["split_role"] == "external_test").all()
    table.to_csv(MANIFEST, index=False)
    print(f"saved manifest -> {MANIFEST}")
