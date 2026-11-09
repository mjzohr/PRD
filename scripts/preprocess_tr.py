"""Slice the RespiratoryDatabase@TR cohort (GOLD-graded, external test only) and build its manifest."""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from prd.config import Config

TR_ROOT = Config.PROJECT_ROOT / "data" / "RespiratoryDatabase@TR"
AUDIO_DIR = TR_ROOT / "RespiratoryDatabase@TR"
LABELS_XLSX = TR_ROOT / "Labels.xlsx"
PROCESSED_DIR = Config.PROJECT_ROOT / "data" / "processed_audio" / "tr"
MANIFEST = Config.PROJECT_ROOT / "data" / "manifests" / "tr_external_slices.csv"
PID_OFFSET = 20000  # keeps patient ids distinct from other cohorts


def read_labels() -> dict[str, int]:
    # parse the xlsx xml directly to avoid an Excel dependency
    archive = zipfile.ZipFile(LABELS_XLSX)
    shared = re.findall(
        r"<t[^>]*>(.*?)</t>", archive.read("xl/sharedStrings.xml").decode("utf8", "ignore")
    )
    xml = archive.read("xl/worksheets/sheet1.xml").decode("utf8", "ignore")
    labels: dict[str, int] = {}
    for row in re.findall(r"<row[^>]*>(.*?)</row>", xml, re.S):
        values = []
        for cell in re.findall(r"<c[^>]*?(?:\s|>)(?:.*?)</c>|<c[^>]*/>", row, re.S):
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
        stem = path.stem
        parts = stem.split("_")
        if len(parts) != 2 or parts[0] not in labels:
            skipped += 1
            continue
        subject, channel = parts
        gold = labels[subject]
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
                "recording_type": channel[0],
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
    print(f"wrote {MANIFEST}")
