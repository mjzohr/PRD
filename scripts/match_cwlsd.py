"""Restrict the CWLSD manifest to diagnoses represented in ICBHI (drops heart failure, fibrosis, effusion)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from prd.config import Config

SOURCE = Config.PROJECT_ROOT / "data" / "manifests" / "cwlsd_external_slices.csv"
TARGET = Config.PROJECT_ROOT / "data" / "manifests" / "cwlsd_matched_slices.csv"

KEEP = {
    "n": "Healthy", "normal": "Healthy", "healthy": "Healthy",
    "copd": "COPD",
    "asthma": "Other", "pneumonia": "Other", "bron": "Other",
    "bronchiectasis": "Other", "bronchiolitis": "Other", "lrti": "Other", "urti": "Other",
}
# comorbid labels such as "Heart Failure + COPD" are dropped too: they are
# ambiguous between two classes
EXCLUDE_TOKENS = ("heart failure", "fibrosis", "effusion", "plueral", "pleural")


def raw_diagnosis(filename: str) -> str:
    stem = re.sub(r"_\d+\.wav$", "", str(filename))
    rest = stem.split("_", 1)[1] if "_" in stem else stem
    return rest.split(",")[0].strip().lower()


def main() -> None:
    frame = pd.read_csv(SOURCE)
    frame["raw_diagnosis"] = frame["filename"].map(raw_diagnosis)

    excluded = frame["raw_diagnosis"].apply(
        lambda d: any(token in d for token in EXCLUDE_TOKENS)
    )
    mapped = frame["raw_diagnosis"].map(KEEP)
    keep = (~excluded) & mapped.notna()

    dropped = frame.loc[~keep].drop_duplicates("pid")["raw_diagnosis"].value_counts()
    matched = frame.loc[keep].copy()
    matched["disease"] = mapped.loc[keep].to_numpy()
    matched["label_idx"] = matched["disease"].map(lambda d: Config.CLASSES.index(d))

    print(f"kept   {matched['pid'].nunique():3d} patients / {len(matched):5d} slices")
    print(f"dropped {frame.loc[~keep, 'pid'].nunique():3d} patients:",
          dropped.to_dict())
    print("class balance (patients):",
          matched.drop_duplicates("pid")["disease"].value_counts().to_dict())
    matched.to_csv(TARGET, index=False)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
