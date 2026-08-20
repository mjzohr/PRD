"""Restrict the CWLSD manifest to the label space the model was trained on.

ICBHI's `Other` class covers respiratory conditions (asthma, bronchiectasis,
bronchiolitis, LRTI, pneumonia, URTI). CWLSD additionally contains heart failure,
lung fibrosis and pleural effusion -- cardiac and interstitial conditions absent
from training. Scoring those patients measures the model on a universe it never
saw, so the matched evaluation keeps only diagnoses represented in ICBHI.

The raw diagnosis is preserved in each slice filename
(``BP100_N,N,P R M,70,F_0.wav``), so this only rewrites the manifest -- no audio
is re-sliced and nothing is retrained.

    python make_matched_manifest.py
"""

from __future__ import annotations

import re

import pandas as pd

from config import Config

SOURCE = Config.PROJECT_ROOT / "data" / "manifests" / "cwlsd_external_slices.csv"
TARGET = Config.PROJECT_ROOT / "data" / "manifests" / "cwlsd_matched_slices.csv"

# Diagnoses whose ICBHI counterpart exists. Everything else is dropped, including
# any comorbid label that mentions an excluded condition (e.g. "Heart Failure + COPD",
# which is ambiguous between two classes).
KEEP = {
    "n": "Healthy", "normal": "Healthy", "healthy": "Healthy",
    "copd": "COPD",
    "asthma": "Other", "pneumonia": "Other", "bron": "Other",
    "bronchiectasis": "Other", "bronchiolitis": "Other", "lrti": "Other", "urti": "Other",
}
EXCLUDE_TOKENS = ("heart failure", "fibrosis", "effusion", "plueral", "pleural")


def raw_diagnosis(filename: str) -> str:
    """Recover the diagnosis field from a slice filename."""
    stem = re.sub(r"_\d+\.wav$", "", str(filename))     # strip the slice offset
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
    # Recompute the class label from the matched mapping (identical in practice,
    # but makes the manifest self-consistent).
    matched["disease"] = mapped.loc[keep].to_numpy()
    matched["label_idx"] = matched["disease"].map(lambda d: Config.CLASSES.index(d))

    print(f"kept   {matched['pid'].nunique():3d} patients / {len(matched):5d} slices")
    print(f"dropped {frame.loc[~keep, 'pid'].nunique():3d} patients:",
          dropped.to_dict())
    print("class balance (patients):",
          matched.drop_duplicates("pid")["disease"].value_counts().to_dict())
    matched.to_csv(TARGET, index=False)
    print(f"saved -> {TARGET}")


if __name__ == "__main__":
    main()
