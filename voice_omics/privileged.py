"""Loading and preprocessing for the privileged tabular views (omics, CT-clinical).

Shared by the teacher (fits the preprocessor), the contrastive anchor, and the
bridge (both reload the fitted state). Keeps the leakage boundary in one place:
feature selection / imputation / scaling are fit on training rows only.

Sharp-class option: keep GOLD 0 as Healthy and GOLD >=3 as severe COPD, drop
GOLD 1&2. Falls back to the binary label (with a printed warning) when the GOLD
column is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

from config import PrivilegedView, Sharp


def _map_binary(value) -> int:
    text = str(value).strip().casefold()
    if text in {"healthy", "normal", "control", "n"}:
        return 0
    if text in {"copd", "case"}:
        return 1
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return -1
    if not np.isfinite(numeric) or numeric < 0:
        return -1
    return 0 if numeric == 0 else 1


def _read_indexed(path, index_col) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={index_col: str}, low_memory=False)
    if index_col not in frame.columns:
        raise ValueError(f"{path} has no index column {index_col!r}")
    frame[index_col] = frame[index_col].astype(str)
    return frame.set_index(index_col)


@dataclass
class LoadedView:
    features: pd.DataFrame     # numeric, indexed by sid
    labels: pd.Series          # binary 0/1
    clinical: pd.DataFrame     # for prompts; includes a resolved ``finalGold`` column
    ids: np.ndarray


def load_view(view: PrivilegedView, *, sharp: bool = True) -> LoadedView:
    features = _read_indexed(view.x_csv, view.index_col)
    features.columns = features.columns.astype(str)
    y_frame = _read_indexed(view.y_csv, view.index_col)
    label_col = view.label_col if view.label_col in y_frame.columns else list(y_frame.columns)[0]
    binary = y_frame[label_col].map(_map_binary)

    clinical = (
        _read_indexed(view.clinical_csv, view.index_col)
        if Path(view.clinical_csv).exists()
        else pd.DataFrame(index=features.index)
    )

    # Resolve GOLD stage from y or clinical for sharp filtering / prompts.
    gold = None
    for source in (y_frame, clinical):
        if view.gold_col in source.columns:
            gold = pd.to_numeric(source[view.gold_col], errors="coerce")
            break

    common = features.index.intersection(binary.index)
    features, binary = features.loc[common], binary.loc[common]
    valid = binary != -1
    features, binary = features.loc[valid], binary.loc[valid].astype(int)

    if sharp and Sharp.ENABLE and gold is not None:
        gold = gold.reindex(features.index)
        keep_healthy = gold.isin(Sharp.HEALTHY_GOLD)
        keep_copd = gold.isin(Sharp.COPD_GOLD)
        keep = keep_healthy | keep_copd
        features = features.loc[keep]
        binary = pd.Series(np.where(keep_copd.loc[keep], 1, 0), index=features.index)
    elif sharp and Sharp.ENABLE:
        print(f"[privileged] WARNING: no GOLD column {view.gold_col!r} for view "
              f"{view.name!r}; using binary labels instead of sharp classes.")

    if set(binary.unique()) != {0, 1}:
        raise ValueError(f"View {view.name} needs both classes; found {sorted(binary.unique())}")

    clinical = clinical.reindex(features.index).copy()
    clinical["finalGold"] = (gold.reindex(features.index) if gold is not None else np.nan)
    return LoadedView(features=features, labels=binary, clinical=clinical, ids=features.index.to_numpy(str))


class TeacherPreprocessor:
    """Fit on training rows: drop high-missing / zero-variance columns, ANOVA top-k,
    then apply signed-log1p and standardization. Reused at inference via ``state``."""

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.selected: list[str] = []
        self.medians: pd.Series | None = None
        self.scaler: StandardScaler | None = None

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> "TeacherPreprocessor":
        numeric = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        retained = numeric.columns[numeric.isna().mean() <= 0.20]
        medians = numeric[retained].median(axis=0).fillna(0.0)
        frame = numeric[retained].fillna(medians)
        retained = frame.columns[frame.var(axis=0, ddof=0) > 1e-12]
        frame = frame[retained]
        scores, _ = f_classif(frame.to_numpy(dtype=float), np.asarray(labels))
        ranking = pd.DataFrame(
            {"feature": frame.columns.astype(str), "anova_f": np.nan_to_num(scores, nan=-np.inf)}
        ).sort_values(["anova_f", "feature"], ascending=[False, True])
        self.selected = ranking.head(min(self.max_features, len(ranking)))["feature"].tolist()
        if not self.selected:
            raise ValueError("No features survived teacher preprocessing")
        self.medians = medians.loc[self.selected]
        self.scaler = StandardScaler().fit(self._signed_log1p(features))
        return self

    def _signed_log1p(self, features: pd.DataFrame) -> np.ndarray:
        numeric = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = numeric.reindex(columns=self.selected).fillna(self.medians).to_numpy(dtype=np.float64)
        return np.sign(values) * np.log1p(np.abs(values))

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        assert self.scaler is not None, "TeacherPreprocessor must be fit first"
        return self.scaler.transform(self._signed_log1p(features)).astype(np.float32)

    def to_state(self) -> dict:
        return {
            "selected_features": list(self.selected),
            "imputation_median": np.asarray(self.medians.to_numpy(), dtype=np.float64),
            "mean": self.scaler.mean_,
            "scale": self.scaler.scale_,
        }

    @classmethod
    def from_state(cls, state) -> "TeacherPreprocessor":
        obj = cls()
        obj.selected = [str(f) for f in state["selected_features"]]
        obj.medians = pd.Series(np.asarray(state["imputation_median"], dtype=float), index=obj.selected)
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(state["mean"], dtype=float)
        scaler.scale_ = np.asarray(state["scale"], dtype=float)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(obj.selected)
        obj.scaler = scaler
        return obj
