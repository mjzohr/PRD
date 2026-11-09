"""Patient-level nested cross-validation splits.

All slices from a patient stay in one fold; each outer-train partition gives up
a stratified validation set for early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class NestedPatientSplit:
    repeat_seed: int
    outer_fold: int
    train_ids: tuple[Hashable, ...]
    validation_ids: tuple[Hashable, ...]
    test_ids: tuple[Hashable, ...]

    def assignments(self, patient_labels: pd.DataFrame) -> pd.DataFrame:
        roles = {
            **{pid: "train" for pid in self.train_ids},
            **{pid: "validation" for pid in self.validation_ids},
            **{pid: "test" for pid in self.test_ids},
        }
        output = patient_labels.copy()
        output["split_role"] = output["patient_id"].map(roles)
        output["repeat_seed"] = self.repeat_seed
        output["outer_fold"] = self.outer_fold
        return output.sort_values(
            ["split_role", "label_idx", "patient_id"]
        ).reset_index(drop=True)

    def audit(self, patient_labels: pd.DataFrame) -> dict:
        assignments = self.assignments(patient_labels)
        role_sets = {
            role: set(assignments.loc[assignments["split_role"] == role, "patient_id"])
            for role in ("train", "validation", "test")
        }
        overlaps = {
            "train_validation": len(role_sets["train"] & role_sets["validation"]),
            "train_test": len(role_sets["train"] & role_sets["test"]),
            "validation_test": len(role_sets["validation"] & role_sets["test"]),
        }
        return {
            "repeat_seed": self.repeat_seed,
            "outer_fold": self.outer_fold,
            "patient_counts": {
                role: int(len(role_sets[role])) for role in ("train", "validation", "test")
            },
            "patient_overlap_counts": overlaps,
            "leakage_check_passed": all(value == 0 for value in overlaps.values()),
        }


def patient_label_table(
    frame: pd.DataFrame,
    *,
    patient_column: str = "pid",
    label_column: str = "label_idx",
) -> pd.DataFrame:
    missing = sorted({patient_column, label_column} - set(frame.columns))
    if missing:
        raise ValueError(f"Patient split input is missing columns: {missing}")
    labels_per_patient = frame.groupby(patient_column)[label_column].nunique()
    inconsistent = labels_per_patient[labels_per_patient != 1]
    if not inconsistent.empty:
        raise ValueError(
            f"Patients have inconsistent labels: {inconsistent.index.astype(str).tolist()[:10]}"
        )
    output = frame[[patient_column, label_column]].drop_duplicates(patient_column).copy()
    output.columns = ["patient_id", "label_idx"]
    return output.sort_values("patient_id").reset_index(drop=True)


def build_nested_patient_splits(
    frame: pd.DataFrame,
    *,
    folds: int = 5,
    repeat_seed: int = 42,
    validation_fraction: float = 0.15,
    patient_column: str = "pid",
    label_column: str = "label_idx",
) -> tuple[pd.DataFrame, list[NestedPatientSplit]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    patients = patient_label_table(
        frame, patient_column=patient_column, label_column=label_column
    )
    class_counts = patients["label_idx"].value_counts()
    if (class_counts < folds).any():
        too_small = class_counts[class_counts < folds].to_dict()
        raise ValueError(f"Each class needs at least {folds} patients; observed {too_small}")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=repeat_seed)
    patient_ids = patients["patient_id"].to_numpy()
    labels = patients["label_idx"].to_numpy(dtype=int)
    splits = []
    for outer_fold, (outer_train_positions, test_positions) in enumerate(
        splitter.split(patient_ids, labels)
    ):
        outer_ids = patient_ids[outer_train_positions]
        outer_labels = labels[outer_train_positions]
        train_ids, validation_ids = train_test_split(
            outer_ids,
            test_size=validation_fraction,
            random_state=repeat_seed + outer_fold,
            shuffle=True,
            stratify=outer_labels,
        )
        split = NestedPatientSplit(
            repeat_seed=int(repeat_seed),
            outer_fold=int(outer_fold),
            train_ids=tuple(train_ids.tolist()),
            validation_ids=tuple(validation_ids.tolist()),
            test_ids=tuple(patient_ids[test_positions].tolist()),
        )
        if not split.audit(patients)["leakage_check_passed"]:
            raise RuntimeError(f"Patient leakage in repeat {repeat_seed}, fold {outer_fold}")
        splits.append(split)
    return patients, splits


def select_patient_rows(
    frame: pd.DataFrame,
    patient_ids: Sequence[Hashable],
    *,
    patient_column: str = "pid",
) -> pd.DataFrame:
    return frame[frame[patient_column].isin(set(patient_ids))].copy()
