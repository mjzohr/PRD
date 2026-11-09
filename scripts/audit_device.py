"""Quantify the ICBHI recording-device / diagnosis confound and check leave-one-device-out feasibility."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from prd.config import Config
from prd.utils import save_json


def device_of(filename: str) -> str:
    # patient_index_location_mode_DEVICE_<offset>.wav
    stem = re.sub(r"_\d+\.wav$", "", str(filename))
    parts = stem.split("_")
    return parts[4] if len(parts) > 4 else "unknown"


def main() -> None:
    frame = pd.read_csv(Config.FULL_CSV)
    frame["device"] = frame["filename"].map(device_of)
    patients = frame.drop_duplicates("pid")
    table = pd.crosstab(patients["device"], patients["disease"])
    print("patients by device x diagnosis:\n", table, "\n")

    # best device-only rule for COPD vs rest: predict each device's majority
    correct = 0
    for device, row in table.iterrows():
        copd = int(row.get("COPD", 0))
        other = int(row.sum() - copd)
        correct += max(copd, other)
    device_only_accuracy = correct / len(patients)

    lodo = {}
    for device in table.index:
        held = table.loc[device]
        rest = table.drop(index=device).sum()
        lodo[str(device)] = {
            "test_patients": int(held.sum()),
            "test_classes": int((held > 0).sum()),
            "train_classes": int((rest > 0).sum()),
            "viable": bool((held > 0).sum() >= 2 and (rest > 0).sum() == len(table.columns)),
        }

    spanning = int((frame.groupby("pid")["device"].nunique() > 1).sum())
    report = {
        "patients": int(len(patients)),
        "devices": {str(k): int(v) for k, v in patients["device"].value_counts().items()},
        "crosstab": {str(d): {str(c): int(n) for c, n in row.items()} for d, row in table.iterrows()},
        "copd_vs_rest_accuracy_from_device_alone": round(device_only_accuracy, 4),
        "patients_spanning_multiple_devices": spanning,
        "leave_one_device_out": lodo,
        "any_viable_lodo_split": any(v["viable"] for v in lodo.values()),
    }
    print(f"COPD-vs-rest accuracy from device alone: {device_only_accuracy:.1%}")
    print(f"patients spanning >1 device: {spanning}")
    print(f"any viable leave-one-device-out split: {report['any_viable_lodo_split']}")
    for device, info in lodo.items():
        print(f"   holdout {device:10s} test_patients={info['test_patients']:3d} "
              f"test_classes={info['test_classes']} train_classes={info['train_classes']} "
              f"viable={info['viable']}")

    out = Config.RESULTS_DIR / "reports" / "device_confound.json"
    save_json(report, out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
