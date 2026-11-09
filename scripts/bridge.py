"""Build distillation targets by matching audio files to privileged subjects.

The cohorts are unpaired. Each Healthy/COPD audio file is matched to privileged
subjects with the same label, the same sex and a close age, and their teacher
representations are averaged into that file's target. Other-class files get none.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import torch

from prd.config import Config, PRIVILEGED_VIEWS
from prd.models import TeacherModel
from prd.privileged import TeacherPreprocessor, load_view
from prd.utils import method_run_dir, save_json

SEX_KEYS = ("gender", "Gender", "sex", "Sex", "GENDER", "SEX")
AGE_KEYS = ("age_visit", "age_baseline", "Age_P2", "Age_P1", "age", "Age", "AGE")


def normalize_sex(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return None
    if "f" in text or "2" in text:
        return "Female"
    if "m" in text or "1" in text:
        return "Male"
    return None


def _first_present(row, keys):
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return None


def _to_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _load_teacher(view, seed: int, device, random_init: bool = False):
    """Rebuild the fitted teacher and preprocessor; random_init skips the trained weights."""
    run = method_run_dir("teachers", f"{view.name}/seed{seed}")
    metrics = json.loads((run / "metrics" / "teacher_metrics.json").read_text(encoding="utf-8"))
    input_dim = int(metrics["input_dim"])

    torch.manual_seed(seed)
    model = TeacherModel(input_dim=input_dim, shared_dim=Config.SHARED_DIM, num_classes=2).to(device)
    if random_init:
        print(f"[bridge:{view.name} seed{seed}] random-init teacher, trained weights not loaded")
    else:
        state = torch.load(run / "checkpoints" / "teacher_model.pth", map_location=device)
        model.load_state_dict(state)
    model.eval()

    pre_state = np.load(run / "checkpoints" / "preprocessor.npz", allow_pickle=True)
    pre = TeacherPreprocessor.from_state(pre_state)
    return model, pre


def _build_teacher_db(model, pre, view, device) -> pd.DataFrame:
    """Teacher representation plus label, sex and age for every privileged subject."""
    lv = load_view(view, sharp=True)
    x_priv = pre.transform(lv.features)
    with torch.no_grad():
        reps = model.encoder(torch.from_numpy(x_priv).to(device)).cpu().numpy().astype(np.float32)

    labels = lv.labels.to_numpy().astype(int)
    clinical = lv.clinical

    records = []
    for pos, sid in enumerate(lv.ids):
        row = clinical.loc[sid] if sid in clinical.index else pd.Series(dtype=object)
        sex = normalize_sex(_first_present(row, SEX_KEYS))
        age = _to_float(_first_present(row, AGE_KEYS))
        if sex is None or age is None:
            continue
        records.append({"rep": reps[pos], "label": int(labels[pos]), "sex": sex, "age": age})

    db = pd.DataFrame.from_records(records)
    print(f"[bridge:{view.name}] teacher DB: {len(db)} privileged profiles with sex+age.")
    if db.empty:
        available = sorted(clinical.columns.tolist())
        raise ValueError(
            f"No privileged profiles had usable sex+age for view '{view.name}'. "
            f"The clinical file {view.clinical_csv} must expose a sex column "
            f"({SEX_KEYS}) and an age column ({AGE_KEYS}). Columns present: {available[:40]}"
        )
    return db


def run_bridge(view_name: str, seed: int, age_tolerance: float | None = None,
               random_teacher: bool = False) -> None:
    Config.set_seed(seed)
    device = Config.DEVICE
    view = PRIVILEGED_VIEWS[view_name]
    age_tolerance = Config.AGE_TOLERANCE if age_tolerance is None else float(age_tolerance)

    model, pre = _load_teacher(view, seed, device, random_init=random_teacher)
    db = _build_teacher_db(model, pre, view, device)

    full_df = pd.read_csv(Config.FULL_CSV)
    label_for_disease = {"Healthy": 0, "COPD": 1}

    mapping: dict[str, np.ndarray] = {}
    matched = {0: 0, 1: 0}
    unmatched = 0
    skipped_other = 0
    relaxed = 0

    for _, row in full_df.iterrows():
        disease = str(row.get("disease"))
        if disease not in label_for_disease:
            skipped_other += 1
            continue

        target_label = label_for_disease[disease]
        audio_sex = normalize_sex(row.get("sex"))
        audio_age = _to_float(row.get("age"))
        if audio_sex is None or audio_age is None:
            unmatched += 1
            continue

        # the label always has to match; age and then sex are relaxed when the
        # cohorts don't overlap (ICBHI has young healthy subjects, COPDGene does not)
        same_label = db[db["label"] == target_label]
        candidates = same_label[
            (same_label["sex"] == audio_sex)
            & same_label["age"].between(audio_age - age_tolerance, audio_age + age_tolerance)
        ]
        if len(candidates) == 0:
            candidates = same_label[same_label["sex"] == audio_sex]
            relaxed += 1
        if len(candidates) == 0:
            candidates = same_label
        if len(candidates) == 0:
            unmatched += 1
            continue

        target = np.mean(np.stack(candidates["rep"].tolist()), axis=0).astype(np.float32)
        mapping[str(row["filename"])] = target
        matched[target_label] += 1

    out_dir = Config.PROJECT_ROOT / "data" / "distillation"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"age{int(age_tolerance)}"
    if random_teacher:
        # seed-tagged only, so the trained-teacher targets are never overwritten
        names = [f"{view.name}_targets_randinit_{tag}_seed{seed}.npy"]
        audit_names = [f"{view.name}_bridge_audit_randinit_{tag}_seed{seed}.json"]
    else:
        names = [f"{view.name}_targets_{tag}_seed{seed}.npy", f"{view.name}_targets_{tag}.npy"]
        audit_names = [f"{view.name}_bridge_audit_{tag}_seed{seed}.json"]
        if age_tolerance == Config.AGE_TOLERANCE:
            names.append(f"{view.name}_targets.npy")
            audit_names.append(f"{view.name}_bridge_audit.json")
    for name in names:
        np.save(out_dir / name, mapping, allow_pickle=True)

    audit = {
        "method": "clinical value matching with averaged teacher 512-dim reps",
        "random_teacher": bool(random_teacher),
        "view": view.name,
        "seed": seed,
        "matching_rule": f"same binary label AND same sex AND age within +/-{age_tolerance:g} years",
        "pairing": "no patient pairing across cohorts",
        "age_tolerance_years": age_tolerance,
        "audio_rows_total": int(len(full_df)),
        "teacher_profiles": int(len(db)),
        "matched_total": int(matched[0] + matched[1]),
        "matched_healthy": int(matched[0]),
        "matched_copd": int(matched[1]),
        "matched_with_relaxed_age": int(relaxed),
        "unmatched_eligible": int(unmatched),
        "skipped_other": int(skipped_other),
    }
    for name in audit_names:
        save_json(audit, out_dir / name)

    total_matched = matched[0] + matched[1]
    print(
        f"[bridge:{view.name} seed{seed} {tag}] matched {total_matched} audio files "
        f"(Healthy={matched[0]}, COPD={matched[1]}); unmatched={unmatched}, other={skipped_other}."
    )
    print(f"[bridge:{view.name} seed{seed}] targets -> {out_dir / names[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audio->teacher distillation targets by value matching.")
    parser.add_argument("--view", required=True, choices=sorted(PRIVILEGED_VIEWS.keys()))
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument("--age_tolerance", type=float, default=Config.AGE_TOLERANCE,
                        help="age window in years for matching")
    parser.add_argument("--random_teacher", action="store_true",
                        help="use an untrained teacher encoder (control)")
    args = parser.parse_args()
    run_bridge(args.view, args.seed, args.age_tolerance, args.random_teacher)


if __name__ == "__main__":
    main()
