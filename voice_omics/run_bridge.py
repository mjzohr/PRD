"""Clinical bridge: build audio -> teacher distillation targets (512-dim).

There is NO patient pairing across cohorts. The ICBHI audio subjects and the
COPDGene privileged subjects are unpaired, so we match by clinical *values*:
for each Healthy/COPD audio file we pull the privileged subjects with the same
binary label, same sex, and age within +/-5 years, and average their teacher
512-dim representations into that file's distillation target. ``Other`` audio
rows have no teacher and are skipped entirely.

Run (after training the matching teacher):
    python run_bridge.py --view omics --seed 42
    python run_bridge.py --view ct_clinical
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from config import Config, PRIVILEGED_VIEWS
from models import TeacherModel
from privileged import TeacherPreprocessor, load_view
from utils import method_run_dir, save_json

SEX_KEYS = ("gender", "Gender", "sex", "Sex", "GENDER", "SEX")
AGE_KEYS = ("age_visit", "age_baseline", "Age_P2", "Age_P1", "age", "Age", "AGE")


def normalize_sex(value):
    """Map assorted sex encodings (F/M, 1/2, strings) to 'Female'/'Male'."""
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
    """Rebuild the fitted teacher and preprocessor for (view, seed).

    With ``random_init`` the encoder is left at its initialisation instead of
    loading trained weights. Architecture, preprocessor and every downstream
    step are unchanged, so this isolates the effect of teacher training.
    """
    run = method_run_dir("teachers", f"{view.name}/seed{seed}")
    metrics = json.loads((run / "metrics" / "teacher_metrics.json").read_text(encoding="utf-8"))
    input_dim = int(metrics["input_dim"])

    torch.manual_seed(seed)          # reproducible, and different per seed
    model = TeacherModel(input_dim=input_dim, shared_dim=Config.SHARED_DIM, num_classes=2).to(device)
    if random_init:
        print(f"[bridge:{view.name} seed{seed}] RANDOM-INIT teacher: "
              f"trained weights deliberately NOT loaded.")
    else:
        state = torch.load(run / "checkpoints" / "teacher_model.pth", map_location=device)
        model.load_state_dict(state)
    model.eval()

    pre_state = np.load(run / "checkpoints" / "preprocessor.npz", allow_pickle=True)
    pre = TeacherPreprocessor.from_state(pre_state)
    return model, pre


def _build_teacher_db(model, pre, view, device) -> pd.DataFrame:
    """Encode every privileged subject to its 512-dim teacher representation and
    attach label / sex / age for value matching."""
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

    # 1-2. Load teacher + preprocessor, build the value-matching database.
    model, pre = _load_teacher(view, seed, device, random_init=random_teacher)
    db = _build_teacher_db(model, pre, view, device)

    # 3. Match each Healthy/COPD audio file against the teacher DB by value.
    full_df = pd.read_csv(Config.FULL_CSV)
    label_for_disease = {"Healthy": 0, "COPD": 1}

    mapping: dict[str, np.ndarray] = {}
    matched = {0: 0, 1: 0}
    unmatched = 0
    skipped_other = 0
    relaxed = 0

    for _, row in full_df.iterrows():
        disease = str(row.get("disease"))
        if disease not in label_for_disease:  # 'Other' rows have no teacher.
            skipped_other += 1
            continue

        target_label = label_for_disease[disease]
        audio_sex = normalize_sex(row.get("sex"))
        audio_age = _to_float(row.get("age"))
        if audio_sex is None or audio_age is None:
            unmatched += 1
            continue

        # Same disease label always; then prefer sex+age, relaxing age (and finally
        # sex) if the demographics don't overlap across cohorts. Cross-cohort age
        # ranges differ (ICBHI includes young/healthy patients; COPDGene is older),
        # so a strict age match alone would drop an entire class.
        same_label = db[db["label"] == target_label]
        candidates = same_label[
            (same_label["sex"] == audio_sex)
            & same_label["age"].between(audio_age - age_tolerance, audio_age + age_tolerance)
        ]
        if len(candidates) == 0:
            candidates = same_label[same_label["sex"] == audio_sex]  # relax age
            relaxed += 1
        if len(candidates) == 0:
            candidates = same_label  # relax sex too; keep disease label
        if len(candidates) == 0:
            unmatched += 1
            continue

        target = np.mean(np.stack(candidates["rep"].tolist()), axis=0).astype(np.float32)
        mapping[str(row["filename"])] = target
        matched[target_label] += 1

    # 4. Persist the targets and an audit trail. Tag by age tolerance so a
    #    sensitivity sweep keeps its files separate; also write the canonical
    #    name at the default tolerance so the main pipeline is unaffected.
    out_dir = Config.PROJECT_ROOT / "data" / "distillation"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"age{int(age_tolerance)}"
    if random_teacher:
        # Seed-tagged name only. Writing the shared or canonical files here would
        # overwrite the real targets that every other run reads.
        names = [f"{view.name}_targets_randinit_{tag}_seed{seed}.npy"]
        audit_names = [f"{view.name}_bridge_audit_randinit_{tag}_seed{seed}.json"]
    else:
        # Seed-tagged file is primary (each audio seed pairs with its own teacher, and
        # parallel seed shards never collide). Non-seed names are kept for compatibility.
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
                        help="Match audio to privileged subjects within +/- this many years.")
    parser.add_argument("--random_teacher", action="store_true",
                        help="Build targets from an UNTRAINED teacher encoder. Everything "
                             "else -- privileged inputs, preprocessor, value matching, group "
                             "averaging -- is identical, so this isolates whether teacher "
                             "training is what the student benefits from.")
    args = parser.parse_args()
    run_bridge(args.view, args.seed, args.age_tolerance, args.random_teacher)


if __name__ == "__main__":
    main()
