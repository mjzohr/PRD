"""Masked feature distillation: the audio student borrows a privileged teacher
target (omics or CT-clinical) via a cosine pull on matched Healthy/COPD samples,
while classifying the full 3-class task. Same repeated nested patient k-fold and
patient-level protocol as the baseline, so the two are directly comparable.

Run, e.g.::

    python train_distilled.py --seed 42 --encoder CNN --view omics --distill_weight 2.0

The teacher targets are produced by ``run_bridge.py`` and expected at
``data/distillation/<view>_targets.npy``. Outputs land under
``results/distilled/<view>/<encoder>/w<weight>/seed<seed>[/abl_<ablation>]/``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, Hparams
from dataset import AudioClassifierDataset, DistilledAudioDataset
from losses import masked_distillation_loss
from models import ClassifierHead, DistillationProjector, get_audio_encoder
from splitting import build_nested_patient_splits, select_patient_rows
from utils import (
    EarlyStopping,
    bootstrap_ci,
    evaluate_patient_level,
    evaluate_task_loss,
    fold_metrics,
    macro_f1,
    method_run_dir,
    patient_weighted_sampler,
    prediction_frame,
    save_classification_report,
    save_json,
)

# Signal-specificity controls, ordered by how much of the target they remove.
# audit_shuffle.py imports this list, so the two stay in step.
ABLATIONS = [
    "none",               # full bridge: label + sex + age
    "class_prototype",    # label-only groups
    "shuffled_targets",   # within-class permutation
    "random_prototype",   # one random norm-matched vector per class
    "crossed_targets",    # real teacher targets drawn from the opposite class
    "global_prototype",   # a single target for every file, no class dependence
    "random_teacher",     # same bridge, same matching, UNTRAINED teacher encoder
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked feature distillation student.")
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument(
        "--encoder",
        type=str,
        default=Config.AUDIO_ENCODER,
        choices=["CNN", "CRNN", "TRANSFORMER"],
    )
    parser.add_argument("--view", type=str, default="omics", choices=["omics", "ct_clinical"])
    parser.add_argument("--distill_weight", type=float, default=Hparams.DISTILL_WEIGHT)
    parser.add_argument(
        "--distill_loss", type=str, default=Hparams.DISTILL_LOSS,
        choices=["cosine", "mse", "smooth_l1", "cosine_mse", "relational"],
        help="cosine = current method; others are additional variants (relational = RKD).",
    )
    parser.add_argument(
        "--age_tolerance", type=float, default=None,
        help="Use bridge targets built at +/- this many years (from run_bridge). "
             "Omit to use the canonical targets file.",
    )
    parser.add_argument(
        "--ablation", type=str, default="none", choices=ABLATIONS,
        help="Controls ordered by what they remove. shuffled_targets = within-class "
             "permutation (removes demographic correspondence); class_prototype = "
             "replace targets by class means (removes all detail beyond class "
             "identity); random_prototype = one arbitrary norm-matched vector per "
             "class (severs the privileged cohort, keeps class-conditional "
             "regression); crossed_targets = real teacher targets drawn from the "
             "OPPOSITE class (keeps teacher geometry and target diversity, destroys "
             "the label-to-target association); global_prototype = one vector for "
             "every file regardless of class (removes class dependence entirely, so "
             "any surviving gain is generic regularization).",
    )
    return parser.parse_args()


def resolve_target_path(view: str, ablation: str, seed: int, age_tolerance=None) -> Path:
    """Return the distillation targets for this run.

    ``age_tolerance`` selects tolerance-tagged targets from run_bridge; omit for
    the canonical file. Ablations write their modified targets to a temporary
    file, leaving the originals untouched.
    """
    base = Config.PROJECT_ROOT / "data" / "distillation"
    if ablation == "random_teacher":
        # A separate targets file from run_bridge.py --random_teacher, not a
        # perturbation of the real one. Returned as-is.
        if age_tolerance is None:
            raise ValueError("--ablation random_teacher requires --age_tolerance "
                             "(targets are written per tolerance and seed).")
        path = base / f"{view}_targets_randinit_age{int(age_tolerance)}_seed{seed}.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing random-init targets: {path}. Build them first with "
                f"python run_bridge.py --view {view} --seed {seed} "
                f"--age_tolerance {int(age_tolerance)} --random_teacher")
        return path
    if age_tolerance is None:
        candidates = [base / f"{view}_targets.npy"]
    else:
        # Prefer the seed-matched targets; fall back to the shared age-tagged file.
        candidates = [
            base / f"{view}_targets_age{int(age_tolerance)}_seed{seed}.npy",
            base / f"{view}_targets_age{int(age_tolerance)}.npy",
        ]
    target_path = next((p for p in candidates if p.is_file()), None)
    if target_path is None:
        raise FileNotFoundError(
            f"Missing distillation targets for view '{view}': tried {[str(p) for p in candidates]}. "
            f"Generate them first with run_bridge.py"
            + ("" if age_tolerance is None else f" --age_tolerance {age_tolerance}") + "."
        )
    if ablation == "none":
        return target_path

    target_map = np.load(target_path, allow_pickle=True).item()
    manifest = pd.read_csv(Config.FULL_CSV).set_index("filename")["disease"]
    keys = [k for k in target_map if k in manifest.index]
    classes = manifest.loc[keys].to_numpy()
    rng = np.random.default_rng(seed)
    new_map = dict(target_map)

    if ablation == "shuffled_targets":
        # Permute within each class, breaking the demographic correspondence while
        # holding class identity fixed. A global shuffle would not work: ~96% of
        # targets are COPD, so most files would keep a COPD-like target.
        for label in np.unique(classes):
            group = [k for k, c in zip(keys, classes) if c == label]
            permuted = list(group)
            rng.shuffle(permuted)
            for destination, source in zip(group, permuted):
                new_map[destination] = target_map[source]
    elif ablation == "class_prototype":
        # Replace each target by its class mean, leaving only class identity.
        for label in np.unique(classes):
            group = [k for k, c in zip(keys, classes) if c == label]
            mean = np.mean(np.stack([target_map[k] for k in group]), axis=0)
            for k in group:
                new_map[k] = mean
    elif ablation == "random_prototype":
        # One fixed random vector per class, norm-matched to the real class mean so
        # the loss stays on the same scale. Keeps class-conditional regression but
        # drops every link to the privileged cohort.
        for label in np.unique(classes):
            group = [k for k, c in zip(keys, classes) if c == label]
            mean = np.mean(np.stack([target_map[k] for k in group]), axis=0)
            vector = rng.standard_normal(mean.shape).astype(mean.dtype)
            vector *= float(np.linalg.norm(mean)) / max(float(np.linalg.norm(vector)), 1e-8)
            for k in group:
                new_map[k] = vector
    elif ablation == "crossed_targets":
        # Each file takes a real teacher target from the opposite class. Keeps the
        # teacher geometry and target diversity, inverts the label association.
        # Within-class shuffling cannot do this: within-class spread is only 14%
        # of the between-class distance, so permuted targets still identify the
        # correct class.
        for label in np.unique(classes):
            group = [k for k, c in zip(keys, classes) if c == label]
            donors = [k for k, c in zip(keys, classes) if c != label]
            if not donors:                       # single-class pool: nothing to cross
                continue
            picks = rng.integers(0, len(donors), size=len(group))
            for destination, pick in zip(group, picks):
                new_map[destination] = target_map[donors[int(pick)]]
    elif ablation == "global_prototype":
        # A single vector for every file. Unlike the controls above, this removes
        # class-conditional regression as well.
        #
        # Averaged over the class means rather than over files. A file-weighted
        # mean would land on the COPD centroid (96% of files) and barely move
        # those targets, making this indistinguishable from class_prototype.
        centroid = np.mean(
            np.stack([
                np.mean(np.stack([target_map[k] for k, c in zip(keys, classes) if c == label]),
                        axis=0)
                for label in np.unique(classes)
            ]),
            axis=0,
        )
        for k in keys:
            new_map[k] = centroid
    else:
        raise ValueError(f"unknown ablation: {ablation!r}")

    handle = tempfile.NamedTemporaryFile(
        prefix=f"{view}_targets_{ablation}_seed{seed}_", suffix=".npy", delete=False
    )
    handle.close()
    temp_path = Path(handle.name)
    np.save(temp_path, new_map)
    return temp_path


def make_loader(frame, target_path, *, seed, fold, train) -> DataLoader:
    dataset = DistilledAudioDataset(frame, target_path)
    if train:
        generator = Config.get_generator(seed * 1000 + fold)
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=patient_weighted_sampler(frame, generator),
            shuffle=False,
            drop_last=True,
            num_workers=Config.NUM_WORKERS,
            persistent_workers=Config.NUM_WORKERS > 0,
            pin_memory=Config.PIN_MEMORY,
            worker_init_fn=Config.seed_worker,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        persistent_workers=Config.NUM_WORKERS > 0,
        pin_memory=Config.PIN_MEMORY,
        worker_init_fn=Config.seed_worker,
    )


def train_one_fold(train_frame, target_path, args, fold, checkpoint_path, val_loader):
    """Train the student + projector on one outer fold with a ramped distill weight."""
    model = ClassifierHead(get_audio_encoder(args.encoder), Config.NUM_CLASSES).to(Config.DEVICE)
    projector = DistillationProjector(512, 512).to(Config.DEVICE)
    # Optimization must match the audio-only baseline exactly (encoder and head
    # learning rates), so the only difference between the two is the auxiliary
    # objective -- otherwise the comparison confounds transfer with a LR change.
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": Config.BASELINE_ENCODER_LR},
            {"params": model.head.parameters(), "lr": Config.HEAD_LR},
            {"params": projector.parameters(), "lr": Config.HEAD_LR},
        ]
    )
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, path=str(checkpoint_path), maximize=False,
        min_epochs=Config.MIN_EPOCHS,
    )
    train_loader = make_loader(train_frame, target_path, seed=args.seed, fold=fold, train=True)

    best_epoch, best_score, best_loss = -1, 0.0, np.inf
    for epoch in range(Config.FINETUNE_EPOCHS):
        model.train()
        projector.train()
        # Constant weight: the auxiliary objective is at full strength from the
        # first step (no ramp), so model selection cannot land inside a window
        # where the transfer signal is still attenuated.
        current_weight = args.distill_weight
        distill_total, mask_total, n_batches = 0.0, 0.0, 0
        for spec, label, target, mask, _ in train_loader:
            spec, label = spec.to(Config.DEVICE), label.to(Config.DEVICE)
            target, mask = target.to(Config.DEVICE), mask.to(Config.DEVICE)
            optimizer.zero_grad()
            logits, feat = model(spec, return_features=True)
            ce = criterion(logits, label)
            distill = masked_distillation_loss(projector(feat), target, mask, kind=args.distill_loss)
            loss = ce + current_weight * distill
            loss.backward()
            optimizer.step()
            distill_total += float(distill.detach().cpu())
            mask_total += float(mask.float().mean().detach().cpu())
            n_batches += 1
        # Guard: a zero distillation loss / empty mask means the auxiliary signal
        # never reaches the model and the run silently degenerates to the baseline.
        mask_fraction = mask_total / max(n_batches, 1)
        if mask_fraction == 0.0:
            raise RuntimeError(
                f"No training sample had a teacher target (fold {fold}). The targets in "
                f"{target_path} do not match the manifest filenames -- rebuild them with run_bridge.py."
            )

        val_loss = evaluate_task_loss(model, val_loader)
        truth, pred, _, _ = evaluate_patient_level(model, val_loader)
        score = macro_f1(truth, pred)
        print(f"   [distilled fold {fold}] epoch {epoch + 1:02d} | w {current_weight:.2f} "
              f"| distill {distill_total / max(n_batches, 1):.4f} | matched {mask_fraction:.2f} "
              f"| val loss {val_loss:.4f} | val macro-F1 {score:.4f}", flush=True)
        if val_loss < best_loss:                       # selection on validation loss
            best_loss, best_score, best_epoch = val_loss, score, epoch + 1
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    return model, best_epoch, best_score


def main() -> None:
    args = parse_args()
    Config.set_seed(args.seed)
    target_path = resolve_target_path(args.view, args.ablation, args.seed, args.age_tolerance)

    # Study mode (an age tolerance is given) tags runs by tolerance + loss so the
    # sensitivity sweep doesn't collide; the main pipeline (no age tolerance) keeps
    # the original compact tag.
    study_mode = args.age_tolerance is not None
    tag = f"{args.view}/{args.encoder}"
    if study_mode:
        tag += f"/age{int(args.age_tolerance)}/dl_{args.distill_loss}"
    tag += f"/w{args.distill_weight}/seed{args.seed}"
    if args.ablation != "none":
        tag += f"/abl_{args.ablation}"
    run = method_run_dir("distilled", tag)
    save_json(
        {
            "method": "distilled",
            "view": args.view,
            "encoder": args.encoder,
            "seed": args.seed,
            "distill_weight": args.distill_weight,
            "distill_loss": args.distill_loss,
            "age_tolerance": args.age_tolerance,
            "distill_weight_schedule": "constant (no ramp)",
            "model_selection": "inner-validation cross-entropy loss",
            "min_epochs": Config.MIN_EPOCHS,
            "patience": Config.PATIENCE,
            "ablation": args.ablation,
            "target_path": str(target_path),
            "num_folds": Config.NUM_FOLDS,
            "inner_validation_fraction": Config.INNER_VALIDATION_FRACTION,
            "epochs": Config.FINETUNE_EPOCHS,
            "patience": Config.PATIENCE,
            "batch_size": Config.BATCH_SIZE,
            "encoder_lr": 1e-4,
            "projector_lr": Config.HEAD_LR,
            "classes": Config.CLASSES,
        },
        run / "config.json",
    )

    development = pd.read_csv(Config.FULL_CSV)
    patient_labels, splits = build_nested_patient_splits(
        development,
        folds=Config.NUM_FOLDS,
        repeat_seed=args.seed,
        validation_fraction=Config.INNER_VALIDATION_FRACTION,
    )

    external = None
    if Config.EXTERNAL_CSV.exists():
        external = pd.read_csv(Config.EXTERNAL_CSV)

    fold_rows, pooled_frames = [], []
    for split in splits:
        fold = split.outer_fold
        train_frame = select_patient_rows(development, split.train_ids)
        val_frame = select_patient_rows(development, split.validation_ids)
        test_frame = select_patient_rows(development, split.test_ids)
        save_json(split.audit(patient_labels), run / "metrics" / f"split_audit_fold_{fold}.json")

        val_loader = make_loader(val_frame, target_path, seed=args.seed, fold=fold, train=False)
        test_loader = make_loader(test_frame, target_path, seed=args.seed, fold=fold, train=False)

        checkpoint_path = run / "checkpoints" / f"fold_{fold}.pth"
        model, best_epoch, best_score = train_one_fold(
            train_frame, target_path, args, fold, checkpoint_path, val_loader
        )

        truth, pred, probs, ids = evaluate_patient_level(model, test_loader)
        save_classification_report(truth, pred, run / "reports", f"fold_{fold}")
        preds = prediction_frame(truth, pred, probs, ids)
        preds["seed"] = args.seed
        preds["fold"] = fold
        preds.to_csv(run / "predictions" / f"patient_preds_fold_{fold}.csv", index=False)
        pooled_frames.append(preds)

        fold_rows.append(
            {
                "fold": fold,
                "seed": args.seed,
                **fold_metrics(truth, pred),
                "best_epoch": best_epoch,
                "best_val_macro_f1": float(best_score),
            }
        )

        if external is not None:
            ext_loader = DataLoader(
                AudioClassifierDataset(external, processed_dir=Config.EXTERNAL_PROCESSED_DIR),
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
            )
            ext_truth, ext_pred, ext_probs, ext_ids = evaluate_patient_level(model, ext_loader)
            ext_frame = prediction_frame(ext_truth, ext_pred, ext_probs, ext_ids)
            ext_frame["seed"] = args.seed
            ext_frame["fold"] = fold
            ext_frame.to_csv(run / "predictions" / f"cwlsd_preds_fold_{fold}.csv", index=False)

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(run / "metrics" / "fold_metrics.csv", index=False)
    save_json(
        {
            metric: {
                "mean": float(fold_frame[metric].mean()),
                "std": float(fold_frame[metric].std(ddof=0)),
            }
            for metric in ("accuracy", "macro_f1", "weighted_f1")
        },
        run / "metrics" / "summary_metrics.json",
    )

    pooled = pd.concat(pooled_frames, ignore_index=True)
    save_classification_report(pooled["y_true"], pooled["y_pred"], run / "reports", "pooled")
    save_json(bootstrap_ci(pooled), run / "metrics" / "bootstrap_ci.json")
    print(f"Distilled audio outputs saved under: {run}")


if __name__ == "__main__":
    main()
