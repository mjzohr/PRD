"""Audio-only baseline with the same folds and protocol as the distilled student."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prd.config import Config
from prd.data import AudioClassifierDataset
from prd.models import ClassifierHead, get_audio_encoder
from prd.splitting import build_nested_patient_splits, select_patient_rows
from prd.utils import (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio-only baseline (train from scratch).")
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument(
        "--encoder",
        type=str,
        default=Config.AUDIO_ENCODER,
        choices=["CNN", "CRNN", "TRANSFORMER"],
    )
    return parser.parse_args()


def make_loader(frame: pd.DataFrame, *, seed: int, fold: int, train: bool) -> DataLoader:
    dataset = AudioClassifierDataset(frame)
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


def train_one_fold(train_frame, seed, fold, encoder_name, checkpoint_path, val_loader):
    """Train from scratch on one outer fold, early-stopping on validation loss."""
    model = ClassifierHead(get_audio_encoder(encoder_name), Config.NUM_CLASSES).to(Config.DEVICE)
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": Config.BASELINE_ENCODER_LR},
            {"params": model.head.parameters(), "lr": Config.HEAD_LR},
        ]
    )
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, path=str(checkpoint_path), maximize=False,
        min_epochs=Config.MIN_EPOCHS,
    )
    train_loader = make_loader(train_frame, seed=seed, fold=fold, train=True)

    best_epoch, best_score, best_loss = -1, 0.0, np.inf
    for epoch in range(Config.FINETUNE_EPOCHS):
        model.train()
        total, batches = 0.0, 0
        for spec, label, _ in train_loader:
            spec, label = spec.to(Config.DEVICE), label.to(Config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(spec), label)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1

        val_loss = evaluate_task_loss(model, val_loader)
        truth, pred, _, _ = evaluate_patient_level(model, val_loader)
        score = macro_f1(truth, pred)
        print(f"   [baseline fold {fold}] epoch {epoch + 1:02d} | "
              f"train loss {total / max(batches, 1):.4f} | val loss {val_loss:.4f} "
              f"| val macro-F1 {score:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_score, best_epoch = val_loss, score, epoch + 1
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    return model, best_epoch, best_score


def main() -> None:
    args = parse_args()
    Config.set_seed(args.seed)

    run = method_run_dir("baseline_audio", f"{args.encoder}/seed{args.seed}")
    save_json(
        {
            "method": "baseline_audio",
            "encoder": args.encoder,
            "seed": args.seed,
            "num_folds": Config.NUM_FOLDS,
            "inner_validation_fraction": Config.INNER_VALIDATION_FRACTION,
            "epochs": Config.FINETUNE_EPOCHS,
            "patience": Config.PATIENCE,
            "batch_size": Config.BATCH_SIZE,
            "encoder_lr": Config.BASELINE_ENCODER_LR,
            "head_lr": Config.HEAD_LR,
            "trained_from_scratch": True,
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

        val_loader = make_loader(val_frame, seed=args.seed, fold=fold, train=False)
        test_loader = make_loader(test_frame, seed=args.seed, fold=fold, train=False)

        checkpoint_path = run / "checkpoints" / f"fold_{fold}.pth"
        model, best_epoch, best_score = train_one_fold(
            train_frame, args.seed, fold, args.encoder, checkpoint_path, val_loader
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
    print(f"Baseline audio outputs saved under: {run}")


if __name__ == "__main__":
    main()
