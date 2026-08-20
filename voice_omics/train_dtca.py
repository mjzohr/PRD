"""Disease-Topic Consensus Alignment (DTCA), fixed so the privileged signal
actually reaches the audio encoder.

A privileged teacher (train_teachers.py) shapes a shared space and exports, per
class, a disease prototype and K clinical-topic prototypes. Here the audio model
classifies THROUGH its shared subspace and is pulled toward those detached
prototypes for Healthy/COPD patients (``Other`` is masked out). The guidance is a
plain ramped weighted term -- no gradient-orthogonalisation, no cosine gate, no
norm cap -- and because classification flows through the guided subspace it
cannot be routed around.

Inference is audio-only.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import Config, Hparams, PRIVILEGED_VIEWS
from dataset import AudioClassifierDataset
from losses import disease_consensus_loss
from models import ConsensusAudioModel, get_audio_encoder
from splitting import build_nested_patient_splits, select_patient_rows
from torch.utils.data import DataLoader
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

ABLATIONS = ("none", "no_disease", "no_topic", "audio_task_only", "shuffled_labels", "random_features")


def _load_prototypes(view_name: str, seed: int, ablation: str):
    path = Config.RESULTS_DIR / "teachers" / f"{view_name}/seed{seed}" / "checkpoints" / "prototypes.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing teacher prototypes: {path}. Run train_teachers.py --view {view_name} --seed {seed} first."
        )
    data = np.load(path)
    class_p = torch.tensor(np.asarray(data["class_prototypes"]), dtype=torch.float32)
    topic_p = torch.tensor(np.asarray(data["topic_prototypes"]), dtype=torch.float32)

    rng = np.random.default_rng(seed)
    if ablation == "shuffled_labels":
        order = rng.permutation(class_p.shape[0])
        class_p, topic_p = class_p[order], topic_p[order]
    elif ablation == "random_features":
        class_p = torch.tensor(rng.standard_normal(class_p.shape), dtype=torch.float32)
        topic_p = torch.tensor(rng.standard_normal(topic_p.shape), dtype=torch.float32)
    return class_p.to(Config.DEVICE), topic_p.to(Config.DEVICE)


def _loader(frame, seed, fold, *, train):
    dataset = AudioClassifierDataset(frame)
    if train:
        generator = Config.get_generator(seed * 1000 + fold)
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=patient_weighted_sampler(frame, generator),
            drop_last=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            worker_init_fn=Config.seed_worker,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )


def train_dtca(seed: int, encoder: str, view_name: str, guidance_weight: float, ablation: str) -> None:
    Config.set_seed(seed)
    view = PRIVILEGED_VIEWS[view_name]
    class_prototypes, topic_prototypes = _load_prototypes(view.name, seed, ablation)
    disable_disease = ablation == "no_disease"
    disable_topic = ablation == "no_topic"
    if ablation == "audio_task_only":
        guidance_weight = 0.0

    tag = f"{view.name}/{encoder}/g{guidance_weight}/seed{seed}"
    if ablation != "none":
        tag += f"/abl_{ablation}"
    run = method_run_dir("dtca", tag)
    save_json(
        {
            "method": "disease_topic_consensus",
            "view": view.name,
            "encoder": encoder,
            "guidance_weight": guidance_weight,
            "ablation": ablation,
            "seed": seed,
            "shared_dim": Config.SHARED_DIM,
            "num_topics": Hparams.NUM_TOPICS,
            "inference": "audio_only",
            "classifier_reads": "shared_subspace_only",
        },
        run / "config.json",
    )

    development = pd.read_csv(Config.FULL_CSV)
    patient_labels, splits = build_nested_patient_splits(
        development, folds=Config.NUM_FOLDS, repeat_seed=seed,
        validation_fraction=Config.INNER_VALIDATION_FRACTION,
    )
    external = pd.read_csv(Config.EXTERNAL_CSV) if Config.EXTERNAL_CSV.exists() else None

    ce_loss = nn.CrossEntropyLoss()
    fold_rows, pooled_predictions = [], []
    for split in splits:
        fold = split.outer_fold
        save_json(split.audit(patient_labels), run / "metrics" / f"split_audit_fold_{fold}.json")
        train_frame = select_patient_rows(development, split.train_ids)
        val_frame = select_patient_rows(development, split.validation_ids)
        test_frame = select_patient_rows(development, split.test_ids)
        train_loader = _loader(train_frame, seed, fold, train=True)
        val_loader = _loader(val_frame, seed, fold, train=False)
        test_loader = _loader(test_frame, seed, fold, train=False)

        # DTCA trains the audio encoder from scratch (no pretraining), so the
        # encoder uses the larger from-scratch LR and BatchNorm stays trainable.
        model = ConsensusAudioModel(get_audio_encoder(encoder), Config.NUM_CLASSES, Config.SHARED_DIM).to(Config.DEVICE)
        optimizer = torch.optim.Adam(
            [
                {"params": model.encoder.parameters(), "lr": Config.BASELINE_ENCODER_LR},
                {"params": model.shared.parameters(), "lr": Config.HEAD_LR},
                {"params": model.classifier.parameters(), "lr": Config.HEAD_LR},
            ]
        )
        checkpoint = run / "checkpoints" / f"fold_{fold}.pth"
        stopper = EarlyStopping(patience=Config.PATIENCE, path=checkpoint, maximize=False,
                                min_epochs=Config.MIN_EPOCHS)
        best_epoch, best_score, best_loss = -1, 0.0, np.inf
        history = []
        for epoch in range(Config.FINETUNE_EPOCHS):
            model.train()
            ramp = min(1.0, (epoch + 1) / max(Hparams.GUIDANCE_RAMPUP_EPOCHS, 1))
            weight = guidance_weight * ramp
            totals = {"ce": 0.0, "guidance": 0.0}
            batches = 0
            for spec, label, _pid in train_loader:
                spec, label = spec.to(Config.DEVICE), label.to(Config.DEVICE)
                optimizer.zero_grad()
                logits, shared = model(spec, return_shared=True)
                ce = ce_loss(logits, label)
                mask = (label < len(Config.PRIVILEGED_CLASSES)).float()  # Healthy/COPD only
                guidance, _parts = disease_consensus_loss(
                    shared, label, class_prototypes, mask,
                    topic_prototypes=topic_prototypes,
                    topic_temperature=Hparams.TOPIC_TEMPERATURE,
                    disable_disease=disable_disease,
                    disable_topic=disable_topic,
                )
                loss = ce + weight * guidance
                loss.backward()
                optimizer.step()
                totals["ce"] += float(ce.detach().cpu())
                totals["guidance"] += float(guidance.detach().cpu())
                batches += 1
            val_loss = evaluate_task_loss(model, val_loader)
            truth, prediction, _, _ = evaluate_patient_level(model, val_loader)
            score = macro_f1(truth, prediction)
            history.append({
                "epoch": epoch + 1,
                "guidance_weight": weight,
                "train_ce": totals["ce"] / max(batches, 1),
                "train_guidance": totals["guidance"] / max(batches, 1),
                "val_loss": val_loss,
                "val_macro_f1": score,
            })
            print(f"   [dtca fold {fold}] epoch {epoch + 1:02d} | w {weight:.2f} "
                  f"| val loss {val_loss:.4f} | val macro-F1 {score:.4f}", flush=True)
            if val_loss < best_loss:                   # selection on validation loss
                best_loss, best_score, best_epoch = val_loss, score, epoch + 1
            stopper(val_loss, model)
            if stopper.early_stop:
                break
        pd.DataFrame(history).to_csv(run / "metrics" / f"training_history_fold_{fold}.csv", index=False)

        model.load_state_dict(torch.load(checkpoint, map_location=Config.DEVICE))
        truth, prediction, probabilities, ids = evaluate_patient_level(model, test_loader)
        save_classification_report(truth, prediction, run / "reports", f"fold_{fold}")
        frame = prediction_frame(truth, prediction, probabilities, ids)
        frame["seed"], frame["fold"] = seed, fold
        frame.to_csv(run / "predictions" / f"patient_preds_fold_{fold}.csv", index=False)
        pooled_predictions.append(frame)
        fold_rows.append({"fold": fold, "best_epoch": best_epoch, "best_val_macro_f1": best_score, **fold_metrics(truth, prediction)})

        if external is not None:
            ext_loader = DataLoader(
                AudioClassifierDataset(external, processed_dir=Config.EXTERNAL_PROCESSED_DIR),
                batch_size=Config.BATCH_SIZE, shuffle=False,
            )
            e_truth, e_pred, e_prob, e_ids = evaluate_patient_level(model, ext_loader)
            e_frame = prediction_frame(e_truth, e_pred, e_prob, e_ids)
            e_frame["seed"], e_frame["fold_model"] = seed, fold
            e_frame.to_csv(run / "predictions" / f"cwlsd_preds_fold_{fold}.csv", index=False)

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(run / "metrics" / "fold_metrics.csv", index=False)
    save_json(
        {m: {"mean": float(fold_frame[m].mean()), "std": float(fold_frame[m].std(ddof=0))}
         for m in ("accuracy", "macro_f1", "weighted_f1")},
        run / "metrics" / "summary_metrics.json",
    )
    pooled = pd.concat(pooled_predictions, ignore_index=True)
    save_classification_report(pooled["y_true"], pooled["y_pred"], run / "reports", "pooled")
    save_json(bootstrap_ci(pooled, seed=seed), run / "metrics" / "bootstrap_ci.json")
    print(f"[dtca] done -> {run}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disease-Topic Consensus Alignment (audio guided by privileged prototypes)")
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument("--encoder", default=Config.AUDIO_ENCODER, choices=["CNN", "CRNN", "TRANSFORMER"])
    parser.add_argument("--view", default="omics", choices=sorted(PRIVILEGED_VIEWS))
    parser.add_argument("--guidance_weight", type=float, default=Hparams.GUIDANCE_WEIGHT)
    parser.add_argument("--ablation", default="none", choices=ABLATIONS)
    args = parser.parse_args()
    train_dtca(args.seed, args.encoder.upper(), args.view, args.guidance_weight, args.ablation)
