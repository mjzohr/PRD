"""Text-anchored contrastive clinical pivot (unpaired audio <- privileged transfer).

Two phases, repeated over the nested patient folds:

PHASE 1 (pretrain): an audio encoder and a privileged tabular encoder are each
aligned to a FROZEN clinical-text manifold. Same-diagnosis samples are positives
(``label_aware``); the audio and privileged cohorts are never paired. The audio
encoder weights are the transferable product of this phase.

PHASE 2 (fine-tune): the pretrained audio encoder is reloaded into a 3-class
classifier and fine-tuned with the proven recipe (gentle encoder LR, faster head,
frozen BatchNorm, patient-weighted sampling, patient-level early stopping).

Ablations via ``--loss_type``: ``vanilla`` swaps the label-aware CLIP loss for the
diagonal-only InfoNCE; ``direct`` drops the text pivot and aligns audio<->privileged
embeddings directly.
"""

from __future__ import annotations

import argparse
import math
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, Hparams, PRIVILEGED_VIEWS
from dataset import AudioClassifierDataset, AudioTextDataset, PrivilegedTextDataset
from losses import label_aware_clip_loss, vanilla_clip_loss
from models import (
    ClassifierHead,
    ProjectionHead,
    TabularEncoder,
    TextEncoder,
    get_audio_encoder,
)
from privileged import TeacherPreprocessor, load_view
from splitting import build_nested_patient_splits, select_patient_rows
from utils import (
    EarlyStopping,
    bootstrap_ci,
    evaluate_patient_level,
    evaluate_task_loss,
    fold_metrics,
    freeze_bn_stats,
    macro_f1,
    method_run_dir,
    patient_weighted_sampler,
    prediction_frame,
    save_classification_report,
    save_json,
)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ContrastiveModel(nn.Module):
    """Audio + privileged encoders, each projected next to a frozen clinical-text
    anchor. ``logit_scale`` is a learnable temperature (CLIP convention)."""

    def __init__(self, priv_input_dim, temperature=Hparams.TEMPERATURE, encoder=None,
                 clinical_encoder="bioclinicalbert"):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.audio_enc = get_audio_encoder(encoder)
        self.priv_enc = TabularEncoder(priv_input_dim)
        self.text_enc = TextEncoder(Config.DEVICE, encoder_name=clinical_encoder)

        projection_dim = Hparams.PROJECTION_DIM
        self.audio_proj = ProjectionHead(512, embed_dim=768, projection_dim=projection_dim)
        self.priv_proj = ProjectionHead(512, embed_dim=768, projection_dim=projection_dim)
        # Text width depends on the clinical encoder (e.g. GatorTron is 1024).
        self.text_proj = ProjectionHead(self.text_enc.output_dim, embed_dim=768, projection_dim=projection_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / temperature))

    def forward(self, audio, priv, audio_text_list=None, priv_text_list=None, skip_text=False):
        audio_emb = self.audio_proj(self.audio_enc(audio))
        priv_emb = self.priv_proj(self.priv_enc(priv))
        if skip_text:
            return audio_emb, priv_emb, None, None
        text_from_audio_emb = self.text_proj(self.text_enc(audio_text_list))
        text_from_priv_emb = self.text_proj(self.text_enc(priv_text_list))
        return audio_emb, priv_emb, text_from_audio_emb, text_from_priv_emb


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #
def pretrain_fold(args, audio_train_frame, priv_ds, priv_input_dim, generator, checkpoint_path):
    """PHASE 1: align audio and privileged views to the frozen clinical-text manifold."""
    audio_ds = AudioTextDataset(audio_train_frame, augment=False)
    audio_loader = DataLoader(
        audio_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        worker_init_fn=Config.seed_worker,
        generator=generator,
    )
    priv_loader = DataLoader(
        priv_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        worker_init_fn=Config.seed_worker,
        generator=generator,
    )

    model = ContrastiveModel(
        priv_input_dim=priv_input_dim,
        temperature=args.temperature,
        encoder=args.encoder,
        clinical_encoder=args.clinical_encoder,
    ).to(Config.DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4
    )
    skip_text = args.loss_type == "direct"

    # Cycle the shorter loader so every step of the longer one is used.
    if len(audio_loader) >= len(priv_loader):
        larger, smaller, large_is_audio = audio_loader, priv_loader, True
    else:
        larger, smaller, large_is_audio = priv_loader, audio_loader, False
    steps = max(len(audio_loader), len(priv_loader))

    for epoch in range(Hparams.CONTRASTIVE_EPOCHS):
        model.train()
        running = 0.0
        for batch_large, batch_small in zip(larger, cycle(smaller)):
            if large_is_audio:
                (aud_spec, aud_text, aud_label) = batch_large
                (priv_feat, priv_text, priv_label) = batch_small
            else:
                (priv_feat, priv_text, priv_label) = batch_large
                (aud_spec, aud_text, aud_label) = batch_small

            aud_spec = aud_spec.to(Config.DEVICE)
            priv_feat = priv_feat.to(Config.DEVICE)
            aud_label = aud_label.to(Config.DEVICE, dtype=torch.long)
            priv_label = priv_label.to(Config.DEVICE, dtype=torch.long)

            optimizer.zero_grad()
            aud_emb, priv_emb, txt_aud_emb, txt_priv_emb = model(
                aud_spec, priv_feat, list(aud_text), list(priv_text), skip_text=skip_text
            )

            with torch.no_grad():
                model.logit_scale.clamp_(0.0, 4.605)
            scale = model.logit_scale.exp()

            if args.loss_type == "label_aware":
                loss = 0.5 * label_aware_clip_loss(aud_emb, txt_aud_emb, aud_label, aud_label, scale) \
                    + 0.5 * label_aware_clip_loss(priv_emb, txt_priv_emb, priv_label, priv_label, scale)
            elif args.loss_type == "audio_text_only":
                # Matched audio-only control: same encoder + recipe, but NO privileged
                # modality -- only the audio<->clinical-text term. Isolates the gain
                # that omics/CT actually add on top of the audio+text pivot.
                loss = label_aware_clip_loss(aud_emb, txt_aud_emb, aud_label, aud_label, scale)
            elif args.loss_type == "vanilla":
                loss = 0.5 * vanilla_clip_loss(aud_emb, txt_aud_emb, scale) \
                    + 0.5 * vanilla_clip_loss(priv_emb, txt_priv_emb, scale)
            else:  # direct: skip the text pivot, align audio<->privileged directly
                loss = 0.5 * label_aware_clip_loss(aud_emb, priv_emb, aud_label, priv_label, scale) \
                    + 0.5 * label_aware_clip_loss(priv_emb, aud_emb, priv_label, aud_label, scale)

            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
        print(f"      [pretrain] epoch {epoch + 1:02d}/{Hparams.CONTRASTIVE_EPOCHS} "
              f"| loss {running / max(steps, 1):.4f} | scale {float(scale.detach().cpu()):.3f}")

    torch.save(model.audio_enc.state_dict(), checkpoint_path)
    return checkpoint_path


def finetune_fold(args, development, split, pretrained_path, generator, ckpt_dir):
    """PHASE 2: reload the pretrained audio encoder and fine-tune a 3-class classifier."""
    audio_train_frame = select_patient_rows(development, split.train_ids)
    val_frame = select_patient_rows(development, split.validation_ids)
    test_frame = select_patient_rows(development, split.test_ids)

    clf = ClassifierHead(get_audio_encoder(args.encoder), Config.NUM_CLASSES).to(Config.DEVICE)
    clf.encoder.load_state_dict(torch.load(pretrained_path, map_location=Config.DEVICE))

    optimizer = torch.optim.Adam(
        [
            {"params": clf.encoder.parameters(), "lr": Config.ENCODER_LR},
            {"params": clf.head.parameters(), "lr": Config.HEAD_LR},
        ]
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        AudioClassifierDataset(audio_train_frame),
        batch_size=Config.BATCH_SIZE,
        sampler=patient_weighted_sampler(audio_train_frame, generator),
        shuffle=False,
        drop_last=True,  # BatchNorm needs >1 sample; avoids a size-1 final batch
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        worker_init_fn=Config.seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        AudioClassifierDataset(val_frame),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        worker_init_fn=Config.seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        AudioClassifierDataset(test_frame),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        worker_init_fn=Config.seed_worker,
        generator=generator,
    )

    checkpoint = ckpt_dir / f"finetune_fold_{split.outer_fold}.pth"
    early_stopping = EarlyStopping(patience=Config.PATIENCE, path=str(checkpoint), maximize=False,
                                   min_epochs=Config.MIN_EPOCHS)
    best_epoch, best_score, best_loss = -1, 0.0, np.inf
    for epoch in range(Config.FINETUNE_EPOCHS):
        clf.train()
        freeze_bn_stats(clf)
        total_loss = 0.0
        for spec, label, _ in train_loader:
            spec, label = spec.to(Config.DEVICE), label.to(Config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(clf(spec), label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        val_loss = evaluate_task_loss(clf, val_loader)
        truth, prediction, _, _ = evaluate_patient_level(clf, val_loader)
        score = macro_f1(truth, prediction)
        if val_loss < best_loss:                       # selection on validation loss
            best_loss, best_score, best_epoch = val_loss, score, epoch + 1
        print(f"      [finetune] epoch {epoch + 1:02d} | train loss "
              f"{total_loss / max(len(train_loader), 1):.4f} | val loss {val_loss:.4f} "
              f"| val macro-F1 {score:.4f}")
        early_stopping(val_loss, clf)
        if early_stopping.early_stop:
            print("      [finetune] early stopping")
            break

    clf.load_state_dict(torch.load(checkpoint, map_location=Config.DEVICE))
    truth, prediction, probabilities, ids = evaluate_patient_level(clf, test_loader)
    return clf, (truth, prediction, probabilities, ids), best_epoch, best_score, test_loader


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(args):
    Config.set_seed(args.seed)
    view = PRIVILEGED_VIEWS[args.view]

    tag = f"{args.view}/{args.encoder}/ce_{args.clinical_encoder}/t{args.temperature}/seed{args.seed}"
    if args.loss_type != "label_aware":
        tag += f"/abl_{args.loss_type}"
    run_dir = method_run_dir("contrastive", tag)
    metrics_dir = run_dir / "metrics"
    reports_dir = run_dir / "reports"
    preds_dir = run_dir / "predictions"
    ckpt_dir = run_dir / "checkpoints"

    save_json(
        {
            "method": "contrastive_clinical_pivot",
            "view": args.view,
            "encoder": args.encoder,
            "clinical_encoder": args.clinical_encoder,
            "loss_type": args.loss_type,
            "temperature": args.temperature,
            "seed": args.seed,
            "CLASSES": Config.CLASSES,
            "NUM_FOLDS": Config.NUM_FOLDS,
            "CONTRASTIVE_EPOCHS": Hparams.CONTRASTIVE_EPOCHS,
            "PROJECTION_DIM": Hparams.PROJECTION_DIM,
            "FINETUNE_EPOCHS": Config.FINETUNE_EPOCHS,
            "PATIENCE": Config.PATIENCE,
            "ENCODER_LR": Config.ENCODER_LR,
            "HEAD_LR": Config.HEAD_LR,
            "INNER_VALIDATION_FRACTION": Config.INNER_VALIDATION_FRACTION,
            "audio_privileged_pairing_used": False,
            "EXTERNAL_POLICY": "evaluation only; never used for fitting or early stopping",
            "VALIDATION_PROTOCOL": "repeated nested patient-stratified cross-validation",
        },
        run_dir / "config.json",
    )

    # --- audio development cohort + nested patient folds ---
    development = pd.read_csv(Config.FULL_CSV)
    _, nested_splits = build_nested_patient_splits(
        development,
        folds=Config.NUM_FOLDS,
        repeat_seed=args.seed,
        validation_fraction=Config.INNER_VALIDATION_FRACTION,
    )

    # --- privileged anchor: fit the preprocessor ONCE, shared across folds ---
    lv = load_view(view, sharp=True)
    preprocessor = TeacherPreprocessor(view.max_features).fit(lv.features, lv.labels.to_numpy())
    x_priv = preprocessor.transform(lv.features)
    priv_ds = PrivilegedTextDataset(x_priv, lv.clinical, lv.labels.to_numpy(), source_type=view.name)
    priv_input_dim = x_priv.shape[1]

    external = None
    if Path(Config.EXTERNAL_CSV).is_file():
        external = pd.read_csv(Config.EXTERNAL_CSV)

    fold_rows, pooled_frames = [], []
    started = time.time()
    for split in nested_splits:
        fold = split.outer_fold
        print(f"\n=== Fold {fold} ===")
        generator = Config.get_generator(args.seed * 1000 + fold)

        # PHASE 1
        audio_train_frame = select_patient_rows(development, split.train_ids)
        pretrained_path = ckpt_dir / f"pretrained_encoder_fold_{fold}.pth"
        pretrain_fold(args, audio_train_frame, priv_ds, priv_input_dim, generator, pretrained_path)

        # PHASE 2
        clf, (truth, prediction, probabilities, ids), best_epoch, best_score, _ = finetune_fold(
            args, development, split, pretrained_path, generator, ckpt_dir
        )

        # per-fold reporting
        save_classification_report(truth, prediction, reports_dir, f"fold_{fold}")
        fold_frame = prediction_frame(truth, prediction, probabilities, ids)
        fold_frame["seed"] = args.seed
        fold_frame["fold"] = fold
        fold_frame.to_csv(preds_dir / f"patient_preds_fold_{fold}.csv", index=False)
        pooled_frames.append(fold_frame)

        metrics = fold_metrics(truth, prediction)
        fold_rows.append(
            {
                "fold": fold,
                "seed": args.seed,
                "evaluation_split": "outer_test",
                **metrics,
                "best_epoch": best_epoch,
                "best_val_macro_f1": float(best_score),
            }
        )
        print(f"   fold {fold} test | acc {metrics['accuracy']:.4f} "
              f"| macro-F1 {metrics['macro_f1']:.4f} | weighted-F1 {metrics['weighted_f1']:.4f}")

        # external CWLSD (evaluation only)
        if external is not None:
            ext_loader = DataLoader(
                AudioClassifierDataset(external, processed_dir=Config.EXTERNAL_PROCESSED_DIR),
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=Config.seed_worker,
                generator=generator,
            )
            e_truth, e_pred, e_prob, e_ids = evaluate_patient_level(clf, ext_loader)
            ext_frame = prediction_frame(e_truth, e_pred, e_prob, e_ids)
            ext_frame["seed"] = args.seed
            ext_frame["fold"] = fold
            ext_frame.to_csv(preds_dir / f"cwlsd_preds_fold_{fold}.csv", index=False)

    # --- aggregate across folds ---
    fold_frame_all = pd.DataFrame(fold_rows)
    fold_frame_all.to_csv(metrics_dir / "fold_metrics.csv", index=False)
    summary = {
        metric: {
            "mean": float(fold_frame_all[metric].mean()),
            "std": float(fold_frame_all[metric].std(ddof=0)),
        }
        for metric in ("accuracy", "macro_f1", "weighted_f1")
    }
    summary["runtime_seconds"] = time.time() - started
    save_json(summary, metrics_dir / "summary_metrics.json")

    # --- pooled out-of-fold predictions ---
    pooled = pd.concat(pooled_frames, ignore_index=True)
    save_classification_report(
        pooled["y_true"].tolist(), pooled["y_pred"].tolist(), reports_dir, "pooled"
    )
    save_json(bootstrap_ci(pooled, seed=args.seed), metrics_dir / "bootstrap_ci.json")

    print(f"\nAll contrastive outputs saved under: {run_dir}")
    print(f"Pooled macro-F1: {macro_f1(pooled['y_true'].tolist(), pooled['y_pred'].tolist()):.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Text-anchored contrastive clinical pivot.")
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument("--encoder", type=str, default=Config.AUDIO_ENCODER)
    parser.add_argument("--view", choices=["omics", "ct_clinical"], default="omics")
    parser.add_argument(
        "--clinical_encoder",
        choices=Config.CLINICAL_ENCODER_CHOICES,
        default=Config.CLINICAL_ENCODER,
    )
    parser.add_argument("--temperature", type=float, default=Hparams.TEMPERATURE)
    parser.add_argument(
        "--loss_type",
        choices=["label_aware", "audio_text_only", "vanilla", "direct"],
        default="label_aware",
        help="label_aware = full method; audio_text_only = matched audio-only control "
             "(no privileged modality); vanilla/direct = loss-form ablations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
