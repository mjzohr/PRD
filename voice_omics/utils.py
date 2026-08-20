"""Shared training/evaluation helpers. Kept small and dependency-light.

Evaluation is always patient-level: slice softmax probabilities are averaged per
patient, then argmax. Model selection and every reported metric use this view.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import WeightedRandomSampler

from config import Config


class EarlyStopping:
    """Stop when the monitored validation metric stops improving. Saves the best."""

    def __init__(self, patience=5, delta=0.0, path="checkpoint.pth", maximize=True,
                 min_epochs=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = str(path)
        self.maximize = maximize
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.counter = 0
        self.epochs_seen = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric, model):
        self.epochs_seen += 1
        score = val_metric if self.maximize else -val_metric
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            # A minimum number of epochs must elapse before stopping, so a noisy
            # early validation reading cannot end training prematurely.
            if self.counter >= self.patience and self.epochs_seen >= self.min_epochs:
                self.early_stop = True


def freeze_bn_stats(module: nn.Module) -> None:
    """Put BatchNorm layers in eval mode so small fine-tune sets don't corrupt
    the pretrained running statistics. Apply after ``model.train()`` each epoch."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def method_run_dir(method: str, tag: str):
    """Standard results layout so ``report.py`` can aggregate every experiment:
    ``results/<method>/<tag>/{metrics,reports,predictions,checkpoints}/``.
    ``tag`` encodes view/encoder/knob/seed, e.g. ``omics/CNN/w2.0/seed42``.
    """
    root = Config.RESULTS_DIR / method / tag
    for sub in ("metrics", "reports", "predictions", "checkpoints"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def save_json(value, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def patient_weighted_sampler(frame: pd.DataFrame, generator: torch.Generator) -> WeightedRandomSampler:
    """Each patient contributes equally regardless of slice count or class size."""
    patient_class_counts = frame[["pid", "label_idx"]].drop_duplicates("pid")["label_idx"].value_counts()
    slices_per_patient = frame.groupby("pid").size()
    weights = np.asarray(
        [
            1.0 / (patient_class_counts.loc[row.label_idx] * slices_per_patient.loc[row.pid])
            for row in frame[["pid", "label_idx"]].itertuples(index=False)
        ],
        dtype=np.float64,
    )
    return WeightedRandomSampler(
        torch.from_numpy(weights).double(), num_samples=len(weights), replacement=True, generator=generator
    )


@torch.no_grad()
def evaluate_patient_level(model, loader):
    """Aggregate slice probabilities to one prediction per patient.

    Works with any loader whose batch is ``(spec, label, ..., pid)`` -- the audio
    spectrogram is first, the integer label second, the patient id last.
    """
    model.eval()
    probs_by_patient: dict[int, list[np.ndarray]] = {}
    truth_by_patient: dict[int, int] = {}
    for batch in loader:
        spec, label, pid = batch[0], batch[1], batch[-1]
        probabilities = torch.softmax(model(spec.to(Config.DEVICE)), dim=1).cpu().numpy()
        for i in range(len(pid)):
            key = int(pid[i].item()) if torch.is_tensor(pid[i]) else int(pid[i])
            probs_by_patient.setdefault(key, []).append(probabilities[i])
            truth_by_patient[key] = int(label[i].item())
    ids, truths, predictions, probabilities = [], [], [], []
    for key, values in probs_by_patient.items():
        average = np.mean(values, axis=0)
        ids.append(key)
        truths.append(truth_by_patient[key])
        predictions.append(int(average.argmax()))
        probabilities.append(average)
    return truths, predictions, probabilities, ids


@torch.no_grad()
def evaluate_task_loss(model, loader) -> float:
    """Mean slice-level cross-entropy on a loader -- the model-selection signal.

    Continuous, so it does not suffer the severe quantization of patient macro-F1
    over a small inner-validation split. Only the TASK loss is used (never the
    auxiliary term), so baseline and transfer runs are selected identically.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, count = 0.0, 0
    for batch in loader:
        spec, label = batch[0].to(Config.DEVICE), batch[1].to(Config.DEVICE)
        logits = model(spec)
        total += float(criterion(logits, label).detach().cpu()) * int(label.numel())
        count += int(label.numel())
    return total / max(count, 1)


def macro_f1(truth, prediction) -> float:
    return float(
        f1_score(truth, prediction, labels=range(Config.NUM_CLASSES), average="macro", zero_division=0)
    )


def prediction_frame(truths, predictions, probabilities, ids) -> pd.DataFrame:
    values = np.vstack(probabilities)
    frame = pd.DataFrame({"patient_id": ids, "y_true": truths, "y_pred": predictions})
    for index, class_name in enumerate(Config.CLASSES):
        frame[f"prob_{class_name}"] = values[:, index]
    return frame


def fold_metrics(truth, prediction) -> dict:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": macro_f1(truth, prediction),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
    }


def save_classification_report(truth, prediction, out_dir: Path, tag: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = dict(labels=range(Config.NUM_CLASSES), target_names=Config.CLASSES, zero_division=0)
    report = classification_report(truth, prediction, output_dict=True, **args)
    save_json(report, out_dir / f"classification_report_{tag}.json")
    (out_dir / f"classification_report_{tag}.txt").write_text(
        classification_report(truth, prediction, **args), encoding="utf-8"
    )
    matrix = confusion_matrix(truth, prediction, labels=range(Config.NUM_CLASSES))
    pd.DataFrame(matrix, index=Config.CLASSES, columns=Config.CLASSES).to_csv(
        out_dir / f"confusion_matrix_{tag}.csv"
    )
    return report


def bootstrap_ci(frame: pd.DataFrame, samples: int = Config.BOOTSTRAP_SAMPLES, seed: int = Config.SEED) -> dict:
    rng = np.random.default_rng(seed)
    truth = frame["y_true"].to_numpy(dtype=int)
    prediction = frame["y_pred"].to_numpy(dtype=int)
    accuracy, f1 = [], []
    for _ in range(samples):
        positions = rng.integers(0, len(frame), len(frame))
        accuracy.append(accuracy_score(truth[positions], prediction[positions]))
        f1.append(macro_f1(truth[positions], prediction[positions]))
    return {
        "accuracy_ci_low": float(np.quantile(accuracy, 0.025)),
        "accuracy_ci_high": float(np.quantile(accuracy, 0.975)),
        "macro_f1_ci_low": float(np.quantile(f1, 0.025)),
        "macro_f1_ci_high": float(np.quantile(f1, 0.975)),
        "bootstrap_samples": samples,
    }
