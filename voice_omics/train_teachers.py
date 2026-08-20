"""Train one privileged tabular teacher (omics or CT-clinical).

The teacher learns Healthy vs (severe) COPD from the COPDGene tabular view, using
the shared foundation modules for loading/preprocessing (``privileged.py``), the
model (``models.TeacherModel``) and the results layout (``utils.method_run_dir``).

It exports everything the downstream bridge/distillation needs:
  * the full model state dict,
  * the fitted preprocessor state,
  * class- and topic-level prototypes computed in the shared subspace,
  * a locked-test classification report and a metrics/config summary.

Run:
    python train_teachers.py --view omics --seed 42
    python train_teachers.py --view ct_clinical
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import Config, Hparams, PRIVILEGED_VIEWS, Sharp
from models import TeacherModel
from privileged import TeacherPreprocessor, load_view
from utils import method_run_dir, save_classification_report, save_json

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
TEACHER_TARGET_NAMES = ["Healthy", "COPD"]


def _binary_macro_f1(truth, prediction) -> float:
    """Macro F1 over the two teacher classes (Healthy=0, COPD=1)."""
    return float(f1_score(truth, prediction, labels=[0, 1], average="macro", zero_division=0))


def _predict(model, loader, device):
    model.eval()
    truth, prediction = [], []
    with torch.no_grad():
        for values, labels in loader:
            logits = model(values.to(device))
            prediction.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            truth.extend(labels.numpy().tolist())
    return truth, prediction


def _stratified_positions(y: np.ndarray, seed: int):
    """Split subjects into train / inner-val / locked-test (stratified)."""
    dev_pos, test_pos = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=seed, shuffle=True, stratify=y
    )
    train_pos, val_pos = train_test_split(
        dev_pos,
        test_size=Config.INNER_VALIDATION_FRACTION,
        random_state=seed,
        shuffle=True,
        stratify=y[dev_pos],
    )
    return train_pos, val_pos, test_pos


def _compute_prototypes(model, x_train: np.ndarray, y_train: np.ndarray, seed: int, device):
    """Encode all training subjects through the shared subspace and derive
    class means and KMeans topic centroids per class."""
    model.eval()
    with torch.no_grad():
        _, shared = model(torch.from_numpy(x_train).to(device), return_shared=True)
    shared = shared.cpu().numpy().astype(np.float32)

    num_topics = Hparams.NUM_TOPICS
    shared_dim = shared.shape[1]
    class_prototypes = np.zeros((2, shared_dim), dtype=np.float32)
    topic_prototypes = np.zeros((2, num_topics, shared_dim), dtype=np.float32)

    for cls in (0, 1):
        embeddings = shared[y_train == cls]
        class_mean = embeddings.mean(axis=0)
        class_prototypes[cls] = class_mean
        if len(embeddings) < num_topics:
            topic_prototypes[cls] = np.tile(class_mean, (num_topics, 1))
        else:
            km = KMeans(n_clusters=num_topics, random_state=seed, n_init=10).fit(embeddings)
            topic_prototypes[cls] = km.cluster_centers_.astype(np.float32)

    return class_prototypes, topic_prototypes


def train_teacher(view_name: str, seed: int) -> None:
    Config.set_seed(seed)
    device = Config.DEVICE
    view = PRIVILEGED_VIEWS[view_name]

    # 1. Load the privileged view (sharp Healthy vs severe COPD labels).
    lv = load_view(view, sharp=True)
    features = lv.features
    y = lv.labels.to_numpy()

    # 2. Stratified subject split.
    train_pos, val_pos, test_pos = _stratified_positions(y, seed)

    # 3. Fit preprocessor on training rows only; transform each split.
    pre = TeacherPreprocessor(view.max_features).fit(features.iloc[train_pos], y[train_pos])
    x_train = pre.transform(features.iloc[train_pos])
    x_val = pre.transform(features.iloc[val_pos])
    x_test = pre.transform(features.iloc[test_pos])
    y_train = y[train_pos].astype(np.int64)
    y_val = y[val_pos].astype(np.int64)
    y_test = y[test_pos].astype(np.int64)

    input_dim = int(x_train.shape[1])

    # 4. Build and train the teacher.
    generator = Config.get_generator(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # BatchNorm needs >1 sample; avoids a size-1 final batch
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TeacherModel(input_dim=input_dim, shared_dim=Config.SHARED_DIM, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_epoch, best_state = -1.0, -1, None
    for epoch in range(EPOCHS):
        model.train()
        for values, labels in train_loader:
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(values), labels)
            loss.backward()
            optimizer.step()
        truth, prediction = _predict(model, val_loader, device)
        val_macro_f1 = _binary_macro_f1(truth, prediction)
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Teacher training produced no selectable model")

    # 5. Restore best-by-validation weights, evaluate on the locked test.
    model.load_state_dict(best_state)
    test_truth, test_pred = _predict(model, test_loader, device)
    test_accuracy = float(accuracy_score(test_truth, test_pred))
    test_macro_f1 = _binary_macro_f1(test_truth, test_pred)
    print(f"[teacher:{view.name} seed{seed}] best val macro-F1 = {best_f1:.4f} (epoch {best_epoch})")
    print(classification_report(test_truth, test_pred, target_names=TEACHER_TARGET_NAMES, zero_division=0))

    # 6. Shared-subspace prototypes from all training subjects.
    class_prototypes, topic_prototypes = _compute_prototypes(model, x_train, y_train, seed, device)

    # 7. Export everything under the standard results layout.
    run = method_run_dir("teachers", f"{view.name}/seed{seed}")
    checkpoints = run / "checkpoints"

    torch.save(model.state_dict(), checkpoints / "teacher_model.pth")

    pre_state = pre.to_state()
    pre_state["selected_features"] = np.asarray(pre_state["selected_features"], dtype=object)
    np.savez(checkpoints / "preprocessor.npz", **pre_state)

    np.savez(
        checkpoints / "prototypes.npz",
        class_prototypes=class_prototypes,
        topic_prototypes=topic_prototypes,
    )

    save_classification_report(test_truth, test_pred, run / "reports", "locked_test")

    save_json(
        {
            "best_validation_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "locked_test_accuracy": test_accuracy,
            "locked_test_macro_f1": test_macro_f1,
            "input_dim": input_dim,
            "input_features": int(features.shape[1]),
            "selected_features": int(len(pre.selected)),
            "train_subjects": int(len(y_train)),
            "validation_subjects": int(len(y_val)),
            "locked_test_subjects": int(len(y_test)),
            "sharp": bool(Sharp.ENABLE),
        },
        run / "metrics" / "teacher_metrics.json",
    )

    pd.DataFrame(
        {"rank": np.arange(1, len(pre.selected) + 1), "feature": pre.selected}
    ).to_csv(run / "selected_features.csv", index=False)

    save_json(
        {
            "view": view.name,
            "seed": seed,
            "shared_dim": Config.SHARED_DIM,
            "max_features": view.max_features,
            "sharp": bool(Sharp.ENABLE),
        },
        run / "config.json",
    )

    print(f"[teacher:{view.name} seed{seed}] locked-test macro-F1 = {test_macro_f1:.4f}; saved to {run}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one privileged tabular teacher.")
    parser.add_argument("--view", required=True, choices=sorted(PRIVILEGED_VIEWS.keys()))
    parser.add_argument("--seed", type=int, default=Config.SEED)
    args = parser.parse_args()
    train_teacher(args.view, args.seed)


if __name__ == "__main__":
    main()
