"""Architecture search for the CNN, CRNN and Transformer audio encoders.

Finds better hyper-parameters by training the audio-only classifier for each
config over patient nested CV and ranking on inner-VALIDATION macro-F1 (test is
reported for reference only, never used for selection). At the end it logs the
winning architecture + parameters for each encoder, which you then plug into the
distillation comparison.

Grids are intentionally small (eight configs per family) but cover the parameters
that matter most:
  CNN         : conv widths, kernel size, dropout
  CRNN        : GRU hidden size, GRU depth, dropout
  TRANSFORMER : model width, depth, and POSITIONAL ENCODING (the default has none,
                so attention is order-blind -- likely the main culprit)

Every grid contains the current default config, so the search can confirm the
shipped architecture as well as replace it.

Run:  RESULTS_ROOT=results_arch_search python arch_search.py --seeds 42
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import AudioClassifierDataset
from models import AudioEncoderCNN, AudioEncoderCRNN, AudioEncoderTransformer, ClassifierHead
from splitting import build_nested_patient_splits, select_patient_rows
from utils import evaluate_patient_level, macro_f1, method_run_dir, patient_weighted_sampler, save_json

CNN_GRID = [
    {"conv_channels": ch, "kernel_size": k, "dropout": d}
    for ch in ((128, 256, 512), (64, 128, 256)) for k in (3, 5) for d in (0.0, 0.3)
]
CRNN_GRID = [
    {"gru_hidden": h, "gru_layers": layers, "dropout": d}
    for h in (128, 256) for layers in (1, 2) for d in (0.0, 0.3)
]
TRANSFORMER_GRID = [
    {"d_model": dm, "nhead": nh, "layers": layers, "dim_feedforward": 2 * dm,
     "dropout": 0.1, "positional_encoding": pe}
    for dm, nh in ((256, 4), (512, 8)) for layers in (2, 4) for pe in (False, True)
]
GRIDS = {"CNN": CNN_GRID, "CRNN": CRNN_GRID, "TRANSFORMER": TRANSFORMER_GRID}


ENCODERS = {"CNN": AudioEncoderCNN, "CRNN": AudioEncoderCRNN, "TRANSFORMER": AudioEncoderTransformer}


def _build(kind: str, params: dict) -> nn.Module:
    return ENCODERS[kind](**params)


def _loader(frame, seed, fold, *, train):
    dataset = AudioClassifierDataset(frame)
    if train:
        g = Config.get_generator(seed * 1000 + fold)
        return DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                          sampler=patient_weighted_sampler(frame, g), drop_last=True,
                          num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
                          worker_init_fn=Config.seed_worker, generator=g)
    return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                      num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)


def evaluate_config(kind: str, params: dict, seeds, epochs: int):
    """Train the audio-only classifier for this config; return (val_mean, val_std, test_mean)."""
    development = pd.read_csv(Config.FULL_CSV)
    val_scores, test_scores = [], []
    for seed in seeds:
        Config.set_seed(seed)
        _, splits = build_nested_patient_splits(
            development, folds=Config.NUM_FOLDS, repeat_seed=seed,
            validation_fraction=Config.INNER_VALIDATION_FRACTION,
        )
        for split in splits:
            fold = split.outer_fold
            train_frame = select_patient_rows(development, split.train_ids)
            val_frame = select_patient_rows(development, split.validation_ids)
            test_frame = select_patient_rows(development, split.test_ids)
            model = ClassifierHead(_build(kind, params), Config.NUM_CLASSES).to(Config.DEVICE)
            optimizer = torch.optim.Adam([
                {"params": model.encoder.parameters(), "lr": Config.BASELINE_ENCODER_LR},
                {"params": model.head.parameters(), "lr": Config.HEAD_LR},
            ])
            criterion = nn.CrossEntropyLoss()
            train_loader = _loader(train_frame, seed, fold, train=True)
            val_loader = _loader(val_frame, seed, fold, train=False)
            test_loader = _loader(test_frame, seed, fold, train=False)

            best_val, best_state = -1.0, None
            for _ in range(epochs):
                model.train()
                for spec, label, _pid in train_loader:
                    spec, label = spec.to(Config.DEVICE), label.to(Config.DEVICE)
                    optimizer.zero_grad()
                    criterion(model(spec), label).backward()
                    optimizer.step()
                truth, pred, _, _ = evaluate_patient_level(model, val_loader)
                score = macro_f1(truth, pred)
                if score > best_val:
                    best_val, best_state = score, copy.deepcopy(model.state_dict())
            val_scores.append(best_val)
            model.load_state_dict(best_state)
            t, p, _, _ = evaluate_patient_level(model, test_loader)
            test_scores.append(macro_f1(t, p))
    return float(np.mean(val_scores)), float(np.std(val_scores)), float(np.mean(test_scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search CNN / CRNN / Transformer audio-encoder architectures.")
    parser.add_argument("--encoders", nargs="+", default=["CNN", "CRNN", "TRANSFORMER"],
                        choices=["CNN", "CRNN", "TRANSFORMER"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    run = method_run_dir("arch_search", "+".join(args.encoders))
    rows, best = [], {}
    for kind in args.encoders:
        for params in GRIDS[kind]:
            val_mean, val_std, test_mean = evaluate_config(kind, params, args.seeds, args.epochs)
            flat = {k: (str(v) if isinstance(v, tuple) else v) for k, v in params.items()}
            rows.append({"encoder": kind, **flat,
                         "val_macro_f1_mean": val_mean, "val_macro_f1_std": val_std,
                         "test_macro_f1_mean": test_mean})
            print(f"[{kind}] {params} -> val {val_mean:.4f}+/-{val_std:.4f} | test {test_mean:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(run / "arch_search.csv", index=False)  # incremental

    frame = pd.DataFrame(rows)
    frame.to_csv(run / "arch_search.csv", index=False)
    print("\n================ BEST ARCHITECTURES (selected on validation) ================")
    for kind in args.encoders:
        sub = frame[frame["encoder"] == kind].sort_values("val_macro_f1_mean", ascending=False)
        top = sub.iloc[0].to_dict()
        best[kind] = top
        param_str = ", ".join(f"{k}={top[k]}" for k in GRIDS[kind][0].keys())
        print(f"{kind}: val macro-F1 = {top['val_macro_f1_mean']:.4f} (test {top['test_macro_f1_mean']:.4f}) "
              f"| {param_str}")
    save_json(best, run / "best_architectures.json")
    print(f"\nFull table + best_architectures.json saved under: {run}")


if __name__ == "__main__":
    main()
