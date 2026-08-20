"""Optuna (TPE) search over the distillation hyper-parameters, per encoder+view.

Replaces naive grid search where we only need the best configuration (tuned
CRNN / Transformer students). Each trial trains the distilled student over the
nested patient folds of the search seed(s) and is scored by mean inner-
VALIDATION macro-F1 -- the outer test folds are never touched during search.
Unpromising trials are pruned per fold (median pruner). The final numbers must
come from re-running the best config on all publication seeds with
train_distilled.py (printed at the end).

Search space:
    distill_loss   categorical {cosine, mse, smooth_l1, relational}
    distill_weight log-uniform [0.5, 8.0]
    age_tolerance  categorical {3, 5, 7}   (bridge targets must already exist)

Run (bridges/teachers for the search seeds must exist, e.g. via PREP_ONLY):
    RESULTS_ROOT=results_distillation python tune_distilled.py \
        --encoder CRNN --view ct_clinical --trials 20
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError as exc:
    raise ImportError("tune_distilled.py needs optuna: uv pip install 'optuna>=4,<5'") from exc

from config import Config
from splitting import build_nested_patient_splits, select_patient_rows
from train_distilled import make_loader, resolve_target_path, train_one_fold
from utils import method_run_dir, save_json


def objective_factory(encoder: str, view: str, seeds, run_dir):
    development = pd.read_csv(Config.FULL_CSV)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "distill_loss": trial.suggest_categorical(
                "distill_loss", ["cosine", "mse", "smooth_l1", "relational"]
            ),
            "distill_weight": trial.suggest_float("distill_weight", 0.5, 8.0, log=True),
            "age_tolerance": trial.suggest_categorical("age_tolerance", [3.0, 5.0, 7.0]),
        }
        scores = []
        step = 0
        for seed in seeds:
            Config.set_seed(seed)
            args = SimpleNamespace(encoder=encoder, view=view, seed=seed, ablation="none", **params)
            target_path = resolve_target_path(view, "none", seed, params["age_tolerance"])
            _, splits = build_nested_patient_splits(
                development,
                folds=Config.NUM_FOLDS,
                repeat_seed=seed,
                validation_fraction=Config.INNER_VALIDATION_FRACTION,
            )
            for split in splits:
                train_frame = select_patient_rows(development, split.train_ids)
                val_frame = select_patient_rows(development, split.validation_ids)
                val_loader = make_loader(
                    val_frame, target_path, seed=seed, fold=split.outer_fold, train=False
                )
                checkpoint = run_dir / "checkpoints" / "tune_current_fold.pth"
                _, _, best_val = train_one_fold(
                    train_frame, target_path, args, split.outer_fold, checkpoint, val_loader
                )
                scores.append(float(best_val))
                step += 1
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search for the distilled student.")
    parser.add_argument("--encoder", required=True, choices=["CNN", "CRNN", "TRANSFORMER"])
    parser.add_argument("--view", required=True, choices=["omics", "ct_clinical"])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Search seeds (objective averages over them). Keep small; "
                             "confirm the winner on all publication seeds afterwards.")
    parser.add_argument("--sampler_seed", type=int, default=0)
    args = parser.parse_args()

    run = method_run_dir("tuning", f"{args.view}/{args.encoder}")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name=f"distilled_{args.view}_{args.encoder}",
    )
    study.optimize(
        objective_factory(args.encoder, args.view, args.seeds, run),
        n_trials=args.trials,
        show_progress_bar=False,
    )

    trials = study.trials_dataframe()
    trials.to_csv(run / "trials.csv", index=False)
    best = {
        "encoder": args.encoder,
        "view": args.view,
        "search_seeds": args.seeds,
        "n_trials": args.trials,
        "best_val_macro_f1": study.best_value,
        **study.best_params,
    }
    save_json(best, run / "best_params.json")

    print("\n================ BEST CONFIG (selected on inner validation) ================")
    print(f"{args.encoder} / {args.view}: val macro-F1 = {study.best_value:.4f}")
    print(f"  {study.best_params}")
    print("\nConfirm on all publication seeds (test folds untouched until now):")
    p = study.best_params
    for seed in Config.SEEDS:
        print(
            f"  python train_distilled.py --view {args.view} --encoder {args.encoder} "
            f"--distill_loss {p['distill_loss']} --distill_weight {p['distill_weight']:.3g} "
            f"--age_tolerance {p['age_tolerance']:g} --seed {seed}"
        )
    print(f"\nArtifacts: {run}/trials.csv, {run}/best_params.json")


if __name__ == "__main__":
    main()
