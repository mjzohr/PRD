#!/usr/bin/env python
"""Print the training commands for each experiment grid, skipping runs that already exist."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(os.environ.get("PRD_ROOT", Path(__file__).resolve().parents[1])).resolve()
RESULTS = ROOT / os.environ.get("RESULTS_ROOT", "results")
TARGETS = ROOT / "data" / "distillation"

SEEDS = [11, 23, 42, 73, 101, 3, 5, 7, 13, 19]
VIEWS = ["ct_clinical", "omics"]
ENCODERS = ["CNN", "CRNN", "TRANSFORMER"]
LOSSES = ["cosine", "mse", "smooth_l1", "relational"]
WEIGHT = "1.0"
AGE = 5

# sensitivity sweeps use the CNN student, CT teacher and the first five seeds
SENS_SEEDS = SEEDS[:5]
SENS_WEIGHTS = ["0.5", "1.0", "2.0", "4.0", "8.0"]
SENS_AGES = [3, 7]

# ablation operating point; controls are ordered by how much of the target they remove
ABL_VIEW, ABL_LOSS = "ct_clinical", "smooth_l1"
ABLATIONS = ["none", "class_prototype", "shuffled_targets", "random_prototype",
             "crossed_targets", "global_prototype"]


def done_distilled(view, enc, age, loss, w, seed, abl="none"):
    tag = f"{view}/{enc}/age{int(age)}/dl_{loss}/w{w}/seed{seed}"
    if abl != "none":
        tag += f"/abl_{abl}"
    return (RESULTS / "distilled" / tag / "metrics" / "summary_metrics.json").is_file()


def done_baseline(enc, seed):
    return (RESULTS / "baseline_audio" / enc / f"seed{seed}" / "metrics" / "summary_metrics.json").is_file()


def done_teacher(view, seed):
    return (RESULTS / "teachers" / view / f"seed{seed}" / "metrics" / "teacher_metrics.json").is_file()


def done_bridge(view, age, seed, random_teacher=False):
    tag = "_randinit" if random_teacher else ""
    return (TARGETS / f"{view}_targets{tag}_age{int(age)}_seed{seed}.npy").is_file()


def distill(view, enc, age, loss, w, seed, abl="none"):
    cmd = (f"python scripts/train_distilled.py --view {view} --encoder {enc} --age_tolerance {age} "
           f"--distill_loss {loss} --distill_weight {w} --seed {seed}")
    return cmd if abl == "none" else f"{cmd} --ablation {abl}"


def baselines():
    return [f"python scripts/train_baseline.py --encoder {enc} --seed {seed}"
            for seed in SEEDS for enc in ENCODERS if not done_baseline(enc, seed)]


def prep_jobs():
    jobs = []
    for view in VIEWS:
        for seed in SEEDS:
            ages = [AGE] + (SENS_AGES if view == "ct_clinical" and seed in SENS_SEEDS else [])
            parts = []
            if not done_teacher(view, seed):
                parts.append(f"python scripts/train_teachers.py --view {view} --seed {seed}")
            parts += [f"python scripts/bridge.py --view {view} --seed {seed} --age_tolerance {age}"
                      for age in ages if not done_bridge(view, age, seed)]
            if parts:
                jobs.append(" && ".join(parts))
    return jobs


def main_jobs():
    # ordered by seed so a partial run still covers every cell evenly
    jobs = baselines()
    for seed in SEEDS:
        for view in VIEWS:
            for enc in ENCODERS:
                for loss in LOSSES:
                    if not done_distilled(view, enc, AGE, loss, WEIGHT, seed):
                        jobs.append(distill(view, enc, AGE, loss, WEIGHT, seed))
    return jobs


def sensitivity_jobs():
    jobs = []
    for seed in SENS_SEEDS:
        for loss in LOSSES:
            for w in SENS_WEIGHTS:
                if not done_distilled("ct_clinical", "CNN", AGE, loss, w, seed):
                    jobs.append(distill("ct_clinical", "CNN", AGE, loss, w, seed))
            for age in SENS_AGES:
                if not done_distilled("ct_clinical", "CNN", age, loss, WEIGHT, seed):
                    jobs.append(distill("ct_clinical", "CNN", age, loss, WEIGHT, seed))
    return jobs


def ablation_jobs():
    jobs = baselines()
    for seed in SEEDS:
        for enc in ENCODERS:
            for abl in ABLATIONS:
                if not done_distilled(ABL_VIEW, enc, AGE, ABL_LOSS, WEIGHT, seed, abl):
                    jobs.append(distill(ABL_VIEW, enc, AGE, ABL_LOSS, WEIGHT, seed, abl))
    return jobs


def random_teacher_prep_jobs():
    return [f"python scripts/bridge.py --view {ABL_VIEW} --seed {seed} --age_tolerance {AGE} --random_teacher"
            for seed in SEEDS if not done_bridge(ABL_VIEW, AGE, seed, random_teacher=True)]


def random_teacher_jobs():
    return [distill(ABL_VIEW, enc, AGE, ABL_LOSS, WEIGHT, seed, "random_teacher")
            for seed in SEEDS for enc in ENCODERS
            if not done_distilled(ABL_VIEW, enc, AGE, ABL_LOSS, WEIGHT, seed, "random_teacher")]


PHASES = {
    "prep": prep_jobs,
    "main": main_jobs,
    "sensitivity": sensitivity_jobs,
    "ablation": ablation_jobs,
    "random_teacher_prep": random_teacher_prep_jobs,
    "random_teacher": random_teacher_jobs,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=[*PHASES, "count"])
    args = p.parse_args()
    if args.phase == "count":
        for name, fn in PHASES.items():
            print(f"{name:20s} {len(fn())}")
        return
    for job in PHASES[args.phase]():
        print(job)


if __name__ == "__main__":
    main()
