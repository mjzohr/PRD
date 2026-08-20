"""Emit the exact command list needed to finish the paper, in priority order.

Everything already computed is skipped, so the run is resumable and does not
redo the ~120 finished runs. Scope is deliberately narrow: only the cells the
write-up still needs, at the standard five publication seeds.

Priority (highest first -- if you run out of time, the top blocks are the ones
that matter):
  1. SIGNAL SPECIFICITY  shuffled-target ablation at the primary configs.
  2. REPLICATION         CNN x omics (independent teacher) at the primary grid.
  3. SENSITIVITY (age)   CNN x CT at tolerance +/-7y, completing the age axis.
  4. TABLE 1 BREADTH     tuned CRNN / Transformer students, small sweep.

Usage (called by scripts/run_paper_final.sh):
    python make_paper_jobs.py --phase prep  > jobs_prep.txt
    python make_paper_jobs.py --phase train > jobs_train.txt
    python make_paper_jobs.py --phase count
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(os.environ.get("VOICE_OMICS_ROOT", Path(__file__).resolve().parents[1])).resolve()
RESULTS = ROOT / os.environ.get("RESULTS_ROOT", "results_distillation")
DIST = ROOT / "data" / "distillation"

PUB_SEEDS = [11, 23, 42, 73, 101]
# Confirmatory seeds: take the reported claims from n=5 to n=10. Fixed in advance
# and reported regardless of outcome.
EXTRA_SEEDS = [3, 5, 7, 13, 19]
ALL_LOSSES = ["cosine", "mse", "smooth_l1", "relational"]

# Confirmatory grid: a FULLY SYMMETRIC factorial evaluated at all ten seeds.
# Deliberately no pre-selection of encoder, view, or loss -- picking configurations
# that looked strong at n=5 would bake an n=5 conclusion into the n=10 evidence and
# leave the architectures unequally covered. Every cell gets identical treatment,
# so whichever architecture wins, the comparison behind it is balanced.
CONFIRM_VIEWS = ["ct_clinical", "omics"]
CONFIRM_ENCODERS = ["CNN", "CRNN", "TRANSFORMER"]
CONFIRM_LOSSES = ALL_LOSSES
CONFIRM_WEIGHT = "1.0"        # single canonical operating point; the weight sweep
                              # remains exploratory at n=5
CONFIRM_AGE = 5
CONFIRM_ABLATION_LOSSES = ["mse", "smooth_l1"]
CONFIRM_ABLATIONS = ["shuffled_targets", "class_prototype", "random_prototype"]
ALL_SEEDS = PUB_SEEDS + EXTRA_SEEDS
PRIMARY_LOSSES = ["mse", "smooth_l1"]      # magnitude-aware family that won
PRIMARY_WEIGHTS = ["0.5", "1.0"]
WEIGHT_GRID = ["0.5", "1.0", "2.0", "4.0", "8.0"]   # sensitivity weight axis
PRIMARY_AGE = 5
SENS_WEIGHT = "1.0"                         # operating point for the age axis


# ----------------------------- existence checks -----------------------------
def has_distilled(view, enc, age, loss, w, seed, abl="none") -> bool:
    tag = f"{view}/{enc}/age{int(age)}/dl_{loss}/w{w}/seed{seed}"
    if abl != "none":
        tag += f"/abl_{abl}"
    return (RESULTS / "distilled" / tag / "metrics" / "summary_metrics.json").is_file()


def has_teacher(view, seed) -> bool:
    return (RESULTS / "teachers" / view / f"seed{seed}" /
            "metrics" / "teacher_metrics.json").is_file()


def has_bridge(view, age, seed) -> bool:
    return (DIST / f"{view}_targets_age{int(age)}_seed{seed}.npy").is_file()


# --------------------------------- job lists --------------------------------
def prep_jobs() -> list[str]:
    """Backfill any missing teacher/bridge for the cells below (normally empty:
    the earlier PREP_ONLY phase already built all of them)."""
    jobs = []
    for view in ("ct_clinical", "omics"):
        for seed in PUB_SEEDS:
            parts = []
            if not has_teacher(view, seed):
                parts.append(f"python train_teachers.py --view {view} --seed {seed}")
            for age in (PRIMARY_AGE, 7):
                if not has_bridge(view, age, seed):
                    parts.append(f"python run_bridge.py --view {view} --seed {seed} "
                                 f"--age_tolerance {age}")
            if parts:
                jobs.append(" && ".join(parts))
    return jobs


def _distill_cmd(view, enc, age, loss, w, seed, abl="none") -> str:
    cmd = (f"python train_distilled.py --view {view} --encoder {enc} "
           f"--age_tolerance {age} --distill_loss {loss} --distill_weight {w} --seed {seed}")
    if abl != "none":
        cmd += f" --ablation {abl}"
    return cmd


def has_baseline(enc, seed) -> bool:
    return (RESULTS / "baseline_audio" / enc / f"seed{seed}" /
            "metrics" / "summary_metrics.json").is_file()


def train_jobs() -> list[str]:
    jobs: list[str] = []

    # 0. Baselines -- the reference for every delta. Must share the current
    #    protocol (val-loss selection, patience, min-epochs) with the students.
    for seed in PUB_SEEDS:
        for enc in ("CNN", "CRNN", "TRANSFORMER"):
            if not has_baseline(enc, seed):
                jobs.append(f"python train_baseline_audio.py --encoder {enc} --seed {seed}")

    # 1. Signal specificity: within-class shuffle (removes demographic
    #    correspondence) and class prototypes (removes everything but class).
    for seed in PUB_SEEDS:
        for loss in PRIMARY_LOSSES:
            for ablation in ("shuffled_targets", "class_prototype", "random_prototype"):
                if not has_distilled("ct_clinical", "CNN", PRIMARY_AGE, loss,
                                     SENS_WEIGHT, seed, ablation):
                    jobs.append(_distill_cmd("ct_clinical", "CNN", PRIMARY_AGE, loss,
                                             SENS_WEIGHT, seed, ablation))

    # 2. Replication with the independent omics teacher.
    for seed in PUB_SEEDS:
        for loss in ALL_LOSSES:
            for w in PRIMARY_WEIGHTS:
                if not has_distilled("omics", "CNN", PRIMARY_AGE, loss, w, seed):
                    jobs.append(_distill_cmd("omics", "CNN", PRIMARY_AGE, loss, w, seed))

    # 3. CNN sensitivity grid (the paper's Fig. 4).
    #    (a) weight axis: every loss across the full weight grid at the canonical
    #        tolerance; (b) age axis: every loss at the operating weight, all tolerances.
    for seed in PUB_SEEDS:
        for loss in ALL_LOSSES:
            for w in WEIGHT_GRID:
                if not has_distilled("ct_clinical", "CNN", PRIMARY_AGE, loss, w, seed):
                    jobs.append(_distill_cmd("ct_clinical", "CNN", PRIMARY_AGE, loss, w, seed))
    for seed in PUB_SEEDS:
        for loss in ALL_LOSSES:
            for age in (3, 7):
                if not has_distilled("ct_clinical", "CNN", age, loss, SENS_WEIGHT, seed):
                    jobs.append(_distill_cmd("ct_clinical", "CNN", age, loss, SENS_WEIGHT, seed))

    # 4. Table 1 breadth: tuned CRNN / Transformer students.
    for seed in PUB_SEEDS:
        for enc in ("CRNN", "TRANSFORMER"):
            for loss in PRIMARY_LOSSES:
                for w in PRIMARY_WEIGHTS:
                    if not has_distilled("ct_clinical", enc, PRIMARY_AGE, loss, w, seed):
                        jobs.append(_distill_cmd("ct_clinical", enc, PRIMARY_AGE, loss, w, seed))
    return jobs


def confirm_prep_jobs() -> list[str]:
    """Teachers + age-5 bridges for every seed of the confirmatory grid."""
    jobs = []
    for view in CONFIRM_VIEWS:
        for seed in ALL_SEEDS:
            parts = []
            if not has_teacher(view, seed):
                parts.append(f"python train_teachers.py --view {view} --seed {seed}")
            if not has_bridge(view, CONFIRM_AGE, seed):
                parts.append(f"python run_bridge.py --view {view} --seed {seed} "
                             f"--age_tolerance {CONFIRM_AGE}")
            if parts:
                jobs.append(" && ".join(parts))
    return jobs


def confirm_jobs() -> list[str]:
    """Symmetric confirmatory factorial over all seeds (existing runs skipped).

    Ordered baselines -> main grid -> ablations, and within each block by seed,
    so an interrupted run still yields balanced coverage across architectures
    rather than completing one encoder before starting another.
    """
    jobs: list[str] = []
    for seed in ALL_SEEDS:
        for enc in CONFIRM_ENCODERS:
            if not has_baseline(enc, seed):
                jobs.append(f"python train_baseline_audio.py --encoder {enc} --seed {seed}")
    for seed in ALL_SEEDS:
        for view in CONFIRM_VIEWS:
            for enc in CONFIRM_ENCODERS:
                for loss in CONFIRM_LOSSES:
                    if not has_distilled(view, enc, CONFIRM_AGE, loss, CONFIRM_WEIGHT, seed):
                        jobs.append(_distill_cmd(view, enc, CONFIRM_AGE, loss,
                                                 CONFIRM_WEIGHT, seed))
    for seed in ALL_SEEDS:
        for enc in CONFIRM_ENCODERS:
            for loss in CONFIRM_ABLATION_LOSSES:
                for ablation in CONFIRM_ABLATIONS:
                    if not has_distilled("ct_clinical", enc, CONFIRM_AGE, loss,
                                         CONFIRM_WEIGHT, seed, ablation):
                        jobs.append(_distill_cmd("ct_clinical", enc, CONFIRM_AGE, loss,
                                                 CONFIRM_WEIGHT, seed, ablation))
    return jobs


# ------------------------- signal-specificity ablation ----------------------
# Deliberately narrow: ONE teacher, ONE loss, ONE weight, ONE tolerance, Config A.
# The only thing that varies is what the target carries, which is the question.
# Adding a second loss or teacher would double the cost without adding evidence,
# since the main table already fixes smooth-L1 at w=1, tau=5 by validation.
ABL_VIEW = "ct_clinical"
ABL_LOSS = "smooth_l1"
ABL_WEIGHT = "1.0"
ABL_AGE = 5
# Ordered by how much they remove. The first four already exist at ten seeds; the
# last two are what audit_shuffle.py showed to be missing -- every earlier control
# leaves the label-to-target association intact, so none of them can separate
# privileged structure from a class-conditional centre loss.
ABL_LADDER = [
    "none",               # full bridge: label + sex + age
    "class_prototype",    # label-only groups
    "shuffled_targets",   # within-class permutation (weak: see audit_shuffle.py)
    "random_prototype",   # class-conditional constant, no link to the cohort
    "crossed_targets",    # real targets from the opposite class
    "global_prototype",   # one target for everything: no class dependence at all
]


def ablation_jobs() -> list[str]:
    """Signal-specificity ladder at the confirmatory operating point, all seeds.

    Ordered by seed so an interrupted run still covers every condition evenly
    rather than finishing one ablation before starting the next.
    """
    jobs: list[str] = []
    for seed in ALL_SEEDS:
        for enc in CONFIRM_ENCODERS:
            if not has_baseline(enc, seed):
                jobs.append(f"python train_baseline_audio.py --encoder {enc} --seed {seed}")
            for ablation in ABL_LADDER:
                if not has_distilled(ABL_VIEW, enc, ABL_AGE, ABL_LOSS,
                                     ABL_WEIGHT, seed, ablation):
                    jobs.append(_distill_cmd(ABL_VIEW, enc, ABL_AGE, ABL_LOSS,
                                             ABL_WEIGHT, seed, ablation))
    return jobs


# ---------------------- random-init teacher control -------------------------
# The exact-match control for this method. Everything the pipeline does is kept:
# real privileged inputs, the fitted preprocessor, label+sex+age value matching,
# group averaging, the same 37-way target structure. The ONLY difference is that
# the teacher encoder is left at initialisation instead of trained, so a null
# result means teacher training -- not the auxiliary loss, not the bridge -- is
# what fails to matter. Internal (ICBHI) only, and no external scoring.


def random_teacher_prep_jobs() -> list[str]:
    """Bridges built from an UNTRAINED teacher encoder.

    Split from the student phase because the two need different RESULTS_ROOTs:
    this reads the fitted teacher/preprocessor out of the existing root, while
    the students below write into a fresh one. Targets land in data/distillation,
    which is independent of RESULTS_ROOT, so the handover needs no copying.
    """
    return [f"python run_bridge.py --view {ABL_VIEW} --seed {seed} "
            f"--age_tolerance {ABL_AGE} --random_teacher"
            for seed in ALL_SEEDS
            if not (DIST / f"{ABL_VIEW}_targets_randinit_age{ABL_AGE}_seed{seed}.npy").is_file()]


def random_teacher_jobs() -> list[str]:
    """The three students at all seeds, against the random-init targets."""
    return [_distill_cmd(ABL_VIEW, enc, ABL_AGE, ABL_LOSS, ABL_WEIGHT, seed, "random_teacher")
            for seed in ALL_SEEDS for enc in CONFIRM_ENCODERS
            if not has_distilled(ABL_VIEW, enc, ABL_AGE, ABL_LOSS, ABL_WEIGHT,
                                 seed, "random_teacher")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit paper-completion job list.")
    parser.add_argument("--phase", required=True,
                        choices=["prep", "train", "count", "confirm_prep", "confirm",
                                 "ablation", "random_teacher_prep", "random_teacher"])
    args = parser.parse_args()
    if args.phase == "count":
        print(f"prep jobs         : {len(prep_jobs())}")
        print(f"train jobs        : {len(train_jobs())}")
        print(f"confirm prep jobs : {len(confirm_prep_jobs())}")
        print(f"confirm jobs      : {len(confirm_jobs())}")
        print(f"ablation jobs     : {len(ablation_jobs())}")
        print(f"random-teacher prep: {len(random_teacher_prep_jobs())}")
        print(f"random-teacher    : {len(random_teacher_jobs())}")
        return
    emit = {"prep": prep_jobs, "train": train_jobs,
            "confirm_prep": confirm_prep_jobs, "confirm": confirm_jobs,
            "ablation": ablation_jobs,
            "random_teacher_prep": random_teacher_prep_jobs,
            "random_teacher": random_teacher_jobs}[args.phase]
    for job in emit():
        print(job)


if __name__ == "__main__":
    main()
