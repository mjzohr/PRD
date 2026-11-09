# Privileged Representation Distillation (PRD)

PRD transfers task-relevant representations from an information-rich but costly privileged modality to a cheap, deployable one, without requiring both to be observed for the same subjects.

We apply PRD to COPD. Respiratory audio is the deployable modality; chest CT and blood multi-omics from COPDGene are the privileged modalities. The two come from different patient populations, so no subject has both, and inference uses audio alone.

![PRD framework](PRD_framework.png)

A teacher is trained on the privileged data. A semantic bridge groups privileged subjects by variables both datasets record (diagnosis, sex, age) and reduces each group to a single target vector. The audio student is trained with its usual classification loss plus a masked distillation term toward the matched targets; recordings whose class has no privileged counterpart are masked out.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Results were produced with Python 3.11, PyTorch 2.13 and torchaudio 2.11 on NVIDIA H200 GPUs.

## Data

No data is included. See [`data/README.md`](data/README.md) for what to download and where to put it. The audio datasets are public; COPDGene is controlled-access, so the privileged teachers cannot be trained without a data-use agreement.

## Run

```bash
bash scripts/launch.sh preprocess       # slice audio, build manifests and feature cache
bash scripts/launch.sh quick            # one seed: audio-only baseline vs PRD
bash scripts/launch.sh main             # full grid over teachers, encoders, losses and seeds
bash scripts/launch.sh ablation         # signal-specificity controls
bash scripts/launch.sh random_teacher   # PRD with an untrained teacher
bash scripts/launch.sh report           # tables and figures
```

Or run individual stages:

```bash
python scripts/train_teachers.py --view ct_clinical --seed 42
python scripts/bridge.py --view ct_clinical --seed 42 --age_tolerance 5
python scripts/train_baseline.py --encoder CNN --seed 42
python scripts/train_distilled.py --view ct_clinical --encoder CNN --seed 42 \
    --distill_loss smooth_l1 --distill_weight 1.0 --age_tolerance 5
python scripts/report.py
```

The grid runs are resumable: `scripts/jobs.py` lists only the runs that have not finished yet, and `launch.sh` spreads them over `GPUS` devices. Outputs go to `$RESULTS_ROOT` (default `results/`).

Audio models are evaluated with nested, patient-stratified 5-fold cross-validation on ICBHI and scored without retraining on two external cohorts (RespiratoryDatabase@TR and CWLSD). The default front end uses the full band without normalization. For cross-cohort deployment, set `AUDIO_F_MAX=2000 NORMALIZE_FEATURES=1`, which matches the analysis band of the external recordings and applies per-channel normalization.

## Ablations

Each control in `--ablation` removes part of what the distillation targets carry:

| Ablation | Target |
|---|---|
| `none` | Full bridge (diagnosis + sex + age groups) |
| `class_prototype` | Diagnosis-only groups |
| `shuffled_targets` | Targets permuted within class |
| `random_prototype` | Random norm-matched vector per class |
| `crossed_targets` | Real targets from the opposite class |
| `global_prototype` | One target for every recording |
| `random_teacher` | Full bridge from an untrained teacher encoder |

`train_contrastive.py` and `train_dtca.py` implement two alternative transfer methods (clinical-text contrastive alignment and disease-topic consensus) that use the same audio encoders and splits.

## Repository structure

```text
prd/                     Config, datasets, models, losses, splits, metrics
scripts/
  preprocess_*.py        Slice ICBHI, TR and CWLSD recordings into segments
  match_cwlsd.py         CWLSD subset restricted to diagnoses seen in training
  cache_features.py      Precompute mel + MFCC features
  train_teachers.py      Privileged CT / omics teachers
  bridge.py              Group matching and distillation targets
  train_baseline.py      Audio-only baseline
  train_distilled.py     PRD student
  train_contrastive.py   Contrastive transfer
  train_dtca.py          Disease-topic consensus transfer
  tune_distilled.py      Hyperparameter search for the student
  arch_search.py         CRNN / Transformer architecture search
  evaluate_external.py   External-cohort evaluation from saved checkpoints
  report.py              Result tables and paired comparisons
  tables.py              Teacher, ablation and random-teacher tables
  figures.py             Sensitivity figures
  audit_*.py             Checks on targets, shuffling and recording devices
  jobs.py                Outstanding runs for each experiment grid
  launch.sh              Experimental workflows
```
