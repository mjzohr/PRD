# voice-omics — unpaired privileged transfer for COPD audio classification

Can an expensive privileged modality (COPDGene **omics** or **CT-clinical**)
improve a cheap respiratory-**audio** COPD classifier when the cohorts are
*unpaired* (different patients)? Audio, omics, and CT-clinical are treated as
different views of the same disease. **Inference is audio-only.**

- Audio: ICBHI, 3-class Healthy / COPD / Other, patient-level nested 5-fold CV.
- Privileged: COPDGene, binary Healthy / COPD, trained on **sharp** classes
  (GOLD 0 vs GOLD 3&4; GOLD 1&2 dropped) for a clean teacher signal.
- CWLSD: locked external cohort, evaluation only.

## Pipeline (each step is one clear script)

```
preprocessing.py        slice raw ICBHI audio -> 6s segments + patient-fold manifest
preprocess_cwlsd.py     slice CWLSD external audio (labels parsed from filenames)
train_teachers.py       Stage 1: omics / CT teacher (sharp) -> encoder + class/topic prototypes
run_bridge.py           audio -> teacher distillation targets (clinical value matching)
train_baseline_audio.py audio-only baseline (the bar every method must beat)
train_contrastive.py    text-anchored contrastive clinical pivot (label-aware CLIP)
train_distilled.py      masked feature distillation student
train_dtca.py           disease-topic consensus (audio guided by teacher prototypes)
report.py               aggregate results -> paper-ready tables
```

Foundation modules: `config.py` (all knobs), `models.py` (encoder factory +
heads), `losses.py`, `dataset.py`, `privileged.py` (view loading + sharp filter +
teacher preprocessing), `splitting.py` (leakage-safe CV), `utils.py`, `clinical_bridge.py`.

## Fair comparison (same audio encoder everywhere)

Every method — baseline, contrastive, distillation, DTCA — uses the **same audio
encoder** via `get_audio_encoder(encoder)`. The only thing that changes is the
cross-modal information added on top. So `results/reports/headline.md` reports
each method's macro-F1 **and its delta over the same-encoder audio-only baseline**.
Each method also has an architecture- and recipe-matched control that turns the
privileged signal off:

- distillation → `distill_weight 0` / baseline (pure CE, identical head)
- contrastive → `--loss_type audio_text_only` (drops the privileged modality;
  keeps the audio↔clinical-text term)
- DTCA → `--ablation audio_task_only` (same consensus head, guidance weight 0)

## Clinical-text encoder variants

The contrastive method's frozen clinical anchor is selectable via
`--clinical_encoder`: `bioclinicalbert` (default, 768-d), `gatortron` (1024-d),
`modernbert` (BioClinical ModernBERT, 768-d). Projection widths adapt to each.
Sweep them with `CLINICAL_ENCODERS="bioclinicalbert gatortron modernbert"`.

## Why this version actually transfers

The prior version throttled the privileged signal to near-zero (gradient
orthogonalisation + cosine gate + norm cap) and let the classifier bypass the
guided subspace. Here:

- **DTCA guidance is a plain ramped weighted term** — no PCGrad, no gate, no cap.
- The DTCA audio model **classifies through the shared subspace only**, so
  guidance on that subspace shapes the decision function directly.
- Disease semantics travel through the **clinical-text bridge** (prompt), not the
  age/sex tabular bridge.
- Privileged teachers use **sharp** classes.

## Hyperparameter sensitivity

One clear knob per method, swept on a small grid, selected on validation:

| Method | Knob | Grid |
|---|---|---|
| distillation | `distill_weight` | 0.5, 1, 2, 4 |
| contrastive | `temperature` | 0.05, 0.07, 0.1, 0.2 |
| DTCA | `guidance_weight` | 0.25, 0.5, 1, 2 |

`report.py` writes `results/reports/sensitivity_<method>.csv` and picks the best
knob per method by validation macro-F1.

## Run

```bash
# one quick pass (seed 42, CNN, both views)
bash scripts/run_all.sh

# full publication run
SEEDS="11 23 42 73 101" ENCODERS="CNN CRNN TRANSFORMER" \
CLINICAL_ENCODERS="bioclinicalbert gatortron modernbert" bash scripts/run_all.sh
```

Scripts read `python` from `$PYTHON_BIN` (default `python`) and run from this
directory. Outputs land in `results/<method>/<view>/<encoder>/<knob>/seed<seed>/`
with per-fold classification reports, confusion matrices, predictions, and
bootstrap CIs; `results/reports/headline.md` compares baseline vs each method.

## Ablations (`scripts/06_ablations.sh`)

distillation `shuffled_targets`; contrastive `audio_text_only` (matched control),
`vanilla`, `direct`; DTCA `audio_task_only` (matched control), `no_disease`,
`no_topic`, `shuffled_labels`, `random_features`. Each breaks one part of the
privileged signal to show it contributes.

## Data notes

- Omics/CT processed tables follow `configs/datasets.sh`
  (`data/copdgene/data_processed/...`); paths are overridable via env vars in
  `config.py`. Omics is server-side; CT-clinical and audio are usable locally.
- Sharp classes need a GOLD column (`finalGold`) in the privileged labels; without
  it, teachers fall back to the binary label with a printed warning.
