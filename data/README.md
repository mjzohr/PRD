# Data

No data is committed to this repository. Three of the four corpora are
third-party redistributions we have no right to publish, COPDGene is
controlled-access, and the audio alone exceeds 2 GB.

Download each source yourself and place it as below, then run
`python voice_omics/preprocessing.py` to build the slice manifests.

```
data/
  ICHBI/
    ICBHI_final_database/                      # .wav + .txt annotation pairs
    ICBHI_Challenge_demographic_information.txt
    patient_diagnosis.csv
  RespiratoryDatabase@TR/
  CWLSD/
  copdgene/
    data_processed/
      clinical/CT/imputed/{X,y}.csv
      clinical/CLINICAL/imputed/X.csv
      omics/formatted_data/concat/{pmr_concat_p2_copdgene,label}.csv
```

Paths are configured in `configs/datasets.sh` and `voice_omics/config.py`, and
can be overridden with environment variables (`CT_X_CSV`, `OMICS_X_CSV`,
`COPDGENE_CLINICAL_CSV`) if your layout differs.

## Sources

| Corpus | Role | Access |
| --- | --- | --- |
| ICBHI 2017 Respiratory Sound Database | development (126 patients) | public — <https://bhichallenge.med.auth.gr/> |
| RespiratoryDatabase@TR | external test (42 patients) | https://data.mendeley.com/datasets/p9z4h98s6j/1 |
| CWLSD | external test (112 patients) | https://data.mendeley.com/datasets/jwyy9np4gv/3 |
| COPDGene | privileged teachers (CT, multi-omics) | **controlled access** — <https://www.copdgene.org/>, via dbGaP |

Verify the two `TODO` links against wherever you actually obtained those
corpora before publishing; do not rely on them being correct here.

## Reproducibility

The audio experiments are fully reproducible from public data. The privileged
arm is not: COPDGene requires a data-use agreement, and the derived artifacts
(teacher checkpoints, teacher representations, bridge targets) are
subject-derived and are therefore excluded by `.gitignore` rather than
released. `run_bridge.py` and `train_teachers.py` are included so the procedure
can be inspected and re-run by anyone holding the data.

Aggregate results contain no subject-level information and can be shared.
