# Data

No data is included in this repository. The audio datasets are public but must be downloaded from their sources, and COPDGene is a controlled-access study.

| Dataset | Role | Access |
|---|---|---|
| ICBHI 2017 Respiratory Sound Database | Development (126 subjects) | https://bhichallenge.med.auth.gr/ |
| RespiratoryDatabase@TR | External test (42 subjects) | https://data.mendeley.com/datasets/p9z4h98s6j/1 |
| CWLSD | External test (112 subjects) | https://data.mendeley.com/datasets/jwyy9np4gv/3 |
| COPDGene (CT and multi-omics) | Privileged teachers | https://www.copdgene.org/ |

Organize the files as follows:

```text
data/
├── ICHBI/
│   ├── ICBHI_final_database/              .wav + .txt annotation pairs
│   ├── ICBHI_Challenge_demographic_information.txt
│   └── patient_diagnosis.csv
├── RespiratoryDatabase@TR/
├── CWLSD/
└── copdgene/
    └── data_processed/
        ├── clinical/
        │   ├── CLINICAL/imputed/X.csv
        │   └── CT/imputed/{X,y}.csv
        └── omics/formatted_data/concat/
            ├── pmr_concat_p2_copdgene.csv
            └── label.csv
```

Then run `bash scripts/launch.sh preprocess` to slice the recordings and build the manifests under `data/manifests/`.

Paths are set in [`prd/config.py`](../prd/config.py) and can be overridden with environment variables (`CT_X_CSV`, `CT_Y_CSV`, `OMICS_X_CSV`, `OMICS_Y_CSV`, `COPDGENE_CLINICAL_CSV`).

The audio experiments can be reproduced from the public datasets. Teacher checkpoints, teacher representations and bridge targets are derived from COPDGene subjects, so they are not distributed and are excluded by `.gitignore`.
