# Data

No data is committed to this repository. The datasets used in this study are
either third-party datasets that we do not have redistribution rights for or
information-rich data that are not released with this repository. The audio
data alone also exceed 2 GB.

Download the public audio datasets separately and place them as shown below.
For the information-rich CT and multi-omics data, use the corresponding local
files available under your authorized data access. Then run
`python voice_omics/preprocessing.py` to build the required manifests.

```text
data/
  ICHBI/
    ICBHI_final_database/                      # .wav + .txt annotation pairs
    ICBHI_Challenge_demographic_information.txt
    patient_diagnosis.csv
  RespiratoryDatabase@TR/
  CWLSD/
  information_rich/
    data_processed/
      clinical/CT/imputed/{X,y}.csv
      clinical/CLINICAL/imputed/X.csv
      omics/formatted_data/concat/{omics_concat,label}.csv
```

Paths are configured in `configs/datasets.sh` and `voice_omics/config.py`, and
can be overridden with environment variables such as `CT_X_CSV`,
`OMICS_X_CSV`, and `CLINICAL_X_CSV` if the local layout differs.

## Sources

| Dataset                                  | Role                         | Access                                          |
| ---------------------------------------- | ---------------------------- | ----------------------------------------------- |
| ICBHI 2017 Respiratory Sound Database    | development (126 subjects)   | public — https://bhichallenge.med.auth.gr/      |
| RespiratoryDatabase@TR                   | external test (42 subjects)  | https://data.mendeley.com/datasets/p9z4h98s6j/1 |
| CWLSD                                    | external test (112 subjects) | https://data.mendeley.com/datasets/jwyy9np4gv/3 |
| Information-rich CT and multi-omics data | privileged teachers          | not distributed with this repository            |

Verify the external-dataset links against the locations from which the data
were obtained before publishing.

## Reproducibility

The audio experiments are reproducible from the publicly available datasets.
The information-rich arm requires access to the corresponding CT and
multi-omics data and is therefore not directly reproducible from the repository
alone. Derived artifacts, including teacher checkpoints, teacher
representations, and bridge targets, are not distributed and are excluded by
`.gitignore`.

`run_bridge.py` and `train_teachers.py` are included so that the complete
procedure can be inspected and re-run by users with access to the required
information-rich data.

Aggregate experimental results contain no subject-level information and can be
shared.
