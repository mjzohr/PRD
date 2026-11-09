"""Paths, audio front-end, training settings and method hyperparameters."""

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(
    os.environ.get("PRD_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()


class Config:
    PROJECT_ROOT = _ROOT
    RAW_DATA_ROOT = PROJECT_ROOT / "data" / "ICHBI"
    AUDIO_DIR = RAW_DATA_ROOT / "ICBHI_final_database"
    DEMO_FILE = RAW_DATA_ROOT / "ICBHI_Challenge_demographic_information.txt"
    DIAG_FILE = RAW_DATA_ROOT / "patient_diagnosis.csv"

    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_audio" / "ichbi"
    EXTERNAL_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_audio" / "cwlsd"
    FULL_CSV = PROJECT_ROOT / "data" / "manifests" / "ichbi_slices_with_folds.csv"
    EXTERNAL_CSV = PROJECT_ROOT / "data" / "manifests" / "cwlsd_external_slices.csv"

    # set RESULTS_ROOT to keep separate studies in separate trees
    RESULTS_DIR = PROJECT_ROOT / os.environ.get("RESULTS_ROOT", "results")

    CLASSES = ["Healthy", "COPD", "Other"]
    NUM_CLASSES = len(CLASSES)
    CLASS_MAPPING = {
        "COPD": "COPD",
        "Healthy": "Healthy",
        "Asthma": "Other",
        "Bronchiectasis": "Other",
        "Bronchiolitis": "Other",
        "LRTI": "Other",
        "Pneumonia": "Other",
        "URTI": "Other",
    }
    # only these classes have a privileged counterpart; Other is masked out of
    # every transfer term
    PRIVILEGED_CLASSES = ("Healthy", "COPD")

    SAMPLE_RATE = 22050
    DURATION = 6
    OVERLAP = 0.5
    STEP_SIZE = int(DURATION * (1 - OVERLAP) * SAMPLE_RATE)
    TARGET_LEN = SAMPLE_RATE * DURATION
    N_MELS = 64
    N_MFCC = 40
    N_FFT = 1024
    HOP_LEN = 512

    # Two front-ends. A (default) is the standard full-band ICBHI setup.
    # B (AUDIO_F_MAX=2000 NORMALIZE_FEATURES=1) is for cross-cohort transfer:
    # it limits both corpora to the band the external recordings cover and
    # applies per-channel CMVN to cancel device and gain offsets.
    _F_MAX_ENV = os.environ.get("AUDIO_F_MAX", "")
    F_MAX = float(_F_MAX_ENV) if _F_MAX_ENV else None  # None means Nyquist
    NORMALIZE_FEATURES = os.environ.get("NORMALIZE_FEATURES", "0") == "1"
    # mel filters inside the MFCC transform: at 2 kHz only ~93 FFT bins remain,
    # too few for 128 filters, so the band-limited front-end uses 64
    MFCC_N_MELS = int(os.environ.get("MFCC_N_MELS", "128" if F_MAX is None else "64"))
    AUDIO_INPUT_CHANNELS = N_MELS + N_MFCC
    AUDIO_FEATURE_DIM = 512

    AUDIO_ENCODER = os.environ.get("AUDIO_ENCODER", "CNN").upper()

    CLINICAL_ENCODER = os.environ.get("CLINICAL_ENCODER", "bioclinicalbert")
    CLINICAL_ENCODER_CHOICES = ("bioclinicalbert", "gatortron", "modernbert")

    SHARED_DIM = 128
    PRIVATE_DIM = 128

    SEED = int(os.environ.get("AUDIO_SEED", "42"))
    SEEDS = (11, 23, 42, 73, 101)
    NUM_FOLDS = 5
    INNER_VALIDATION_FRACTION = float(os.environ.get("INNER_VALIDATION_FRACTION", "0.15"))
    BATCH_SIZE = 32
    DEVICE = torch.device(
        os.environ.get("PRD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    )
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
    PIN_MEMORY = torch.cuda.is_available()
    AGE_TOLERANCE = 5.0  # years, for bridge matching

    ENCODER_LR = 1e-5
    HEAD_LR = 1e-3
    BASELINE_ENCODER_LR = 1e-4  # the baseline trains its encoder from scratch
    FINETUNE_EPOCHS = 30
    PATIENCE = 8
    MIN_EPOCHS = 12
    BOOTSTRAP_SAMPLES = 2000

    @staticmethod
    def set_seed(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        # avoids thread contention when many runs share a machine
        torch.set_num_threads(1)

    @staticmethod
    def get_generator(seed=SEED):
        g = torch.Generator()
        g.manual_seed(int(seed))
        return g

    @staticmethod
    def check_audio_paths():
        for path in (Config.AUDIO_DIR, Config.DEMO_FILE, Config.DIAG_FILE):
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing required audio input: {path}")
        Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


_CG = Config.PROJECT_ROOT / "data" / "copdgene" / "data_processed"


@dataclass(frozen=True)
class PrivilegedView:
    """One privileged tabular modality."""

    name: str
    x_csv: Path
    y_csv: Path
    clinical_csv: Path
    index_col: str = "sid"
    # the GOLD stage (0-4) gives both the label and the sharp-class filter
    label_col: str = "finalgold_visit"
    gold_col: str = "finalgold_visit"
    max_features: int = 1000  # ANOVA top-k, fit on teacher training rows


OMICS_VIEW = PrivilegedView(
    name="omics",
    x_csv=Path(os.environ.get("OMICS_X_CSV", _CG / "omics/formatted_data/concat/pmr_concat_p2_copdgene.csv")),
    y_csv=Path(os.environ.get("OMICS_Y_CSV", _CG / "omics/formatted_data/concat/label.csv")),
    clinical_csv=Path(os.environ.get("COPDGENE_CLINICAL_CSV", _CG / "clinical/CLINICAL/imputed/X.csv")),
)

CT_VIEW = PrivilegedView(
    name="ct_clinical",
    x_csv=Path(os.environ.get("CT_X_CSV", _CG / "clinical/CT/imputed/X.csv")),
    y_csv=Path(os.environ.get("CT_Y_CSV", _CG / "clinical/CT/imputed/y.csv")),
    clinical_csv=Path(os.environ.get("COPDGENE_CLINICAL_CSV", _CG / "clinical/CLINICAL/imputed/X.csv")),
    max_features=134,  # all CT-clinical features
)

PRIVILEGED_VIEWS = {view.name: view for view in (OMICS_VIEW, CT_VIEW)}


class Sharp:
    """Sharp teacher classes: GOLD 0 as Healthy, GOLD 3-4 as COPD, GOLD 1-2 dropped."""

    ENABLE = os.environ.get("SHARP_CLASSES", "1") == "1"
    HEALTHY_GOLD = (0,)
    COPD_GOLD = (3, 4)


class Hparams:
    # masked distillation
    DISTILL_WEIGHT = 2.0
    DISTILL_WEIGHT_GRID = (0.5, 1.0, 2.0, 4.0)
    # no warm-up ramp: with early stopping, the selected epoch could otherwise
    # fall before the term reaches full weight
    DISTILL_LOSS = "cosine"
    DISTILL_LOSS_GRID = ("cosine", "mse", "smooth_l1", "relational")
    AGE_TOLERANCE_GRID = (3.0, 5.0, 7.0)

    # contrastive clinical pivot
    TEMPERATURE = 0.07
    TEMPERATURE_GRID = (0.05, 0.07, 0.1, 0.2)
    CONTRASTIVE_EPOCHS = 15
    PROJECTION_DIM = 128

    # disease-topic consensus (DTCA)
    GUIDANCE_WEIGHT = 1.0
    GUIDANCE_WEIGHT_GRID = (0.25, 0.5, 1.0, 2.0)
    GUIDANCE_RAMPUP_EPOCHS = 6
    NUM_TOPICS = 4  # topic prototypes per class
    TOPIC_TEMPERATURE = 0.5
