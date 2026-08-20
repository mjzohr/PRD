"""Central configuration. All knobs are visible here -- no env-var sprawl.

Environment overrides exist only for the few things a server run legitimately
changes (repo root, seed, device, audio encoder). Everything else is a plain
class attribute you can read and edit.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(
    os.environ.get("VOICE_OMICS_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()


class Config:
    # --- Paths (audio) ---
    PROJECT_ROOT = _ROOT
    RAW_DATA_ROOT = PROJECT_ROOT / "data" / "ICHBI"
    AUDIO_DIR = RAW_DATA_ROOT / "ICBHI_final_database"
    DEMO_FILE = RAW_DATA_ROOT / "ICBHI_Challenge_demographic_information.txt"
    DIAG_FILE = RAW_DATA_ROOT / "patient_diagnosis.csv"

    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_audio" / "ichbi"
    EXTERNAL_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_audio" / "cwlsd"
    FULL_CSV = PROJECT_ROOT / "data" / "manifests" / "ichbi_slices_with_folds.csv"
    EXTERNAL_CSV = PROJECT_ROOT / "data" / "manifests" / "cwlsd_external_slices.csv"

    # Override to isolate a study, e.g. RESULTS_ROOT=results_distillation
    RESULTS_DIR = PROJECT_ROOT / os.environ.get("RESULTS_ROOT", "results")

    # --- 3-class audio task ---
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
    # Classes that can receive privileged transfer (share a definition with the
    # binary Healthy/COPD teachers). ``Other`` is masked out of every transfer term.
    PRIVILEGED_CLASSES = ("Healthy", "COPD")

    # --- Audio features (mel + MFCC, computed lazily in the Dataset) ---
    SAMPLE_RATE = 22050
    DURATION = 6
    OVERLAP = 0.5
    STEP_SIZE = int(DURATION * (1 - OVERLAP) * SAMPLE_RATE)
    TARGET_LEN = SAMPLE_RATE * DURATION
    N_MELS = 64
    N_MFCC = 40
    N_FFT = 1024
    HOP_LEN = 512
    # --- Two front-end configurations, selected by environment ---
    # A (default): full band, no normalization -- the standard ICBHI setup used
    #    for the in-distribution benchmark and comparable to prior work.
    # B (AUDIO_F_MAX=2000 NORMALIZE_FEATURES=1): the cross-cohort deployment
    #    configuration. Capping analysis at the external cohort's Nyquist (4 kHz
    #    source -> 2 kHz) puts both corpora in the same band, and per-channel
    #    CMVN cancels device/gain offsets. Without these, a model trained on
    #    ICBHI collapses to a single class on an unseen cohort.
    _F_MAX_ENV = os.environ.get("AUDIO_F_MAX", "")
    F_MAX = float(_F_MAX_ENV) if _F_MAX_ENV else None      # None -> Nyquist
    NORMALIZE_FEATURES = os.environ.get("NORMALIZE_FEATURES", "0") == "1"
    # Mel filters used *inside* the MFCC transform. Band-limiting shrinks the
    # number of usable FFT bins (~93 at f_max=2 kHz for n_fft=1024 @22.05 kHz),
    # so 128 filters no longer fit and torchaudio emits all-zero filterbanks.
    # Drop to 64 whenever the band is capped; full band keeps the original 128.
    MFCC_N_MELS = int(os.environ.get("MFCC_N_MELS", "128" if F_MAX is None else "64"))
    AUDIO_INPUT_CHANNELS = N_MELS + N_MFCC  # 104
    AUDIO_FEATURE_DIM = 512                 # encoder output width

    # --- Audio encoder choice (CNN | CRNN | TRANSFORMER) ---
    AUDIO_ENCODER = os.environ.get("AUDIO_ENCODER", "CNN").upper()

    # --- Clinical-text encoder choice (contrastive method) ---
    CLINICAL_ENCODER = os.environ.get("CLINICAL_ENCODER", "bioclinicalbert")
    CLINICAL_ENCODER_CHOICES = ("bioclinicalbert", "gatortron", "modernbert")

    # --- Shared / private latent dims (used by the DTCA consensus head) ---
    SHARED_DIM = 128
    PRIVATE_DIM = 128

    # --- Training ---
    SEED = int(os.environ.get("AUDIO_SEED", "42"))
    SEEDS = (11, 23, 42, 73, 101)  # publication repeats
    NUM_FOLDS = 5
    INNER_VALIDATION_FRACTION = float(os.environ.get("INNER_VALIDATION_FRACTION", "0.15"))
    BATCH_SIZE = 32
    DEVICE = torch.device(
        os.environ.get("VOICE_OMICS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    )
    # Audio mel/MFCC is computed per item -> parallelise loading across CPU cores.
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
    PIN_MEMORY = torch.cuda.is_available()
    # Bridge age-matching tolerance (years); default used when not swept.
    AGE_TOLERANCE = 5.0

    # --- Fine-tune recipe (proven: gentle encoder, faster head, BN frozen) ---
    ENCODER_LR = 1e-5
    HEAD_LR = 1e-3
    BASELINE_ENCODER_LR = 1e-4   # baseline trains from scratch, so a larger encoder LR
    FINETUNE_EPOCHS = 30
    # Model selection uses validation LOSS: patient macro-F1 over a ~15-patient
    # inner-validation split is heavily quantized, so selecting on it latches onto
    # noise in the first epochs and leaves the model undertrained.
    PATIENCE = 8
    MIN_EPOCHS = 12          # early stopping cannot fire before this many epochs
    BOOTSTRAP_SAMPLES = 2000

    # --- Reproducibility ---
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
        # Loader workers do light per-item work; extra intra-op threads only
        # cause contention when many runs share the machine.
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


# --------------------------------------------------------------------------- #
# Privileged views (COPDGene, unpaired with the audio cohort)
# --------------------------------------------------------------------------- #
_CG = Config.PROJECT_ROOT / "data" / "copdgene" / "data_processed"


@dataclass(frozen=True)
class PrivilegedView:
    """One privileged tabular modality. Paths follow ``configs/datasets.sh``."""

    name: str
    x_csv: Path
    y_csv: Path
    clinical_csv: Path
    index_col: str = "sid"
    # COPDGene stores the raw GOLD stage (0-4) in this one column; it doubles as
    # the label source (0 = Healthy, >0 = COPD) and the sharp-class filter
    # (keep 0 vs 3&4). Same column name for both omics and CT views.
    label_col: str = "finalgold_visit"
    gold_col: str = "finalgold_visit"
    max_features: int = 1000         # top-k ANOVA features (fold-local for teacher)


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
    max_features=134,  # CT-clinical is already low-dimensional; keep all
)

PRIVILEGED_VIEWS = {view.name: view for view in (OMICS_VIEW, CT_VIEW)}


class Sharp:
    """Sharp / high-contrast privileged label definition (Healthy vs severe COPD).

    Restores the setting that improved over unimodal audio: GOLD 0 = Healthy,
    GOLD 3&4 = severe COPD, GOLD 1&2 dropped. When ``ENABLE`` is off, or the GOLD
    column is absent, teachers fall back to the binary label with a warning.
    """

    ENABLE = os.environ.get("SHARP_CLASSES", "1") == "1"
    HEALTHY_GOLD = (0,)
    COPD_GOLD = (3, 4)


# --------------------------------------------------------------------------- #
# Method hyperparameters (defaults + small sensitivity grids)
# --------------------------------------------------------------------------- #
class Hparams:
    # Masked distillation -- key knob: max distillation weight
    DISTILL_WEIGHT = 2.0
    DISTILL_WEIGHT_GRID = (0.5, 1.0, 2.0, 4.0)
    # No ramp: the auxiliary term is applied at full weight from the first step.
    # A ramp interacts badly with early stopping (if the selected epoch falls
    # inside the ramp the objective never reaches its nominal strength), and the
    # weight sweep already explores signal strength directly.
    # Distillation objective. "cosine" is the current/default method; the others
    # are additional, more-defensible variants (never replace cosine):
    #   mse/smooth_l1 -- match magnitude too; relational (RKD) -- match the
    #   pairwise geometry between patients rather than absolute vectors.
    DISTILL_LOSS = "cosine"
    DISTILL_LOSS_GRID = ("cosine", "mse", "smooth_l1", "relational")
    # Bridge age-matching tolerance sensitivity grid (default is Config.AGE_TOLERANCE).
    AGE_TOLERANCE_GRID = (3.0, 5.0, 7.0)

    # Contrastive clinical-pivot -- key knob: temperature
    TEMPERATURE = 0.07
    TEMPERATURE_GRID = (0.05, 0.07, 0.1, 0.2)
    CONTRASTIVE_EPOCHS = 15
    PROJECTION_DIM = 128

    # Disease-topic consensus (DTCA) -- key knob: guidance weight
    GUIDANCE_WEIGHT = 1.0
    GUIDANCE_WEIGHT_GRID = (0.25, 0.5, 1.0, 2.0)
    GUIDANCE_RAMPUP_EPOCHS = 6
    NUM_TOPICS = 4          # clinical-topic prototypes per class
    TOPIC_TEMPERATURE = 0.5


# Backwards-compatible alias for the omics teacher/dataset code paths.
OmicsConfig = OMICS_VIEW
