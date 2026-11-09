"""Audio and clinical-text datasets.

Every audio batch ends with the patient id so patient-level evaluation can find it.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from prd.config import Config


def feature_cache_dir(processed_dir) -> Path:
    # tagged by analysis band so the two front-ends never share a cache;
    # normalisation is applied at load time and so is not part of the tag
    tag = "full" if Config.F_MAX is None else f"fmax{int(Config.F_MAX)}"
    return Config.PROJECT_ROOT / "data" / "feature_cache" / f"{Path(processed_dir).name}_{tag}"


def _mel_mfcc_transforms():
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE, n_mels=Config.N_MELS, n_fft=Config.N_FFT,
        hop_length=Config.HOP_LEN, f_max=Config.F_MAX,
    )
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=Config.SAMPLE_RATE,
        n_mfcc=Config.N_MFCC,
        melkwargs={"n_fft": Config.N_FFT, "n_mels": Config.MFCC_N_MELS,
                   "hop_length": Config.HOP_LEN, "mel_scale": "htk", "f_max": Config.F_MAX},
    )
    return mel, mfcc


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """Per-channel CMVN over time, cancelling device and gain offsets."""
    if not Config.NORMALIZE_FEATURES:
        return features
    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True).clamp_min(1e-5)
    return (features - mean) / std


class ClinicalTextBridge:
    """Turns a metadata row into a sentence for the frozen clinical language model.

    Diagnosis words only enter the text anchor; the supervised label is carried
    separately. GOLD 0 and GOLD >= 3 mirror the sharp teacher classes.
    """

    def _get(self, row, keys, default="Unknown"):
        for k in keys:
            if k in row and pd.notna(row[k]):
                return row[k]
        return default

    def _demographics(self, row):
        age = self._get(row, ["age_visit", "age_baseline", "age", "Age_P2", "Age_P1", "Age"], default=None)
        age_str = f"{int(float(age))}-year-old" if age not in (None, "Unknown") else "Adult"
        sex_val = str(self._get(row, ["sex", "gender", "Gender", "Sex"], default="")).lower()
        if "f" in sex_val or sex_val == "2":
            sex_str = "female"
        elif "m" in sex_val or sex_val == "1":
            sex_str = "male"
        else:
            sex_str = "patient"
        return age_str, sex_str

    def _gold_phrase(self, row):
        gold = self._get(row, ["finalGold", "finalgold_visit", "finalGold_P1", "final_Gold_Stage"], default=None)
        try:
            gold = int(float(gold))
        except (TypeError, ValueError):
            return None
        if gold == 0:
            return "Healthy condition"
        if gold >= 3:
            return "Severe Chronic Obstructive Pulmonary Disease"
        return "Chronic Obstructive Pulmonary Disease"

    def generate_text(self, row, source_type="icbhi"):
        age_str, sex_str = self._demographics(row)

        if source_type == "icbhi":
            d = row.get("disease", "Unknown")
            disease_str = {
                "COPD": "Chronic Obstructive Pulmonary Disease",
                "Healthy": "Healthy condition",
            }.get(d, str(d))
            c, w = row.get("crackles", 0), row.get("wheezes", 0)
            if c == 1 and w == 1:
                sound = " Auscultation reveals crackles and wheezes."
            elif c == 1:
                sound = " Auscultation reveals crackles."
            elif w == 1:
                sound = " Auscultation reveals wheezes."
            else:
                sound = " Auscultation reveals normal breathing."
            return f"A {age_str} {sex_str} diagnosed with {disease_str}.{sound}"

        if source_type in ("omics", "ct_clinical"):
            disease_str = self._gold_phrase(row) or "respiratory condition"
            modality = "Clinical proteomics profile." if source_type == "omics" else "Chest CT and clinical profile."
            return f"A {age_str} {sex_str} presenting with {disease_str}. {modality}"

        return "Patient data."


class AudioAugmentations:
    def __init__(self):
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)

    def __call__(self, spec):
        spec = self.freq_mask(self.time_mask(spec))
        return spec + torch.randn_like(spec) * 0.05


class _AudioBase(Dataset):
    def __init__(self, data_source, processed_dir=None):
        self.data = (
            pd.read_csv(data_source) if isinstance(data_source, str) else data_source.reset_index(drop=True)
        )
        self.processed_dir = Config.PROCESSED_DIR if processed_dir is None else processed_dir
        self.cache_dir = feature_cache_dir(self.processed_dir)
        self.mel, self.mfcc = _mel_mfcc_transforms()

    def __len__(self):
        return len(self.data)

    def _spec(self, row):
        # the cache holds raw features (scripts/cache_features.py)
        cached = self.cache_dir / (Path(str(row["filename"])).stem + ".npy")
        if cached.exists():
            return normalize_features(torch.from_numpy(np.load(cached)))
        wav, _ = sf.read(os.path.join(self.processed_dir, row["filename"]))
        waveform = torch.from_numpy(wav).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mel = torch.log(self.mel(waveform).squeeze(0) + 1e-9)
        mfcc = self.mfcc(waveform).squeeze(0)
        return normalize_features(torch.cat([mel, mfcc], dim=0))


class AudioClassifierDataset(_AudioBase):
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return self._spec(row), int(row["label_idx"]), row["pid"]


class AudioTextDataset(_AudioBase):
    def __init__(self, data_source, augment=False, processed_dir=None):
        super().__init__(data_source, processed_dir)
        self.augment = augment
        self.augmenter = AudioAugmentations()
        self.bridge = ClinicalTextBridge()

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        spec = self._spec(row)
        if self.augment:
            spec = self.augmenter(spec)
        return spec, self.bridge.generate_text(row, source_type="icbhi"), int(row["label_idx"])


class DistilledAudioDataset(_AudioBase):
    """Audio with its teacher target and a mask that is 1 when a target exists."""

    def __init__(self, data_source, target_map_path, processed_dir=None, target_dim=512):
        super().__init__(data_source, processed_dir)
        self.target_map = np.load(target_map_path, allow_pickle=True).item()
        self.target_dim = target_dim

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        spec = self._spec(row)
        entry = self.target_map.get(row["filename"])
        if entry is None:
            target, mask = torch.zeros(self.target_dim), 0.0
        else:
            target, mask = torch.tensor(np.asarray(entry), dtype=torch.float32), 1.0
        return spec, int(row["label_idx"]), target, mask, row["pid"]


class PrivilegedTextDataset(Dataset):
    """Privileged tabular rows with their clinical-text prompt."""

    def __init__(self, features: np.ndarray, clinical: pd.DataFrame, labels: np.ndarray, source_type: str):
        self.features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        self.rows = clinical.reset_index(drop=True).to_dict("records")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.source_type = source_type
        self.bridge = ClinicalTextBridge()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.bridge.generate_text(self.rows[idx], source_type=self.source_type)
        return self.features[idx], text, int(self.labels[idx])
