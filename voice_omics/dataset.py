"""Torch datasets. Mel+MFCC features are computed lazily from the sliced wavs.

Batch contracts (patient id ``pid`` is always last so ``evaluate_patient_level``
can find it):
- AudioClassifierDataset -> (spec, label, pid)
- DistilledAudioDataset  -> (spec, label, target, mask, pid)
- AudioTextDataset       -> (spec, text, label)
- PrivilegedTextDataset  -> (features, text, label)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from clinical_bridge import ClinicalTextBridge
from config import Config


def feature_cache_dir(processed_dir) -> Path:
    """Cache of precomputed (104 x T) feature maps for a sliced-audio dir
    (see precompute_features.py).

    The path is tagged with the analysis band, because a different ``F_MAX``
    produces different features -- the two front-end configurations must never
    share a cache. Normalization is NOT in the tag: it is applied at load time.
    """
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
    """Per-channel mean/variance normalization over time (CMVN).

    Each frequency channel is standardized within the slice, which cancels the
    constant log-domain offset introduced by a different microphone or gain --
    the dominant cause of cross-cohort collapse. Applied at load time (not baked
    into the cache) so it can be changed without recomputing features.
    """
    if not Config.NORMALIZE_FEATURES:
        return features
    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True).clamp_min(1e-5)
    return (features - mean) / std


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
        # Fast path: precomputed feature map (see precompute_features.py). The
        # cache holds RAW features; normalization is applied here so it stays
        # configurable without recomputing the cache.
        cached = self.cache_dir / (Path(str(row["filename"])).stem + ".npy")
        if cached.exists():
            return normalize_features(torch.from_numpy(np.load(cached)))
        # Fallback: decode + transform on the fly (slow; used only without a cache).
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
    """Adds the teacher distillation target and a mask (1 if a target exists)."""

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
    """Privileged tabular rows + their clinical-text prompt, for contrastive alignment."""

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
