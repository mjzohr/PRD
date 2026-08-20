"""Model definitions: audio encoders (CNN/CRNN/Transformer), a tabular teacher,
frozen clinical-text encoder, and the DTCA consensus head.

Design note (transfer fix): the DTCA audio model classifies through the SHARED
subspace only, so privileged guidance on that subspace cannot be routed around by
an unguided private branch.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn

from config import Config


# Clinical-text encoder variants (frozen). IDs overridable via env in case a
# HuggingFace repo id differs on your server.
CLINICAL_ENCODERS = {
    "bioclinicalbert": os.environ.get("BIOCLINICALBERT_ID", "emilyalsentzer/Bio_ClinicalBERT"),
    "gatortron": os.environ.get("GATORTRON_ID", "UFNLP/gatortron-base"),
    "modernbert": os.environ.get("BIOCLINICAL_MODERNBERT_ID", "thomas-sounack/BioClinical-ModernBERT-base"),
}


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, embed_dim=768, projection_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Audio encoders (input: (B, N_MELS+N_MFCC, T); output: (B, 512))
# --------------------------------------------------------------------------- #
class AudioEncoderCNN(nn.Module):
    """Stack of 1-D conv blocks + global average pooling. Config-searchable.
    Defaults reproduce the original (104->128->256->512, kernel 3, no dropout,
    pooling between blocks); every config still outputs 512-d."""

    def __init__(self, conv_channels=(128, 256, 512), kernel_size=3, dropout=0.0):
        super().__init__()
        c = Config.AUDIO_INPUT_CHANNELS
        blocks, in_ch = [], c
        for i, out_ch in enumerate(conv_channels):
            blocks += [nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                       nn.BatchNorm1d(out_ch), nn.ReLU()]
            if i < len(conv_channels) - 1:  # no pooling after the last block
                blocks.append(nn.MaxPool1d(2))
            if dropout > 0:  # omitted entirely at 0.0 so the default state_dict is unchanged
                blocks.append(nn.Dropout(dropout))
            in_ch = out_ch
        blocks.append(nn.AdaptiveAvgPool1d(1))
        self.net = nn.Sequential(*blocks)
        self.proj = nn.Identity() if in_ch == Config.AUDIO_FEATURE_DIM \
            else nn.Linear(in_ch, Config.AUDIO_FEATURE_DIM)
        self.output_dim = Config.AUDIO_FEATURE_DIM

    def forward(self, x):
        return self.proj(self.net(x).squeeze(-1))


class AudioEncoderCRNN(nn.Module):
    """CNN front-end + bidirectional GRU (temporal memory). Config-searchable.
    Defaults reproduce the original; every config still outputs 512-d."""

    def __init__(self, gru_hidden=128, gru_layers=2, dropout=0.0, conv_channels=(128, 256)):
        # Defaults set from the architecture search (val macro-F1 0.818, seed 42).
        super().__init__()
        c = Config.AUDIO_INPUT_CHANNELS
        ch1, ch2 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv1d(c, ch1, 3, padding=1), nn.BatchNorm1d(ch1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch1, ch2, 3, padding=1), nn.BatchNorm1d(ch2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(ch2, gru_hidden, num_layers=gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0)
        gru_out = gru_hidden * 2
        self.proj = nn.Identity() if gru_out == Config.AUDIO_FEATURE_DIM \
            else nn.Linear(gru_out, Config.AUDIO_FEATURE_DIM)
        self.output_dim = Config.AUDIO_FEATURE_DIM

    def forward(self, x):
        h = self.conv(x).transpose(1, 2)
        out, _ = self.gru(h)
        return self.proj(out.mean(dim=1))


class _SinusoidalPositionalEncoding(nn.Module):
    """Adds fixed sinusoidal position information so attention can use temporal order."""

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AudioEncoderTransformer(nn.Module):
    """Conv stem -> (optional positional encoding) -> Transformer encoder.
    Defaults reproduce the original (no positional encoding); outputs 512-d."""

    def __init__(self, d_model=256, nhead=4, layers=2, dim_feedforward=512,
                 dropout=0.1, positional_encoding=False):
        # Defaults set from the architecture search (val macro-F1 0.757, seed 42).
        super().__init__()
        c = Config.AUDIO_INPUT_CHANNELS
        self.embed = nn.Sequential(
            nn.Conv1d(c, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.pos = _SinusoidalPositionalEncoding(d_model) if positional_encoding else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.proj = nn.Identity() if d_model == Config.AUDIO_FEATURE_DIM \
            else nn.Linear(d_model, Config.AUDIO_FEATURE_DIM)
        self.output_dim = Config.AUDIO_FEATURE_DIM

    def forward(self, x):
        h = self.embed(x).transpose(1, 2)
        if self.pos is not None:
            h = self.pos(h)
        return self.proj(self.transformer(h).mean(dim=1))


_AUDIO_ENCODERS = {
    "CNN": AudioEncoderCNN,
    "CRNN": AudioEncoderCRNN,
    "TRANSFORMER": AudioEncoderTransformer,
}


def get_audio_encoder(name: str | None = None) -> nn.Module:
    key = (name or Config.AUDIO_ENCODER).upper()
    if key not in _AUDIO_ENCODERS:
        raise ValueError(f"Unknown audio encoder {key!r}; choose from {sorted(_AUDIO_ENCODERS)}")
    return _AUDIO_ENCODERS[key]()


# Back-compat name used by the distillation/baseline scripts.
AudioEncoder = AudioEncoderCNN


# --------------------------------------------------------------------------- #
# Tabular encoder (omics / CT) and frozen clinical-text encoder
# --------------------------------------------------------------------------- #
class TabularEncoder(nn.Module):
    output_dim = 512

    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


OmicsEncoder = TabularEncoder  # alias


class TextEncoder(nn.Module):
    """Frozen clinical-text encoder (CLS pooling), a stable semantic anchor.

    Three interchangeable variants (``encoder_name``): ``bioclinicalbert`` (768-d),
    ``gatortron`` (1024-d), ``modernbert`` (BioClinical ModernBERT, 768-d). The
    output width is read from the loaded model, so downstream projections must use
    ``self.output_dim`` rather than assuming 768.
    """

    def __init__(self, device, encoder_name="bioclinicalbert"):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "TextEncoder needs the text extra: uv pip install -e '.[audio,text]'"
            ) from exc
        self.device = device
        self.encoder_name = encoder_name
        model_id = CLINICAL_ENCODERS.get(encoder_name, encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.bert = AutoModel.from_pretrained(model_id)
        self.output_dim = int(self.bert.config.hidden_size)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

    def forward(self, text_list):
        with torch.no_grad():
            inputs = self.tokenizer(
                text_list, return_tensors="pt", padding=True, truncation=True, max_length=64
            ).to(self.device)
            out = self.bert(**inputs)
        return out.last_hidden_state[:, 0, :]  # CLS token


# --------------------------------------------------------------------------- #
# Task heads
# --------------------------------------------------------------------------- #
class ClassifierHead(nn.Module):
    """Encoder + linear head. Used by the audio baseline and distillation student."""

    def __init__(self, encoder, num_classes=Config.NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(getattr(encoder, "output_dim", 512), num_classes)

    def forward(self, x, return_features=False):
        feat = self.encoder(x)
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits


class DistillationProjector(nn.Module):
    """Maps audio features (512) into the teacher target space (512)."""

    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.BatchNorm1d(input_dim), nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TeacherModel(nn.Module):
    """Privileged tabular teacher: encoder -> shared embedding -> binary head.

    The shared embedding lives in ``shared_dim`` so its class/topic centroids can be
    used as DTCA prototypes that the audio model is guided toward.
    """

    def __init__(self, input_dim, shared_dim=Config.SHARED_DIM, num_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = TabularEncoder(input_dim, dropout=dropout)
        self.shared = nn.Sequential(nn.Linear(512, shared_dim), nn.BatchNorm1d(shared_dim), nn.ReLU())
        self.head = nn.Linear(shared_dim, num_classes)

    def forward(self, x, return_shared=False):
        z = self.shared(self.encoder(x))
        logits = self.head(z)
        if return_shared:
            return logits, z
        return logits


class ConsensusAudioModel(nn.Module):
    """DTCA audio model. Classifies through the SHARED subspace only, so privileged
    guidance on that subspace directly shapes the decision function."""

    def __init__(self, encoder, num_classes=Config.NUM_CLASSES, shared_dim=Config.SHARED_DIM):
        super().__init__()
        self.encoder = encoder
        self.shared = nn.Sequential(
            nn.Linear(getattr(encoder, "output_dim", 512), shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(shared_dim, num_classes)

    def forward(self, x, return_shared=False):
        z = self.shared(self.encoder(x))
        logits = self.classifier(z)
        if return_shared:
            return logits, z
        return logits
