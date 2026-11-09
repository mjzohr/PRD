"""Precompute the mel+MFCC feature map of every sliced clip so training skips audio decoding."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import soundfile as sf
import torch

from prd.config import Config
from prd.data import _mel_mfcc_transforms, feature_cache_dir


def build_cache(processed_dir: Path) -> None:
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        print(f"[cache] skip (missing): {processed_dir}")
        return
    cache_dir = feature_cache_dir(processed_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mel, mfcc = _mel_mfcc_transforms()
    wavs = sorted(processed_dir.glob("*.wav"))
    written = skipped = 0
    with torch.no_grad():
        for index, wav in enumerate(wavs):
            out = cache_dir / (wav.stem + ".npy")
            if out.exists():
                skipped += 1
                continue
            data, _ = sf.read(wav)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            features = torch.cat(
                [torch.log(mel(waveform).squeeze(0) + 1e-9), mfcc(waveform).squeeze(0)], dim=0
            )
            np.save(out, features.numpy().astype(np.float32))
            written += 1
            if (index + 1) % 500 == 0:
                print(f"[cache] {processed_dir.name}: {index + 1}/{len(wavs)}", flush=True)
    print(f"[cache] {processed_dir.name}: wrote {written}, kept {skipped}, total {len(wavs)} -> {cache_dir}")


if __name__ == "__main__":
    torch.set_num_threads(max(1, Config.NUM_WORKERS))
    build_cache(Config.PROCESSED_DIR)
    build_cache(Config.EXTERNAL_PROCESSED_DIR)
