import os
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np

EXTS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")


def collect_audio_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dp, _, fs in os.walk(root):
        for n in fs:
            p = Path(dp) / n
            if p.suffix.lower() in EXTS:
                out.append(p)
    return sorted(out)


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-8)


def extract_pattern_sequence(file_path: Path, sr: int = 22050, max_seconds: float = 20.0, max_frames: int = 450) -> Optional[np.ndarray]:
    try:
        y, fs = librosa.load(str(file_path), sr=sr, mono=True)
    except Exception:
        return None

    if y is None or len(y) < 4096:
        return None

    y = y[: int(max_seconds * fs)]
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=fs, hop_length=hop)
    if onset.size < 8:
        return None

    onset_z = _z(onset)
    onset_delta = np.diff(onset_z, prepend=onset_z[0])

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=fs, n_fft=2048)
    low_mask = (freqs >= 20.0) & (freqs <= 180.0)
    low_env = _z(np.mean(S[low_mask, :], axis=0)) if np.any(low_mask) else np.zeros_like(onset_z)

    t = min(len(onset_z), len(onset_delta), len(low_env))
    X = np.vstack([onset_z[:t], onset_delta[:t], low_env[:t]]).astype(np.float32)

    if X.shape[1] > max_frames:
        idx = np.linspace(0, X.shape[1] - 1, max_frames).astype(int)
        X = X[:, idx]

    return X


def extract_loop_features(file_path: Path, sr: int = 22050, max_seconds: float = 12.0) -> Optional[Dict[str, float]]:
    try:
        y, fs = librosa.load(str(file_path), sr=sr, mono=True)
    except Exception:
        return None

    if y is None or len(y) < 4096:
        return None

    y = y[: int(max_seconds * fs)]

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=fs)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=fs)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=fs, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    onset_env = librosa.onset.onset_strength(y=y, sr=fs)
    try:
        tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=fs, aggregate=np.median)[0])
    except Exception:
        tempo = 120.0

    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13)

    out: Dict[str, float] = {
        "duration_sec": float(len(y) / fs),
        "tempo_est": float(tempo if np.isfinite(tempo) else 120.0),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "flatness_mean": float(np.mean(flatness)),
        "onset_strength_mean": float(np.mean(onset_env)) if onset_env.size else 0.0,
        "onset_strength_std": float(np.std(onset_env)) if onset_env.size else 0.0,
    }
    for i in range(mfcc.shape[0]):
        out[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        out[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
    return out
