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


def _fit_ar_coeffs(x: np.ndarray, order: int = 6) -> Optional[Dict[str, np.ndarray | float]]:
    x = np.asarray(x, dtype=float).reshape(-1)
    p = int(max(1, order))
    if x.size <= p + 1:
        return None

    # y[t] = sum_k phi_k * y[t-k] + e[t]
    y = x[p:]
    X = np.column_stack([x[p - k - 1 : -k - 1] for k in range(p)])

    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return None

    pred = X @ coef
    resid = y - pred
    resid_var = float(np.var(resid)) if resid.size else 0.0
    return {
        "coef": np.asarray(coef, dtype=float),
        "resid_var": resid_var,
    }


def _ar_feature_dict(x: np.ndarray, prefix: str, order: int = 6) -> Dict[str, float]:
    out: Dict[str, float] = {}
    fit = _fit_ar_coeffs(x, order=order)
    if fit is None:
        for i in range(1, int(order) + 1):
            out[f"{prefix}_ar{i}"] = 0.0
        out[f"{prefix}_ar_resid_var"] = 0.0
        return out

    coef = np.asarray(fit["coef"], dtype=float)
    for i in range(1, int(order) + 1):
        out[f"{prefix}_ar{i}"] = float(coef[i - 1]) if i - 1 < coef.size else 0.0
    out[f"{prefix}_ar_resid_var"] = float(fit["resid_var"])
    return out


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


def extract_loop_features(
    file_path: Path,
    sr: int = 22050,
    max_seconds: float = 12.0,
    use_ar_features: bool = False,
    ar_order: int = 6,
) -> Optional[Dict[str, float]]:
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
    onset_z = _z(onset_env) if onset_env.size else np.zeros(0, dtype=np.float32)

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=fs, n_fft=2048)
    low_mask = (freqs >= 20.0) & (freqs <= 180.0)
    low_env = _z(np.mean(S[low_mask, :], axis=0)) if np.any(low_mask) else np.zeros_like(onset_z)
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

    if bool(use_ar_features):
        out.update(_ar_feature_dict(onset_z, prefix="onset_z", order=int(ar_order)))
        out.update(_ar_feature_dict(low_env, prefix="low_env", order=int(ar_order)))

    return out
