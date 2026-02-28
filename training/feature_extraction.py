from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import librosa


def _band_energy_envelope(seg: np.ndarray, sr: int, fmin: float, fmax: float, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Compute frame-wise magnitude energy in a frequency band."""
    spec = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.zeros(spec.shape[1], dtype=float)
    env = np.mean(spec[mask, :], axis=0)
    return np.asarray(env, dtype=float)


def _window_mean(env: np.ndarray, center_frame: int, half_width: int) -> float:
    if env.size == 0:
        return 0.0
    a = max(0, center_frame - half_width)
    b = min(env.size, center_frame + half_width + 1)
    if b <= a:
        return 0.0
    return float(np.mean(env[a:b]))


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        v = float(value)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def extract_full_features(y: np.ndarray, sr: int, start_time: float, end_time: float) -> Optional[Dict[str, float]]:
    """Extract lightweight rhythm/percussion features for a time slice."""
    if y is None or sr <= 0:
        return None

    start_idx = max(0, int(start_time * sr))
    end_idx = min(len(y), int(end_time * sr))
    if end_idx <= start_idx:
        return None

    seg = y[start_idx:end_idx]
    if seg.size < 2048:
        return None

    duration = seg.size / float(sr)

    rms = librosa.feature.rms(y=seg, frame_length=2048, hop_length=512)[0]
    zcr = librosa.feature.zero_crossing_rate(seg, frame_length=2048, hop_length=512)[0]
    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85)[0]

    onset_env = librosa.onset.onset_strength(y=seg, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    try:
        y_h, y_p = librosa.effects.hpss(seg)
        e_h = float(np.mean(y_h ** 2))
        e_p = float(np.mean(y_p ** 2))
        percussive_ratio = e_p / (e_h + e_p + 1e-8)
    except Exception:
        percussive_ratio = 0.0

    onset_density = (len(onset_frames) / duration) if duration > 0 else 0.0

    # --- Drum boundary cues for tighter loop starts/phrasing ---
    hop = 512
    kick_env = _band_energy_envelope(seg, sr, fmin=30.0, fmax=160.0, hop=hop)
    snare_env = _band_energy_envelope(seg, sr, fmin=180.0, fmax=2500.0, hop=hop)
    hat_env = _band_energy_envelope(seg, sr, fmin=5000.0, fmax=12000.0, hop=hop)

    # Normalize envelopes for relative scoring.
    kick_norm = kick_env / (np.max(kick_env) + 1e-8)
    snare_norm = snare_env / (np.max(snare_env) + 1e-8)
    hat_norm = hat_env / (np.max(hat_env) + 1e-8)

    # Estimate local tempo from onset envelope for beat-position features.
    try:
        tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop, aggregate=np.median)[0])
    except Exception:
        tempo = 120.0
    tempo = tempo if np.isfinite(tempo) and tempo > 1.0 else 120.0

    beat_period_sec = 60.0 / tempo
    beat_period_frames = max(1, int(round((beat_period_sec * sr) / hop)))
    near_half_beat = max(1, beat_period_frames // 2)
    half_win = max(1, int(round(0.10 * beat_period_frames)))

    # Kick on beat 1: emphasize low-band hit at loop start.
    kick_on_one_score = _window_mean(kick_norm, center_frame=0, half_width=max(1, half_win))

    # Snare on beat 2 or 3: mid-band hit near +1 beat or +2 beats from start.
    snare_on_two = _window_mean(snare_norm, center_frame=beat_period_frames, half_width=max(1, half_win))
    snare_on_three = _window_mean(snare_norm, center_frame=2 * beat_period_frames, half_width=max(1, half_win))
    snare_two_or_three_score = max(snare_on_two, snare_on_three)

    # Hi-hat spacing: autocorrelation peaks near half/quarter/whole-note intervals.
    hat_ac = np.correlate(hat_norm, hat_norm, mode="full")
    hat_ac = hat_ac[hat_ac.size // 2:]
    hat_ac = hat_ac / (hat_ac[0] + 1e-8)

    lag_quarter = beat_period_frames
    lag_half = 2 * beat_period_frames
    lag_whole = 4 * beat_period_frames

    hat_grid_candidates = []
    for lag in (near_half_beat, lag_quarter, lag_half, lag_whole):
        if lag < hat_ac.size:
            hat_grid_candidates.append(float(hat_ac[lag]))
    hihat_grid_score = max(hat_grid_candidates) if hat_grid_candidates else 0.0

    boundary_drum_score = 0.45 * kick_on_one_score + 0.35 * snare_two_or_three_score + 0.20 * hihat_grid_score

    return {
        "duration_sec": _safe_float(duration),
        "rms_mean": _safe_float(np.mean(rms)),
        "rms_std": _safe_float(np.std(rms)),
        "zcr_mean": _safe_float(np.mean(zcr)),
        "spectral_centroid_mean": _safe_float(np.mean(centroid)),
        "spectral_bandwidth_mean": _safe_float(np.mean(bandwidth)),
        "spectral_rolloff_mean": _safe_float(np.mean(rolloff)),
        "onset_strength_mean": _safe_float(np.mean(onset_env)),
        "onset_strength_std": _safe_float(np.std(onset_env)),
        "onset_density": _safe_float(onset_density),
        "percussive_ratio": _safe_float(percussive_ratio),
        "kick_on_one_score": _safe_float(kick_on_one_score),
        "snare_two_or_three_score": _safe_float(snare_two_or_three_score),
        "hihat_grid_score": _safe_float(hihat_grid_score),
        "boundary_drum_score": _safe_float(boundary_drum_score),
    }
