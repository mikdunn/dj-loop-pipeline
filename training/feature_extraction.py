import numpy as np
import librosa
from dataclasses import dataclass
from typing import Optional


def _try_import_torch_audio():
    try:
        import torch
        import torchaudio
    except Exception:
        return None, None
    return torch, torchaudio


def _try_import_openl3():
    try:
        import openl3  # type: ignore
    except Exception:
        return None
    return openl3


@dataclass
class TrackFeatureCache:
    """Per-track cached features to avoid recomputing expensive transforms per candidate."""

    sr: int
    hop_length: int
    n_mels: int
    y_h: np.ndarray
    y_p: np.ndarray
    log_mel: np.ndarray  # shape [n_mels, frames]


def build_track_feature_cache(
    y: np.ndarray,
    sr: int,
    *,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> TrackFeatureCache:
    """Build a cache of per-track transforms used repeatedly by segment feature extraction.

    This makes candidate processing MUCH faster because we avoid calling HPSS and
    mel spectrogram computation for every candidate slice.
    """

    # HPSS once per track
    try:
        y_h, y_p = librosa.effects.hpss(y)
    except Exception:
        y_h = y
        y_p = np.zeros_like(y)

    # Log-mel once per track (librosa path avoids torchaudio overhead per segment)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = np.log10(mel + 1e-10).astype(np.float32, copy=False)

    return TrackFeatureCache(
        sr=int(sr),
        hop_length=int(hop_length),
        n_mels=int(n_mels),
        y_h=y_h.astype(np.float32, copy=False),
        y_p=y_p.astype(np.float32, copy=False),
        log_mel=log_mel,
    )

def extract_openl3_embedding(y, sr, embedding_size=512, input_repr="mel256"):
    openl3 = _try_import_openl3()
    if openl3 is None:
        raise ImportError(
            "openl3 is not installed (or not compatible with this Python). "
            "Install openl3, or use the torchaudio-based embedding instead."
        )

    emb, _ = openl3.get_audio_embedding(
        y,
        sr,
        input_repr=input_repr,
        embedding_size=embedding_size,
        content_type="music"
    )
    return np.mean(emb, axis=0)


def extract_torchaudio_logmel_embedding(
    y: np.ndarray,
    sr: int,
    *,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """PyTorch/torchaudio embedding with no network downloads.

    Produces a fixed-length vector by computing log-mel spectrogram stats:
    concat(mean over time, std over time) => 2*n_mels dims.
    """

    torch, torchaudio = _try_import_torch_audio()
    if torch is None or torchaudio is None:
        raise ImportError("torchaudio/torch not installed")

    # torchaudio expects float tensor in [-1, 1]
    wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )(wav)

    log_mel = torch.log10(mel_spec + 1e-10)  # [1, n_mels, frames]
    mean = log_mel.mean(dim=-1).squeeze(0)
    std = log_mel.std(dim=-1, unbiased=False).squeeze(0)
    emb = torch.cat([mean, std], dim=0)
    return emb.detach().cpu().numpy().astype(np.float32)


def _safe_rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))


def _window(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = int(max(0.0, start_s) * sr)
    b = int(max(0.0, end_s) * sr)
    b = min(b, len(y))
    a = min(a, b)
    return y[a:b]


def _segment_logmel_embedding_from_cache(
    cache: TrackFeatureCache,
    *,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """Fast log-mel embedding: concat(mean, std) across frames for the segment."""

    sr = cache.sr
    hop = cache.hop_length

    # Convert seconds -> frame indices (best-effort)
    start_samp = int(max(0.0, start_time) * sr)
    end_samp = int(max(0.0, end_time) * sr)
    start_f = int(np.floor(start_samp / hop))
    end_f = int(np.ceil(end_samp / hop))

    start_f = max(0, min(start_f, cache.log_mel.shape[1]))
    end_f = max(0, min(end_f, cache.log_mel.shape[1]))
    if end_f <= start_f:
        return np.zeros(cache.n_mels * 2, dtype=np.float32)

    seg = cache.log_mel[:, start_f:end_f]
    mean = seg.mean(axis=1)
    std = seg.std(axis=1)
    return np.concatenate([mean, std], axis=0).astype(np.float32, copy=False)


def extract_drum_activity_features(
    segment: np.ndarray,
    sr: int,
    *,
    boundary_window_s: float = 0.20,
    y_p_override: Optional[np.ndarray] = None,
    y_h_override: Optional[np.ndarray] = None,
) -> dict:
    """Heuristics to quantify "how drummy" a segment is, and whether boundaries are "clean".

    This is not literal drum-source-separation; it uses HPSS (harmonic/percussive)
    and percussive onset strength as a robust proxy.

    Returned keys are designed to help choose cut points at bar boundaries:
    - Lower boundary_* values => cleaner cut (less percussive transient at edge)
    - Lower perc_to_total_rms => fewer drums overall
    """

    if segment.size == 0:
        return {}

    if y_p_override is not None and y_h_override is not None:
        y_h = y_h_override
        y_p = y_p_override
    else:
        # HPSS works best with some minimum length; short slices still work but can be noisy.
        try:
            y_h, y_p = librosa.effects.hpss(segment)
        except Exception:
            # If HPSS fails for any reason, gracefully degrade.
            y_h = segment
            y_p = np.zeros_like(segment)

    total_rms = _safe_rms(segment)
    perc_rms = _safe_rms(y_p)
    harm_rms = _safe_rms(y_h)

    # Percussive onset proxy.
    try:
        onset_p = librosa.onset.onset_strength(y=y_p, sr=sr)
        perc_onset_mean = float(np.mean(onset_p)) if onset_p.size else 0.0
        perc_onset_std = float(np.std(onset_p)) if onset_p.size else 0.0
    except Exception:
        perc_onset_mean = 0.0
        perc_onset_std = 0.0

    # Boundary cleanliness: evaluate percussive energy and onsets very near the edges.
    start_w = _window(y_p, sr, 0.0, boundary_window_s)
    end_w = _window(y_p, sr, max(0.0, (len(y_p) / sr) - boundary_window_s), len(y_p) / sr)

    boundary_perc_rms_start = _safe_rms(start_w)
    boundary_perc_rms_end = _safe_rms(end_w)

    try:
        boundary_onset_start = float(np.mean(librosa.onset.onset_strength(y=start_w, sr=sr))) if start_w.size else 0.0
    except Exception:
        boundary_onset_start = 0.0

    try:
        boundary_onset_end = float(np.mean(librosa.onset.onset_strength(y=end_w, sr=sr))) if end_w.size else 0.0
    except Exception:
        boundary_onset_end = 0.0

    # Ratios are often more stable than raw levels.
    eps = 1e-8
    perc_to_total_rms = float(perc_rms / (total_rms + eps))
    harm_to_total_rms = float(harm_rms / (total_rms + eps))

    # A single scalar that can be used to rank “clean cut” candidates.
    boundary_quiet_score = float(0.5 * (boundary_onset_start + boundary_onset_end) + 0.5 * (boundary_perc_rms_start + boundary_perc_rms_end))

    return {
        "perc_rms": float(perc_rms),
        "harm_rms": float(harm_rms),
        "perc_to_total_rms": perc_to_total_rms,
        "harm_to_total_rms": harm_to_total_rms,
        "perc_onset_mean": float(perc_onset_mean),
        "perc_onset_std": float(perc_onset_std),
        "boundary_perc_rms_start": float(boundary_perc_rms_start),
        "boundary_perc_rms_end": float(boundary_perc_rms_end),
        "boundary_perc_onset_start": float(boundary_onset_start),
        "boundary_perc_onset_end": float(boundary_onset_end),
        "boundary_quiet_score": float(boundary_quiet_score),
    }

def extract_full_features(y, sr, start_time, end_time, *, cache: Optional[TrackFeatureCache] = None):
    start_s = int(start_time * sr)
    end_s = int(end_time * sr)
    segment = y[start_s:end_s]
    if len(segment) == 0:
        return {}
    feats = {}
    # Energy / RMS
    rms = librosa.feature.rms(y=segment)[0]
    feats["rms_mean"] = float(np.mean(rms))
    # Rhythm / onset strength
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    feats["onset_mean"] = float(np.mean(onset_env))

    # Drum activity / "clean cut" heuristics
    if cache is not None:
        y_p_seg = cache.y_p[start_s:end_s]
        y_h_seg = cache.y_h[start_s:end_s]
        feats.update(extract_drum_activity_features(segment, sr, y_p_override=y_p_seg, y_h_override=y_h_seg))
    else:
        feats.update(extract_drum_activity_features(segment, sr))
    # OpenL3 embedding
    # Prefer a lightweight PyTorch/torchaudio embedding (works on Python 3.12),
    # fall back to OpenL3 if installed.
    try:
        if cache is not None:
            emb = _segment_logmel_embedding_from_cache(cache, start_time=start_time, end_time=end_time)
        else:
            emb = extract_torchaudio_logmel_embedding(segment, sr)

        for i, val in enumerate(emb):
            feats[f"pt_mel_{i}"] = float(val)
    except Exception:
        try:
            emb = extract_openl3_embedding(segment, sr)
            for i, val in enumerate(emb):
                feats[f"openl3_{i}"] = float(val)
        except Exception:
            # No learned embedding available; proceed with handcrafted features only.
            pass
    return feats
