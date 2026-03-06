import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")


def collect_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
    return sorted(files)


def _normalize_audio(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if peak < 1e-8:
        return y
    return y / peak


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-8 or bn < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def _segment_feature(segment: np.ndarray, sr: int) -> np.ndarray:
    if segment.size < 512:
        return np.zeros(40, dtype=float)

    seg = _normalize_audio(segment)
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=seg, sr=sr)
    onset_env = librosa.onset.onset_strength(y=seg, sr=sr)

    mfcc_stats = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)], axis=0)
    chroma_stats = np.mean(chroma, axis=1)

    if onset_env.size > 4:
        onset_env = onset_env / (np.max(onset_env) + 1e-8)
        bins = np.array_split(onset_env, 8)
        onset_hist = np.array([float(np.mean(b)) for b in bins], dtype=float)
    else:
        onset_hist = np.zeros(8, dtype=float)

    feat = np.concatenate([mfcc_stats, chroma_stats, onset_hist], axis=0)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_one_shot_features_from_wave(y: np.ndarray, sr: int, max_seconds: float = 1.0) -> Optional[Dict[str, float]]:
    if y is None or len(y) < 512:
        return None

    # Select highest-energy window to capture the dominant hit in the divider.
    max_len = int(max_seconds * sr)
    if len(y) > max_len and max_len > 512:
        hop = max(128, max_len // 8)
        best_start = 0
        best_energy = -1.0
        for s in range(0, len(y) - max_len + 1, hop):
            e = float(np.mean(y[s : s + max_len] ** 2))
            if e > best_energy:
                best_energy = e
                best_start = s
        y = y[best_start : best_start + max_len]

    try:
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception:
        pass

    if len(y) < 512:
        return None

    y = _normalize_audio(y)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_count = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel + 1e-10)
    mfcc = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=13)

    stft = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def band_energy(fmin: float, fmax: float) -> float:
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return 0.0
        return float(np.mean(stft[mask, :]))

    low_e = band_energy(20, 180)
    mid_e = band_energy(180, 2500)
    high_e = band_energy(5000, 12000)
    total_e = low_e + mid_e + high_e + 1e-8

    feats: Dict[str, float] = {
        "duration_sec": float(len(y) / sr),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "flatness_mean": float(np.mean(flatness)),
        "onset_strength_mean": float(np.mean(onset_env)) if onset_env.size else 0.0,
        "onset_strength_std": float(np.std(onset_env)) if onset_env.size else 0.0,
        "onset_count": float(onset_count),
        "low_band_ratio": float(low_e / total_e),
        "mid_band_ratio": float(mid_e / total_e),
        "high_band_ratio": float(high_e / total_e),
    }

    for i in range(mfcc.shape[0]):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    return feats


def _slice_with_offset(y: np.ndarray, n_dividers: int, offset_samples: int) -> List[np.ndarray]:
    n = len(y)
    if n_dividers <= 0 or n <= n_dividers:
        return []

    y2 = np.concatenate([y[offset_samples:], y[:offset_samples]], axis=0)
    boundaries = np.linspace(0, len(y2), n_dividers + 1).astype(int)

    segments: List[np.ndarray] = []
    for i in range(n_dividers):
        s = boundaries[i]
        e = boundaries[i + 1]
        if e - s < 64:
            continue
        segments.append(y2[s:e])
    return segments


def _structure_scores_for_dividers(
    y: np.ndarray,
    sr: int,
    n_dividers: int,
    phase_steps: int = 24,
    drum_bundle: Optional[Dict] = None,
    min_conf: float = 0.25,
    drum_weight: float = 0.18,
    periodicity_weight: float = 0.10,
) -> Dict:
    n = len(y)
    seg_len = max(1, n // n_dividers)

    best: Optional[Dict] = None

    for step in range(phase_steps):
        offset = int(round(step * seg_len / phase_steps))
        segments = _slice_with_offset(y, n_dividers=n_dividers, offset_samples=offset)
        if len(segments) != n_dividers:
            continue

        feats = np.stack([_segment_feature(seg, sr=sr) for seg in segments], axis=0)
        sims = np.zeros((n_dividers, n_dividers), dtype=float)
        for i in range(n_dividers):
            for j in range(i, n_dividers):
                c = _safe_cosine(feats[i], feats[j])
                sims[i, j] = c
                sims[j, i] = c

        # Repetition: mean max similarity to another segment.
        max_other = []
        for i in range(n_dividers):
            others = np.delete(sims[i], i)
            max_other.append(float(np.max(others)) if others.size else 0.0)
        repetition_score = float(np.mean(max_other)) if max_other else 0.0

        half = n_dividers // 2
        if half >= 1:
            first_half = np.mean(feats[:half], axis=0)
            second_half = np.mean(feats[half : 2 * half], axis=0)
            half_similarity = _safe_cosine(first_half, second_half)

            # Superimposition idea: overlap-average waveform halves.
            wav_a = np.concatenate(segments[:half], axis=0)
            wav_b = np.concatenate(segments[half : 2 * half], axis=0)
            min_len = min(len(wav_a), len(wav_b))
            if min_len > 256:
                a = _normalize_audio(wav_a[:min_len])
                b = _normalize_audio(wav_b[:min_len])
                superimpose = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-8 and np.std(b) > 1e-8 else 0.0
            else:
                superimpose = 0.0
        else:
            half_similarity = 0.0
            superimpose = 0.0

        # Smooth structural score
        structure_score = float(0.45 * repetition_score + 0.35 * half_similarity + 0.20 * max(0.0, superimpose))

        drum_score = 0.0
        periodicity_score = 0.0
        periodicity_kick = 0.0
        periodicity_snare = 0.0
        periodicity_hihat = 0.0
        periodicity_best_lag = 0
        periodicity_pattern = ""
        if drum_bundle is not None:
            divider_preds = _divider_drum_predictions(
                segments=segments,
                sr=sr,
                drum_bundle=drum_bundle,
                min_conf=min_conf,
            )
            drum_score = _drum_alignment_score(divider_preds=divider_preds, n_dividers=n_dividers)
            prof = _periodicity_profile(divider_preds=divider_preds, n_dividers=n_dividers)
            periodicity_score = float(prof.get("periodicity_score", 0.0))
            periodicity_kick = float(prof.get("periodicity_kick", 0.0))
            periodicity_snare = float(prof.get("periodicity_snare", 0.0))
            periodicity_hihat = float(prof.get("periodicity_hihat", 0.0))
            periodicity_best_lag = int(prof.get("periodicity_best_lag", 0))
            periodicity_pattern = str(prof.get("periodicity_pattern", ""))

        structure_weight = max(0.0, 1.0 - float(drum_weight) - float(periodicity_weight))
        combined_score = float(
            structure_weight * structure_score + float(drum_weight) * drum_score + float(periodicity_weight) * periodicity_score
        )

        entry = {
            "offset_samples": int(offset),
            "repetition_score": repetition_score,
            "half_similarity": float(half_similarity),
            "superimpose_similarity": float(superimpose),
            "structure_score": structure_score,
            "drum_alignment_score": float(drum_score),
            "periodicity_score": float(periodicity_score),
            "periodicity_kick": float(periodicity_kick),
            "periodicity_snare": float(periodicity_snare),
            "periodicity_hihat": float(periodicity_hihat),
            "periodicity_best_lag": int(periodicity_best_lag),
            "periodicity_pattern": periodicity_pattern,
            "combined_score": float(combined_score),
            "segments": segments,
            "similarity_matrix": sims.tolist(),
        }

        if best is None or entry["combined_score"] > best.get("combined_score", best["structure_score"]):
            best = entry

    if best is None:
        return {
            "offset_samples": 0,
            "repetition_score": 0.0,
            "half_similarity": 0.0,
            "superimpose_similarity": 0.0,
            "structure_score": 0.0,
            "drum_alignment_score": 0.0,
            "periodicity_score": 0.0,
            "periodicity_kick": 0.0,
            "periodicity_snare": 0.0,
            "periodicity_hihat": 0.0,
            "periodicity_best_lag": 0,
            "periodicity_pattern": "",
            "combined_score": 0.0,
            "segments": [],
            "similarity_matrix": [],
        }
    return best


def _assign_symbol_pattern(similarity_matrix: List[List[float]], threshold: float = 0.82) -> str:
    if not similarity_matrix:
        return ""
    sims = np.asarray(similarity_matrix, dtype=float)
    n = sims.shape[0]

    labels: List[int] = []
    prototypes: List[int] = []

    for i in range(n):
        assigned = False
        for proto_idx, p in enumerate(prototypes):
            if float(sims[i, p]) >= float(threshold):
                labels.append(proto_idx)
                assigned = True
                break
        if not assigned:
            prototypes.append(i)
            labels.append(len(prototypes) - 1)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chars = []
    for v in labels:
        if v < len(alphabet):
            chars.append(alphabet[v])
        else:
            chars.append(f"X{v}")
    return "".join(chars)


def _estimate_tempo(y: np.ndarray, sr: int) -> float:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.size == 0:
        return 120.0
    try:
        tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)[0])
        if not np.isfinite(tempo):
            return 120.0
        return tempo
    except Exception:
        return 120.0


def _estimate_beat_times(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if beat_times is None or len(beat_times) == 0:
            return np.array([], dtype=float)
        return np.asarray(beat_times, dtype=float)
    except Exception:
        return np.array([], dtype=float)


def _snap_boundaries_to_beats(boundaries_sec: List[float], beat_times: np.ndarray, max_shift_sec: float) -> List[float]:
    if beat_times.size == 0:
        return boundaries_sec

    out: List[float] = []
    for i, b in enumerate(boundaries_sec):
        if i == 0 or i == len(boundaries_sec) - 1:
            out.append(float(b))
            continue

        idx = int(np.argmin(np.abs(beat_times - b)))
        nearest = float(beat_times[idx])
        if abs(nearest - b) <= max_shift_sec:
            out.append(nearest)
        else:
            out.append(float(b))

    # enforce monotonicity
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _segments_from_boundaries(y: np.ndarray, sr: int, boundaries_sec: List[float]) -> List[np.ndarray]:
    segments: List[np.ndarray] = []
    for i in range(len(boundaries_sec) - 1):
        s = max(0, int(round(boundaries_sec[i] * sr)))
        e = min(len(y), int(round(boundaries_sec[i + 1] * sr)))
        if e - s < 64:
            continue
        segments.append(y[s:e])
    return segments


def _divider_drum_predictions(
    segments: List[np.ndarray],
    sr: int,
    drum_bundle: Optional[Dict],
    min_conf: float,
) -> List[Dict]:
    out: List[Dict] = []
    if drum_bundle is None:
        for i, seg in enumerate(segments, start=1):
            out.append(
                {
                    "divider_index": i,
                    "predicted_label": "unavailable",
                    "confidence": 0.0,
                    "probs": {},
                    "duration_sec": float(len(seg) / sr),
                }
            )
        return out

    model = drum_bundle["model"]
    feature_cols: List[str] = drum_bundle["feature_cols"]

    for i, seg in enumerate(segments, start=1):
        feats = _extract_one_shot_features_from_wave(seg, sr=sr)
        if feats is None:
            out.append(
                {
                    "divider_index": i,
                    "predicted_label": "uncertain",
                    "confidence": 0.0,
                    "probs": {},
                    "duration_sec": float(len(seg) / sr),
                }
            )
            continue

        x = pd.DataFrame([feats])
        for col in feature_cols:
            if col not in x.columns:
                x[col] = 0.0
        x = x[feature_cols]

        probs = model.predict_proba(x)[0]
        classes = list(model.classes_)
        best_idx = int(np.argmax(probs))
        pred = str(classes[best_idx])
        conf = float(probs[best_idx])

        if conf < min_conf:
            pred = "uncertain"

        out.append(
            {
                "divider_index": i,
                "predicted_label": pred,
                "confidence": conf,
                "probs": {str(c): float(p) for c, p in zip(classes, probs)},
                "duration_sec": float(len(seg) / sr),
            }
        )

    return out


def _prob_from_prediction(pred: Dict, tags: Tuple[str, ...]) -> float:
    tags_l = tuple(t.lower() for t in tags)
    probs = pred.get("probs", {}) or {}

    best = 0.0
    for k, v in probs.items():
        kk = str(k).lower()
        if any(t in kk for t in tags_l):
            try:
                best = max(best, float(v))
            except Exception:
                pass

    lbl = str(pred.get("predicted_label", "")).lower()
    conf = float(pred.get("confidence", 0.0))
    if any(t in lbl for t in tags_l):
        best = max(best, conf)
    return float(best)


def _drum_alignment_score(divider_preds: List[Dict], n_dividers: int) -> float:
    if not divider_preds or n_dividers <= 0:
        return 0.0

    beat1 = _prob_from_prediction(divider_preds[0], ("kick", "bd"))

    idxs: List[int] = []
    q1 = n_dividers // 4
    q3 = (3 * n_dividers) // 4
    for i in (q1, q3):
        if 0 <= i < len(divider_preds):
            idxs.append(i)

    if idxs:
        backbeats = [_prob_from_prediction(divider_preds[i], ("snare", "clap")) for i in idxs]
        backbeat = float(np.mean(backbeats))
    else:
        backbeat = 0.0

    return float(0.70 * beat1 + 0.30 * backbeat)


def _lag_similarity(vec: np.ndarray, lag: int) -> float:
    if vec.size == 0 or lag <= 0 or lag >= vec.size:
        return 0.0
    return max(0.0, _safe_cosine(vec.astype(float), np.roll(vec.astype(float), lag)))


def _periodicity_profile(divider_preds: List[Dict], n_dividers: int) -> Dict[str, float | int | str]:
    n = int(max(1, n_dividers))
    if not divider_preds:
        return {
            "periodicity_score": 0.0,
            "periodicity_kick": 0.0,
            "periodicity_snare": 0.0,
            "periodicity_hihat": 0.0,
            "periodicity_best_lag": 0,
            "periodicity_pattern": "",
        }

    kick = np.array([_prob_from_prediction(p, ("kick", "bd")) for p in divider_preds], dtype=float)
    snare = np.array([_prob_from_prediction(p, ("snare", "clap")) for p in divider_preds], dtype=float)
    hihat = np.array([_prob_from_prediction(p, ("hihat", "hat")) for p in divider_preds], dtype=float)

    lag_candidates: List[int] = []
    if n >= 4:
        lag_candidates.append(n // 4)
    if n >= 2:
        lag_candidates.append(n // 2)
    lag_candidates = sorted({lag for lag in lag_candidates if 0 < lag < n})
    if not lag_candidates and n > 1:
        lag_candidates = [1]

    best_lag = 0
    best_lag_score = -1.0
    for lag in lag_candidates:
        s = float(np.mean([_lag_similarity(kick, lag), _lag_similarity(snare, lag), _lag_similarity(hihat, lag)]))
        if s > best_lag_score:
            best_lag_score = s
            best_lag = int(lag)

    p_kick = max((_lag_similarity(kick, lag) for lag in lag_candidates), default=0.0)
    p_snare = max((_lag_similarity(snare, lag) for lag in lag_candidates), default=0.0)
    p_hihat = max((_lag_similarity(hihat, lag) for lag in lag_candidates), default=0.0)
    p_overall = float(0.50 * p_kick + 0.35 * p_snare + 0.15 * p_hihat)

    symbols: List[str] = []
    for i in range(min(n, len(divider_preds))):
        vals = {"K": float(kick[i]), "S": float(snare[i]), "H": float(hihat[i])}
        lbl, v = max(vals.items(), key=lambda kv: kv[1])
        symbols.append(lbl if v >= 0.20 else "x")

    return {
        "periodicity_score": p_overall,
        "periodicity_kick": float(p_kick),
        "periodicity_snare": float(p_snare),
        "periodicity_hihat": float(p_hihat),
        "periodicity_best_lag": int(best_lag),
        "periodicity_pattern": "".join(symbols),
    }


def analyze_file(
    file_path: Path,
    drum_bundle: Optional[Dict],
    sr: int,
    min_conf: float,
    structure_threshold: float,
) -> Optional[Dict]:
    try:
        y, fs = librosa.load(str(file_path), sr=sr, mono=True)
    except Exception:
        return None

    if y is None or len(y) < 4096:
        return None

    y = _normalize_audio(y)

    # Prefer a trimmed body to focus on actual loop core.
    try:
        y_trim, _ = librosa.effects.trim(y, top_db=28)
        if len(y_trim) > 4096:
            y = y_trim
    except Exception:
        pass

    sr_i = int(fs)

    # Run drum-informed structure selection first (when drum model is available).
    score4 = _structure_scores_for_dividers(
        y,
        sr=sr_i,
        n_dividers=4,
        drum_bundle=drum_bundle,
        min_conf=min_conf,
        drum_weight=0.18,
    )
    score8 = _structure_scores_for_dividers(
        y,
        sr=sr_i,
        n_dividers=8,
        drum_bundle=drum_bundle,
        min_conf=min_conf,
        drum_weight=0.18,
    )

    # Choose divider count based on combined (structure + drum) score, with mild 4-divider simplicity bias.
    adjusted4 = score4.get("combined_score", score4["structure_score"]) + 0.015
    adjusted8 = score8.get("combined_score", score8["structure_score"])
    chosen = 4 if adjusted4 >= adjusted8 else 8
    best = score4 if chosen == 4 else score8

    # Then estimate tempo/beat grid for boundary snapping.
    tempo = _estimate_tempo(y, sr=sr_i)
    beat_times = _estimate_beat_times(y, sr=sr_i)

    structure_pattern = _assign_symbol_pattern(best["similarity_matrix"], threshold=structure_threshold)

    linear_boundaries = np.linspace(0.0, float(len(y) / fs), chosen + 1).tolist()
    max_shift = max(0.03, 0.35 * (float(len(y) / fs) / float(chosen)))
    snapped_boundaries = _snap_boundaries_to_beats(linear_boundaries, beat_times=beat_times, max_shift_sec=max_shift)

    segments = _segments_from_boundaries(y, sr=sr_i, boundaries_sec=snapped_boundaries)
    if len(segments) != chosen:
        segments = best["segments"]
        snapped_boundaries = linear_boundaries

    durations = [float(len(seg) / sr_i) for seg in segments]
    divider_preds = _divider_drum_predictions(segments, sr=sr_i, drum_bundle=drum_bundle, min_conf=min_conf)

    return {
        "file": str(file_path),
        "duration_sec": float(len(y) / fs),
        "tempo_est": float(tempo),
        "chosen_dividers": int(chosen),
        "structure_score": float(best["structure_score"]),
        "drum_alignment_score": float(best.get("drum_alignment_score", 0.0)),
        "periodicity_score": float(best.get("periodicity_score", 0.0)),
        "periodicity_kick": float(best.get("periodicity_kick", 0.0)),
        "periodicity_snare": float(best.get("periodicity_snare", 0.0)),
        "periodicity_hihat": float(best.get("periodicity_hihat", 0.0)),
        "periodicity_best_lag": int(best.get("periodicity_best_lag", 0)),
        "periodicity_pattern": str(best.get("periodicity_pattern", "")),
        "combined_structure_score": float(best.get("combined_score", best["structure_score"])),
        "repetition_score": float(best["repetition_score"]),
        "half_similarity": float(best["half_similarity"]),
        "superimpose_similarity": float(best["superimpose_similarity"]),
        "score_4_dividers": float(score4["structure_score"]),
        "score_8_dividers": float(score8["structure_score"]),
        "score_4_periodicity": float(score4.get("periodicity_score", 0.0)),
        "score_8_periodicity": float(score8.get("periodicity_score", 0.0)),
        "score_4_combined": float(score4.get("combined_score", score4["structure_score"])),
        "score_8_combined": float(score8.get("combined_score", score8["structure_score"])),
        "structure_pattern": structure_pattern,
        "divider_boundaries_sec": snapped_boundaries,
        "divider_durations_sec": durations,
        "divider_drum_predictions": divider_preds,
        "offset_samples": int(best["offset_samples"]),
    }


def analyze_folder(
    folder: Path,
    out_json: Path,
    out_csv: Path,
    drum_model_path: Optional[Path],
    sr: int = 22050,
    min_conf: float = 0.25,
    structure_threshold: float = 0.82,
    max_files: Optional[int] = None,
) -> None:
    drum_bundle: Optional[Dict] = None
    if drum_model_path is not None and drum_model_path.exists():
        drum_bundle = joblib.load(drum_model_path)

    files = collect_audio_files(folder)
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    details: List[Dict] = []
    rows: List[Dict] = []

    for i, p in enumerate(files, start=1):
        result = analyze_file(
            p,
            drum_bundle=drum_bundle,
            sr=sr,
            min_conf=min_conf,
            structure_threshold=structure_threshold,
        )
        if result is None:
            continue

        details.append(result)

        divider_labels = [d.get("predicted_label", "") for d in result["divider_drum_predictions"]]
        rows.append(
            {
                "file": result["file"],
                "duration_sec": result["duration_sec"],
                "tempo_est": result["tempo_est"],
                "chosen_dividers": result["chosen_dividers"],
                "structure_pattern": result["structure_pattern"],
                "structure_score": result["structure_score"],
                "periodicity_score": result.get("periodicity_score", 0.0),
                "periodicity_pattern": result.get("periodicity_pattern", ""),
                "repetition_score": result["repetition_score"],
                "half_similarity": result["half_similarity"],
                "superimpose_similarity": result["superimpose_similarity"],
                "divider_labels": ",".join(divider_labels),
            }
        )

        if i % 150 == 0:
            print(f"analyzed {i}/{len(files)} files")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_files": len(details),
        "folder": str(folder),
        "drum_model": str(drum_model_path) if drum_model_path else None,
        "details": details,
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved structure details JSON: {out_json}")
    print(f"Saved structure summary CSV: {out_csv}")
    print(f"Rows: {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze loop structure using repetition/superimposition and classify drum sounds per divider"
    )
    parser.add_argument("--folder", required=True, help="Folder containing loop audio files")
    parser.add_argument("--drum_model", default="training/models/drum_sound_classifier.joblib", help="Optional drum-sound model bundle")
    parser.add_argument("--out_json", default="training/models/loop_structure_analysis.json", help="Detailed JSON output")
    parser.add_argument("--out_csv", default="training/models/loop_structure_analysis.csv", help="Summary CSV output")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--min_conf", type=float, default=0.25, help="Min confidence for drum labels per divider")
    parser.add_argument("--structure_threshold", type=float, default=0.82, help="Similarity threshold for symbolic structure pattern")
    parser.add_argument("--max_files", type=int, default=None, help="Optional file cap for quick runs")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    drum_model_path = Path(args.drum_model) if args.drum_model else None
    analyze_folder(
        folder=folder,
        out_json=Path(args.out_json),
        out_csv=Path(args.out_csv),
        drum_model_path=drum_model_path,
        sr=max(8000, int(args.sr)),
        min_conf=max(0.0, min(1.0, float(args.min_conf))),
        structure_threshold=max(0.5, min(0.99, float(args.structure_threshold))),
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
