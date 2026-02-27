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


def _structure_scores_for_dividers(y: np.ndarray, sr: int, n_dividers: int, phase_steps: int = 24) -> Dict:
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

        entry = {
            "offset_samples": int(offset),
            "repetition_score": repetition_score,
            "half_similarity": float(half_similarity),
            "superimpose_similarity": float(superimpose),
            "structure_score": structure_score,
            "segments": segments,
            "similarity_matrix": sims.tolist(),
        }

        if best is None or entry["structure_score"] > best["structure_score"]:
            best = entry

    if best is None:
        return {
            "offset_samples": 0,
            "repetition_score": 0.0,
            "half_similarity": 0.0,
            "superimpose_similarity": 0.0,
            "structure_score": 0.0,
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

    tempo = _estimate_tempo(y, sr=fs)
    beat_times = _estimate_beat_times(y, sr=fs)

    score4 = _structure_scores_for_dividers(y, sr=fs, n_dividers=4)
    score8 = _structure_scores_for_dividers(y, sr=fs, n_dividers=8)

    # Choose divider count based on stronger structure score, with a mild 4-divider simplicity bias.
    adjusted4 = score4["structure_score"] + 0.015
    adjusted8 = score8["structure_score"]
    chosen = 4 if adjusted4 >= adjusted8 else 8
    best = score4 if chosen == 4 else score8

    structure_pattern = _assign_symbol_pattern(best["similarity_matrix"], threshold=structure_threshold)

    linear_boundaries = np.linspace(0.0, float(len(y) / fs), chosen + 1).tolist()
    max_shift = max(0.03, 0.35 * (float(len(y) / fs) / float(chosen)))
    snapped_boundaries = _snap_boundaries_to_beats(linear_boundaries, beat_times=beat_times, max_shift_sec=max_shift)

    segments = _segments_from_boundaries(y, sr=fs, boundaries_sec=snapped_boundaries)
    if len(segments) != chosen:
        segments = best["segments"]
        snapped_boundaries = linear_boundaries

    durations = [float(len(seg) / fs) for seg in segments]
    divider_preds = _divider_drum_predictions(segments, sr=fs, drum_bundle=drum_bundle, min_conf=min_conf)

    return {
        "file": str(file_path),
        "duration_sec": float(len(y) / fs),
        "tempo_est": float(tempo),
        "chosen_dividers": int(chosen),
        "structure_score": float(best["structure_score"]),
        "repetition_score": float(best["repetition_score"]),
        "half_similarity": float(best["half_similarity"]),
        "superimpose_similarity": float(best["superimpose_similarity"]),
        "score_4_dividers": float(score4["structure_score"]),
        "score_8_dividers": float(score8["structure_score"]),
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
