from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def normalize_audio(y: np.ndarray, target_dbfs: float = -14.0) -> np.ndarray:
    """Simple RMS-based normalization (not true LUFS)."""

    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    gain = 10 ** ((target_dbfs - 20 * np.log10(rms + 1e-8)) / 20)
    return (y * gain).astype(np.float32, copy=False)


def isolate_percussive(y: np.ndarray) -> np.ndarray:
    """Return an approximate drum/percussive stem via HPSS."""

    try:
        _y_h, y_p = librosa.effects.hpss(y)
        return y_p.astype(np.float32, copy=False)
    except Exception:
        return y.astype(np.float32, copy=False)


def isolate_harmonic(y: np.ndarray) -> np.ndarray:
    """Return an approximate harmonic stem via HPSS."""

    try:
        y_h, _y_p = librosa.effects.hpss(y)
        return y_h.astype(np.float32, copy=False)
    except Exception:
        return y.astype(np.float32, copy=False)


def _safe_rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))


def classify_one_shot(y_hit: np.ndarray, sr: int) -> Dict[str, object]:
    """Classify a one-shot into a drum kit piece or a synth sound (heuristic).

    This is designed for labeling exports into folders/metadata (not perfect MIR).
    Returns:
      - bank_type: 'drum_kit' | 'synth' | 'unknown'
      - class_label: e.g. 'kick', 'snare', 'hat', 'tom', 'cymbal', 'perc', 'bass', 'lead', ...
      - confidence: float in [0, 1]

    Heuristics used:
      - HPSS energy ratio (harmonic vs percussive)
      - spectral centroid / rolloff / flatness
      - zero crossing rate
      - rough pitch stability (YIN)
    """

    y_hit = np.asarray(y_hit, dtype=np.float32)
    dur_s = float(len(y_hit) / sr) if sr > 0 else 0.0
    rms = _safe_rms(y_hit)
    if dur_s <= 0.0 or rms <= 1e-6:
        return {"bank_type": "unknown", "class_label": "silence", "confidence": 0.0}

    # Basic spectral features
    try:
        centroid = librosa.feature.spectral_centroid(y=y_hit, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y_hit, sr=sr, roll_percent=0.85)[0]
        flatness = librosa.feature.spectral_flatness(y=y_hit)[0]
        zcr = librosa.feature.zero_crossing_rate(y_hit)[0]

        c_mean = float(np.mean(centroid))
        r_mean = float(np.mean(rolloff))
        f_mean = float(np.mean(flatness))
        z_mean = float(np.mean(zcr))
    except Exception:
        c_mean, r_mean, f_mean, z_mean = 0.0, 0.0, 0.0, 0.0

    # HPSS ratios help separate tonal synth hits from noisy drums
    y_h = isolate_harmonic(y_hit)
    y_p = isolate_percussive(y_hit)
    harm_rms = _safe_rms(y_h)
    perc_rms = _safe_rms(y_p)
    total = harm_rms + perc_rms + 1e-8
    harm_ratio = float(harm_rms / total)
    perc_ratio = float(perc_rms / total)

    # Rough pitch estimate (only meaningful if harmonic dominates)
    f0_med: Optional[float]
    f0_std: Optional[float]
    try:
        f0 = librosa.yin(y_h, fmin=50, fmax=2000, sr=sr)
        f0 = f0[np.isfinite(f0)]
        if f0.size >= 5:
            f0_med = float(np.median(f0))
            f0_std = float(np.std(f0))
        else:
            f0_med, f0_std = None, None
    except Exception:
        f0_med, f0_std = None, None

    pitch_stable = bool(f0_med is not None and f0_std is not None and f0_std < 25.0)

    # Decide bank type
    # Strong harmonic ratio + stable pitch => synth-like.
    if harm_ratio > 0.65 and pitch_stable:
        bank_type = "synth"
    elif perc_ratio > 0.55:
        bank_type = "drum_kit"
    else:
        # mixed: lean drum if it looks noisy/high-ZCR, else synth
        bank_type = "drum_kit" if (f_mean > 0.2 or z_mean > 0.08) else "synth"

    # Subclass
    class_label = "unknown"
    confidence = 0.35

    if bank_type == "drum_kit":
        # Kick: low centroid, low zcr, often longer tail
        if c_mean < 1500 and z_mean < 0.06 and (f0_med is None or f0_med < 200):
            class_label = "kick"
            confidence = 0.75
        # Hat: very bright + high zcr, often short
        elif c_mean > 6000 and z_mean > 0.10 and dur_s < 0.25:
            class_label = "hat"
            confidence = 0.75
        # Cymbal: bright and longer than hat
        elif c_mean > 7000 and dur_s >= 0.20:
            class_label = "cymbal"
            confidence = 0.65
        # Snare: mid/high centroid + noisy
        elif 2000 < c_mean < 7000 and (f_mean > 0.25 or z_mean > 0.08):
            class_label = "snare"
            confidence = 0.60
        # Tom: somewhat tonal and low-mid
        elif pitch_stable and f0_med is not None and 120 <= f0_med <= 450 and c_mean < 3500:
            class_label = "tom"
            confidence = 0.55
        else:
            class_label = "perc"
            confidence = 0.45
    else:
        # Synth bank (very rough)
        if pitch_stable and f0_med is not None:
            if f0_med < 200:
                class_label = "bass"
                confidence = 0.65
            elif f0_med < 800:
                class_label = "lead"
                confidence = 0.55
            else:
                class_label = "lead_high"
                confidence = 0.50
        else:
            # No stable pitch: could be fx/noise/stab
            class_label = "fx" if f_mean > 0.35 else "stab"
            confidence = 0.45

    return {
        "bank_type": bank_type,
        "class_label": class_label,
        "confidence": float(confidence),
        "features": {
            "duration_s": dur_s,
            "rms": float(rms),
            "centroid_mean": float(c_mean),
            "rolloff_mean": float(r_mean),
            "flatness_mean": float(f_mean),
            "zcr_mean": float(z_mean),
            "harm_ratio": float(harm_ratio),
            "perc_ratio": float(perc_ratio),
            "f0_median": None if f0_med is None else float(f0_med),
            "f0_std": None if f0_std is None else float(f0_std),
        },
    }


def compute_break_score(features_row: pd.Series) -> float:
    """Heuristic score to prioritize drum-break-like loops.

    Higher is "more likely drum break":
    - prefer high percussive ratio
    - prefer low boundary activity (clean edges)
    """

    perc_ratio = float(features_row.get("perc_to_total_rms", 0.0))
    boundary = float(features_row.get("boundary_quiet_score", 0.0))
    onset = float(features_row.get("perc_onset_mean", 0.0))

    # Weighting chosen empirically; tune as needed.
    return 2.0 * perc_ratio + 0.5 * onset - 0.75 * boundary


@dataclass
class HitSlice:
    start_s: float
    end_s: float
    peak_s: float


def slice_drum_hits(
    y_perc: np.ndarray,
    sr: int,
    *,
    pre_s: float = 0.010,
    post_s: float = 0.180,
    min_len_s: float = 0.060,
    max_len_s: float = 0.500,
    backtrack: bool = True,
    onset_delta: float = 0.2,
) -> List[HitSlice]:
    """Slice individual drum hits from a percussive signal.

    Uses onset detection. Each hit is a fixed-ish window around the detected onset,
    capped by min/max lengths.

    This is intentionally conservative (better to get clean one-shots than long phrases).
    """

    if y_perc.size == 0:
        return []

    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        units="frames",
        backtrack=backtrack,
        delta=onset_delta,
    )

    if onset_frames.size == 0:
        return []

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    hits: List[HitSlice] = []
    for t in onset_times:
        start = max(0.0, float(t - pre_s))
        end = float(t + post_s)
        # enforce length bounds
        end = max(end, start + min_len_s)
        end = min(end, start + max_len_s)
        hits.append(HitSlice(start_s=start, end_s=end, peak_s=float(t)))

    # Merge overlapping slices
    hits.sort(key=lambda h: h.start_s)
    merged: List[HitSlice] = []
    for h in hits:
        if not merged:
            merged.append(h)
            continue
        prev = merged[-1]
        if h.start_s <= prev.end_s:
            merged[-1] = HitSlice(
                start_s=prev.start_s,
                end_s=max(prev.end_s, h.end_s),
                peak_s=prev.peak_s,
            )
        else:
            merged.append(h)

    return merged


def export_wav(path: Path, y: np.ndarray, sr: int, *, normalize: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if normalize:
        y = normalize_audio(y)
    sf.write(str(path), y, sr)


def export_drum_break_and_hits(
    *,
    y: np.ndarray,
    sr: int,
    track_id: str,
    start_time: float,
    end_time: float,
    outdir: Path,
    break_index: int,
    slice_hits: bool = True,
) -> Dict:
    """Export one identified drum break loop and optionally its one-shots."""

    a = int(max(0.0, start_time) * sr)
    b = int(max(0.0, end_time) * sr)
    b = min(b, len(y))
    a = min(a, b)

    seg = y[a:b].astype(np.float32, copy=False)
    y_perc = isolate_percussive(seg)

    break_dir = outdir / track_id / f"break_{break_index:02d}"
    loop_path = break_dir / "drum_break.wav"
    perc_path = break_dir / "drum_break_perc.wav"

    export_wav(loop_path, seg, sr, normalize=True)
    export_wav(perc_path, y_perc, sr, normalize=True)

    meta: Dict = {
        "track_id": track_id,
        "break_index": break_index,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "duration": float(end_time - start_time),
        "loop_wav": str(loop_path),
        "perc_wav": str(perc_path),
    }

    if slice_hits:
        hits = slice_drum_hits(y_perc, sr)
        hit_paths: List[str] = []
        hit_meta_rows: List[Dict[str, object]] = []
        for i, h in enumerate(hits):
            ha = int(h.start_s * sr)
            hb = int(h.end_s * sr)
            hb = min(hb, len(y_perc))
            ha = min(ha, hb)

            # Export BOTH:
            # - full-mix one-shot (useful for synth/kit banks)
            # - percussive-only one-shot (cleaner drums)
            y_hit_full = seg[ha:hb]
            y_hit_perc = y_perc[ha:hb]

            if y_hit_full.size < int(0.03 * sr):
                continue

            hit_dir = break_dir / "hits"
            hit_path = hit_dir / f"hit_{i:04d}.wav"
            hit_perc_path = hit_dir / f"hit_{i:04d}_perc.wav"

            export_wav(hit_path, y_hit_full, sr, normalize=True)
            export_wav(hit_perc_path, y_hit_perc, sr, normalize=True)

            label = classify_one_shot(y_hit_full, sr)

            hit_paths.append(str(hit_path))
            hit_meta_rows.append(
                {
                    "hit_index": i,
                    "start_s": float(h.start_s),
                    "end_s": float(h.end_s),
                    "peak_s": float(h.peak_s),
                    "hit_wav": str(hit_path),
                    "hit_perc_wav": str(hit_perc_path),
                    "bank_type": label.get("bank_type"),
                    "class_label": label.get("class_label"),
                    "confidence": float(label.get("confidence", 0.0)),
                    # Flattened features
                    **{f"feat_{k}": v for k, v in (label.get("features") or {}).items()},
                }
            )

        meta["num_hits"] = len(hit_paths)
        meta["hit_wavs"] = hit_paths

        # Write per-hit labels/metadata for building sample packs.
        if hit_meta_rows:
            hits_df = pd.DataFrame(hit_meta_rows)
            hits_df.to_csv(break_dir / "hits" / "hits_metadata.csv", index=False)

    return meta


def export_top_drum_breaks_from_ranked_candidates(
    *,
    df_ranked: pd.DataFrame,
    audio_file: Path,
    outdir: Path,
    top_k: int = 5,
    slice_hits: bool = True,
) -> pd.DataFrame:
    """Given a ranked candidate dataframe (from LoopPipelineML), pick drum breaks and export them.

    Uses a heuristic break score based on percussive ratio + boundary cleanliness.

    Returns a metadata dataframe.
    """

    y, sr = librosa.load(str(audio_file), sr=44100, mono=True)

    df = df_ranked.copy()
    df["break_score"] = df.apply(compute_break_score, axis=1)
    df = df.sort_values(["break_score"], ascending=False)

    metas: List[Dict] = []
    for i, row in df.head(top_k).iterrows():
        meta = export_drum_break_and_hits(
            y=y,
            sr=sr,
            track_id=str(row.get("track_id", audio_file.stem)),
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            outdir=outdir,
            break_index=len(metas),
            slice_hits=slice_hits,
        )
        meta["break_score"] = float(row["break_score"])
        # Preserve model score too if present
        if "score" in row:
            meta["ml_score"] = float(row["score"])
        metas.append(meta)

    meta_df = pd.DataFrame(metas)
    outdir.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(outdir / f"{audio_file.stem}_drum_breaks.csv", index=False)
    return meta_df
