from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def normalize_audio(y: np.ndarray, target_dbfs: float = -14.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    gain = 10 ** ((target_dbfs - 20 * np.log10(rms + 1e-8)) / 20)
    return (y * gain).astype(np.float32, copy=False)


def isolate_harmonic(y: np.ndarray) -> np.ndarray:
    """Approximate harmonic stem via HPSS."""

    try:
        y_h, _y_p = librosa.effects.hpss(y)
        return y_h.astype(np.float32, copy=False)
    except Exception:
        return y.astype(np.float32, copy=False)


def detect_note_onsets(
    y: np.ndarray,
    sr: int,
    *,
    use_harmonic: bool = True,
    backtrack: bool = True,
    onset_delta: float = 0.12,
    hop_length: int = 512,
) -> np.ndarray:
    """Detect note onsets.

    For "note hits" (non-drum transients), harmonic onset detection usually works better,
    so we default to using the harmonic stem.

    Returns onset times in seconds relative to the provided signal.
    """

    y0 = isolate_harmonic(y) if use_harmonic else y

    # For tonal material, giving onset_detect access to the waveform tends to help.
    onset_frames = librosa.onset.onset_detect(
        y=y0,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        backtrack=backtrack,
        delta=onset_delta,
        # Conservative defaults that still pick up synth attacks reasonably well
        pre_max=10,
        post_max=10,
        pre_avg=20,
        post_avg=20,
        wait=10,
    )
    if onset_frames.size == 0:
        return np.array([], dtype=np.float32)
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length).astype(np.float32)


def snap_boundaries_to_onsets(
    duration_s: float,
    *,
    onset_times: np.ndarray,
    n_slices: int,
    max_snap_s: float = 0.10,
    min_slice_s: float = 0.03,
) -> List[float]:
    """Create slice boundaries (including 0 and duration), snapping to nearest onset.

    We start from equally spaced ideal boundaries, then snap each to the nearest onset
    if it's within max_snap_s. Enforces monotonicity and a minimum slice length.
    """

    if n_slices < 2:
        return [0.0, float(duration_s)]

    duration_s = float(max(0.0, duration_s))
    if duration_s <= 0.0:
        return [0.0, 0.0]

    # Candidate onsets, excluding extremes
    onsets = onset_times
    onsets = onsets[np.isfinite(onsets)]
    onsets = onsets[(onsets > 0.0) & (onsets < duration_s)]

    boundaries: List[float] = [0.0]

    for k in range(1, n_slices):
        ideal = duration_s * (k / n_slices)
        chosen = ideal

        if onsets.size:
            i = int(np.argmin(np.abs(onsets - ideal)))
            if abs(float(onsets[i]) - ideal) <= max_snap_s:
                chosen = float(onsets[i])

        # Enforce monotonicity and minimum length
        chosen = max(chosen, boundaries[-1] + min_slice_s)
        # Never exceed end
        chosen = min(chosen, duration_s)
        boundaries.append(chosen)

    # Force exact end
    boundaries[-1] = duration_s

    # Ensure strictly increasing by nudging if needed
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = min(duration_s, boundaries[i - 1] + min_slice_s)

    boundaries[-1] = duration_s
    return boundaries


def export_slices(
    *,
    y: np.ndarray,
    sr: int,
    outdir: Path,
    prefix: str,
    boundaries_s: Sequence[float],
    normalize: bool = True,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for i in range(len(boundaries_s) - 1):
        a_s = float(boundaries_s[i])
        b_s = float(boundaries_s[i + 1])
        a = int(a_s * sr)
        b = int(b_s * sr)
        b = min(b, len(y))
        a = min(a, b)

        seg = y[a:b].astype(np.float32, copy=False)
        if seg.size == 0:
            continue

        if normalize:
            seg = normalize_audio(seg)

        p = outdir / f"{prefix}_slice_{i:03d}.wav"
        sf.write(str(p), seg, sr)

        rows.append(
            {
                "slice_index": i,
                "start_s": a_s,
                "end_s": b_s,
                "duration_s": b_s - a_s,
                "wav": str(p),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"{prefix}_slices.csv", index=False)
    return df


def slice_loop_on_note_hits(
    *,
    y_loop: np.ndarray,
    sr: int,
    n_slices: int,
    max_snap_s: float = 0.10,
) -> Tuple[List[float], np.ndarray]:
    """Compute slice boundaries for a loop by snapping equal divisions to note onsets."""

    dur = float(len(y_loop) / sr)
    onsets = detect_note_onsets(y_loop, sr, use_harmonic=True)
    # If harmonic stem is too "smooth" (HPSS can remove attacks), fall back to full signal.
    if onsets.size < max(2, n_slices // 2):
        onsets_full = detect_note_onsets(y_loop, sr, use_harmonic=False)
        if onsets_full.size > onsets.size:
            onsets = onsets_full
    boundaries = snap_boundaries_to_onsets(
        dur,
        onset_times=onsets,
        n_slices=n_slices,
        max_snap_s=max_snap_s,
    )
    return boundaries, onsets
