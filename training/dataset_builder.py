from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .feature_extraction import build_track_feature_cache, extract_full_features

@dataclass
class Candidate:
    track_id: str
    start_time: float
    end_time: float
    bars: int
    center_time: float

def generate_bar_aligned_candidates(track_id, beats, bpm, bars_options=[1,2,4]):
    candidates = []
    for bars in bars_options:
        beats_per_loop = bars * 4
        for i in range(0, len(beats) - beats_per_loop):
            start_time = beats[i]
            end_time = beats[i + beats_per_loop]
            center_time = (start_time + end_time)/2.0
            candidates.append(Candidate(track_id, start_time, end_time, bars, center_time))
    return candidates

def compute_candidate_weight(candidate_center, sample_times, bpm, sigma_beats=0.5):
    if not sample_times:
        return 0.0

    # librosa can return bpm=0 for very short/silent/synthetic audio.
    # Fall back to a reasonable default rather than crashing.
    bpm = float(bpm)
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0

    seconds_per_beat = 60.0 / bpm
    sigma_sec = sigma_beats * seconds_per_beat
    weight = 0.0
    for t in sample_times:
        dist = candidate_center - t
        weight += np.exp(-0.5 * (dist / sigma_sec)**2)
    return float(weight)

def build_training_dataset(audio_dir, timestamps, bars_options=[1,2,4]):
    return build_training_dataset_parallel(audio_dir, timestamps, bars_options=bars_options, n_jobs=1)


def _process_one_track(audio_path: Path, timestamps: dict, bars_options: list[int]):
    import librosa

    y, sr = librosa.load(audio_path, sr=44100, mono=True)
    cache = build_track_feature_cache(y, sr)

    bpm, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")

    # Fallback when beat tracking fails (e.g., silence, synthetic audio, very short clips)
    bpm = float(bpm)
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0

    max_bars = int(max(bars_options)) if bars_options else 4
    min_beats_needed = max_bars * 4 + 1
    if beats is None or len(beats) < min_beats_needed:
        dur_s = float(len(y)) / float(sr)
        spb = 60.0 / bpm
        beats = np.arange(0.0, dur_s + 1e-6, spb, dtype=np.float32)

    candidates = generate_bar_aligned_candidates(audio_path.stem, beats, bpm, bars_options)
    sample_times = timestamps.get(audio_path.stem, [])

    rows = []
    for cand in candidates:
        weight = compute_candidate_weight(cand.center_time, sample_times, bpm)
        feats = extract_full_features(y, sr, cand.start_time, cand.end_time, cache=cache)
        if feats:
            row = {
                "track_id": audio_path.stem,
                "start_time": cand.start_time,
                "end_time": cand.end_time,
                "bars": cand.bars,
                "weight": weight,
            }
            row.update(feats)
            rows.append(row)
    return rows


def build_training_dataset_parallel(
    audio_dir: Path,
    timestamps: dict,
    *,
    bars_options=[1, 2, 4],
    n_jobs: int = 1,
):
    """Build dataset, optionally parallelizing across tracks.

    n_jobs=1 -> sequential (default)
    n_jobs>1 -> joblib parallel across WAV files
    """

    audio_paths = list(audio_dir.glob("*.wav"))
    if not audio_paths:
        return pd.DataFrame([])

    bars_options = list(bars_options)

    if int(n_jobs) <= 1:
        rows = []
        for p in audio_paths:
            rows.extend(_process_one_track(p, timestamps, bars_options))
        return pd.DataFrame(rows)

    try:
        from joblib import Parallel, delayed
    except Exception as e:
        raise ImportError("joblib is required for n_jobs>1") from e

    results = Parallel(n_jobs=int(n_jobs), prefer="processes")(
        delayed(_process_one_track)(p, timestamps, bars_options) for p in audio_paths
    )

    rows = []
    for part in results:
        rows.extend(part)
    return pd.DataFrame(rows)
