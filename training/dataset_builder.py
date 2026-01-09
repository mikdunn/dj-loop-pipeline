from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .feature_extraction import extract_full_features

@dataclass
class Candidate:
    track_id: str
    start_time: float
    end_time: float
    bars: int
    center_time: float

def generate_bar_aligned_candidates(track_id, beats, bpm, bars_options=[1,2,4]):
    candidates = []
    seconds_per_beat = 60.0 / bpm
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
    seconds_per_beat = 60.0 / bpm
    sigma_sec = sigma_beats * seconds_per_beat
    weight = 0.0
    for t in sample_times:
        dist = candidate_center - t
        weight += np.exp(-0.5 * (dist / sigma_sec)**2)
    return float(weight)

def build_training_dataset(audio_dir, timestamps, bars_options=[1,2,4]):
    rows = []
    for audio_path in audio_dir.glob("*.wav"):
        import librosa
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        bpm, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        candidates = generate_bar_aligned_candidates(audio_path.stem, beats, bpm, bars_options)
        sample_times = timestamps.get(audio_path.stem, [])
        for cand in candidates:
            weight = compute_candidate_weight(cand.center_time, sample_times, bpm)
            feats = extract_full_features(y, sr, cand.start_time, cand.end_time)
            if feats:
                row = {
                    "track_id": audio_path.stem,
                    "start_time": cand.start_time,
                    "end_time": cand.end_time,
                    "bars": cand.bars,
                    "weight": weight
                }
                row.update(feats)
                rows.append(row)
    return pd.DataFrame(rows)
