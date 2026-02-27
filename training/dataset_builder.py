from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class Candidate:
    track_id: str
    start_time: float
    end_time: float
    bars: int
    bpm: float

    @property
    def center_time(self) -> float:
        return (self.start_time + self.end_time) * 0.5


def generate_bar_aligned_candidates(
    track_id: str,
    beats: Sequence[float],
    bpm: float,
    bars_list: Iterable[int] = (1, 2, 4),
    beats_per_bar: int = 4,
) -> List[Candidate]:
    """Create bar-aligned loop candidates from beat timestamps."""
    if beats is None:
        return []

    beat_times = np.asarray(beats, dtype=float)
    if beat_times.size < beats_per_bar * 2:
        return []

    safe_bpm = float(bpm) if bpm and np.isfinite(bpm) and bpm > 0 else 120.0
    candidates: List[Candidate] = []

    for bars in bars_list:
        span_beats = int(bars) * beats_per_bar
        if span_beats <= 0:
            continue
        for i in range(0, beat_times.size - span_beats):
            start_t = float(beat_times[i])
            end_t = float(beat_times[i + span_beats])
            if end_t <= start_t:
                continue
            candidates.append(
                Candidate(
                    track_id=track_id,
                    start_time=start_t,
                    end_time=end_t,
                    bars=int(bars),
                    bpm=safe_bpm,
                )
            )
    return candidates


def compute_candidate_weight(center_time: float, timestamps: Sequence[float], bpm: float) -> float:
    """Compute a proximity weight to user-provided reference timestamps."""
    if not timestamps:
        return 1.0

    safe_bpm = float(bpm) if bpm and np.isfinite(bpm) and bpm > 0 else 120.0
    bar_seconds = (60.0 / safe_bpm) * 4.0
    sigma = max(0.5, bar_seconds)

    dists = [abs(float(ts) - center_time) for ts in timestamps]
    nearest = min(dists) if dists else 0.0
    return float(np.exp(-0.5 * (nearest / sigma) ** 2))
