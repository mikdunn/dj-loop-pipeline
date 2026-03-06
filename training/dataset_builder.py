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


def choose_adaptive_bar_options(
    beats: Sequence[float],
    bpm: float,
    bars_candidates: Iterable[int] = (4, 8),
    beats_per_bar: int = 4,
    margin: float = 0.03,
) -> List[int]:
    """Choose between 4 and 8 bars using beat-grid consistency.

    Returns one option when evidence is clear, otherwise returns both options.
    """
    bars = sorted({int(b) for b in bars_candidates if int(b) > 0})
    if not bars:
        return [4, 8]

    if 4 not in bars or 8 not in bars:
        return bars

    beat_times = np.asarray(beats, dtype=float)
    if beat_times.size < 40:
        return [4, 8]

    safe_bpm = float(bpm) if bpm and np.isfinite(bpm) and bpm > 0 else 120.0
    sec_per_beat = 60.0 / safe_bpm

    def _score_for_bars(n_bars: int) -> float:
        span = int(n_bars) * int(beats_per_bar)
        if beat_times.size <= span:
            return 0.0

        durations = beat_times[span:] - beat_times[:-span]
        durations = durations[np.isfinite(durations) & (durations > 0)]
        if durations.size < 4:
            return 0.0

        expected = max(1e-6, span * sec_per_beat)
        err = np.abs(durations - expected) / expected
        mean_err = float(np.mean(err))

        mean_d = float(np.mean(durations))
        std_d = float(np.std(durations))
        cv = std_d / (mean_d + 1e-8)

        consistency = float(np.exp(-mean_err))
        stability = float(1.0 / (1.0 + cv))
        support = float(min(1.0, durations.size / 32.0))
        return 0.45 * consistency + 0.45 * stability + 0.10 * support

    score_4 = _score_for_bars(4)
    score_8 = _score_for_bars(8)

    if score_8 > score_4 + float(margin):
        return [8]
    if score_4 > score_8 + float(margin):
        return [4]
    return [4, 8]


def generate_bar_aligned_candidates(
    track_id: str,
    beats: Sequence[float],
    bpm: float,
    bars_list: Iterable[int] = (1, 2, 4),
    beats_per_bar: int = 4,
    adaptive_4_vs_8: bool = False,
    adaptive_margin: float = 0.03,
) -> List[Candidate]:
    """Create bar-aligned loop candidates from beat timestamps."""
    if beats is None:
        return []

    beat_times = np.asarray(beats, dtype=float)
    bars_to_use = [int(b) for b in bars_list if int(b) > 0]
    if not bars_to_use:
        return []

    min_span_beats = min(int(bars) * int(beats_per_bar) for bars in bars_to_use)
    # Need at least one full span plus the terminal beat timestamp.
    if beat_times.size <= min_span_beats:
        return []

    safe_bpm = float(bpm) if bpm and np.isfinite(bpm) and bpm > 0 else 120.0
    candidates: List[Candidate] = []

    if adaptive_4_vs_8:
        bars_to_use = choose_adaptive_bar_options(
            beats=beat_times,
            bpm=safe_bpm,
            bars_candidates=bars_to_use,
            beats_per_bar=beats_per_bar,
            margin=float(adaptive_margin),
        )

    for bars in bars_to_use:
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
