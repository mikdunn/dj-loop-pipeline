from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from pipelines.loop_note_slices import export_slices, slice_loop_on_note_hits
from pipelines.sample_pack_pipeline_ml import LoopPipelineML

# Choose your model backend (.json for XGBoost, .pt for PyTorch)
MODEL_PATH = "training/models/loop_ranker.pt"
FEATURES_PATH = "training/models/features.json"

RAW_AUDIO_DIR = Path("data/raw_audio")
OUTPUT_DIR = Path("data/exported_quiet_loops")

TOP_K_LOOPS = 5
BARS_OPTIONS = [2, 4, 8, 16]

# Cutoffs (pick ONE style, or combine):
MIN_SCORE = None  # absolute model score threshold
MIN_SCORE_PROB = 0.05  # per-track softmax probability threshold (scale-invariant)
MIN_REL_SCORE = None  # keep only scores >= best + MIN_REL_SCORE

# "quiet drums" thresholds (tune):
MAX_PERC_RATIO = 0.25  # perc_to_total_rms
MAX_BOUNDARY_SCORE = 0.30  # boundary_quiet_score

# Snap slice boundaries to note onsets within this tolerance
MAX_SNAP_S = 0.12


def _select_quiet_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if "perc_to_total_rms" not in df2.columns or "boundary_quiet_score" not in df2.columns:
        return df2

    df2 = df2[
        (df2["perc_to_total_rms"].fillna(0.0) <= MAX_PERC_RATIO)
        & (df2["boundary_quiet_score"].fillna(0.0) <= MAX_BOUNDARY_SCORE)
    ]
    return df2


def _slice_counts_for_bars(bars: int):
    # Always create 8 and 16 slices; if loop is 16 bars, also create 32 slices.
    counts = [8, 16]
    if int(bars) == 16:
        counts.append(32)
    return counts


def _export_loop_wav(outdir: Path, track_id: str, loop_index: int, y_loop: np.ndarray, sr: int):
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / track_id / f"loop_{loop_index:02d}.wav"
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), y_loop.astype(np.float32, copy=False), sr)
    return p


def main() -> int:
    pipeline = LoopPipelineML(MODEL_PATH, FEATURES_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for audio_file in RAW_AUDIO_DIR.glob("*.wav"):
        df_ranked = pipeline.process_track(
            audio_file,
            bars_options=BARS_OPTIONS,
            top_k=None,
            min_score=MIN_SCORE,
            min_score_prob=MIN_SCORE_PROB,
            min_rel_score=MIN_REL_SCORE,
        )
        if len(df_ranked) == 0:
            continue

        # Prefer quiet/drumless candidates; then keep the best by model score.
        df_quiet = _select_quiet_candidates(df_ranked)
        if len(df_quiet) == 0:
            continue

        df_quiet = df_quiet.sort_values("score", ascending=False).head(TOP_K_LOOPS)

        # Load audio once
        y, sr = librosa.load(str(audio_file), sr=44100, mono=True)

        meta_rows = []
        for loop_index, (_i, row) in enumerate(df_quiet.iterrows()):
            track_id = str(row.get("track_id", audio_file.stem))
            start_s = float(row["start_time"])
            end_s = float(row["end_time"])
            bars = int(row.get("bars", 0))

            a = int(start_s * sr)
            b = int(end_s * sr)
            b = min(b, len(y))
            a = min(a, b)
            y_loop = y[a:b]
            if y_loop.size == 0:
                continue

            loop_path = _export_loop_wav(OUTPUT_DIR, track_id, loop_index, y_loop, sr)

            slice_counts = _slice_counts_for_bars(bars)
            for n_slices in slice_counts:
                boundaries, onsets = slice_loop_on_note_hits(
                    y_loop=y_loop,
                    sr=sr,
                    n_slices=n_slices,
                    max_snap_s=MAX_SNAP_S,
                )
                slice_dir = OUTPUT_DIR / track_id / f"loop_{loop_index:02d}" / f"slices_{n_slices:02d}"
                export_slices(
                    y=y_loop,
                    sr=sr,
                    outdir=slice_dir,
                    prefix=f"{track_id}_loop{loop_index:02d}_{n_slices:02d}",
                    boundaries_s=boundaries,
                    normalize=True,
                )

                meta_rows.append(
                    {
                        "track_id": track_id,
                        "audio_file": str(audio_file),
                        "loop_index": loop_index,
                        "bars": bars,
                        "start_time": start_s,
                        "end_time": end_s,
                        "ml_score": float(row.get("score", np.nan)),
                        "perc_to_total_rms": float(row.get("perc_to_total_rms", np.nan)),
                        "boundary_quiet_score": float(row.get("boundary_quiet_score", np.nan)),
                        "loop_wav": str(loop_path),
                        "n_slices": int(n_slices),
                        "slice_dir": str(slice_dir),
                        "num_onsets": int(len(onsets)),
                    }
                )

        if meta_rows:
            pd.DataFrame(meta_rows).to_csv(OUTPUT_DIR / f"{audio_file.stem}_quiet_loops_and_slices.csv", index=False)

    print(f"Done. Output -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
