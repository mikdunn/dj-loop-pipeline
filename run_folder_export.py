import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
import librosa
from export_top_loops_pro import export_top_loops

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")


def generate_bar_aligned_candidates(y: np.ndarray, sr: int, beats: np.ndarray, bars_list=(1, 2, 4)):
    """Generate simple bar-aligned loop candidates based on beat timings.

    Assumes 4/4 time (4 beats per bar). Returns a list of dict rows with:
    track_id, start_time, end_time, bars, weight, score
    """
    if beats is None or len(beats) < 8:
        # Fallback: fixed windows of 2 seconds, stride 1 second
        candidates = []
        duration = len(y) / sr
        start = 0.0
        while start + 2.0 <= duration:
            end = start + 2.0
            seg = y[int(start*sr):int(end*sr)]
            rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
            candidates.append({
                "start_time": start,
                "end_time": end,
                "bars": 0,
                "weight": 1.0,
                "score": rms,
            })
            start += 1.0
        return candidates

    candidates = []
    beats_per_bar = 4
    for bars in bars_list:
        span_beats = bars * beats_per_bar
        for i in range(0, len(beats) - span_beats):
            start_t = float(beats[i])
            end_t = float(beats[i + span_beats])
            seg = y[int(start_t*sr):int(end_t*sr)]
            if len(seg) == 0:
                continue
            # Simple score: mean RMS
            rms = float(np.sqrt(np.mean(seg**2)))
            candidates.append({
                "start_time": start_t,
                "end_time": end_t,
                "bars": bars,
                "weight": 1.0,
                "score": rms,
            })
    return candidates


def process_file(audio_path: Path, top_k: int, export_mp3: bool):
    # Load audio (mono)
    y, sr = librosa.load(str(audio_path), sr=44100, mono=True)
    # Beat tracking
    try:
        bpm, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    except Exception:
        bpm, beats = None, None
    # Generate candidates
    rows = generate_bar_aligned_candidates(y, sr, beats)
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["track_id"] = audio_path.stem
    # Rank by simple score descending
    df = df.sort_values("score", ascending=False)

    outdir = Path("data/exported_loops") / audio_path.stem
    metas = export_top_loops(df, audio_path, outdir, top_k=top_k, sr=sr, export_mp3=export_mp3, add_tags=True)
    return metas


def main():
    parser = argparse.ArgumentParser(description="Run loop export (no ML) on a folder")
    parser.add_argument("--folder", required=True, help="Path to folder containing audio files")
    parser.add_argument("--top_k", type=int, default=5, help="Number of loops to export per track")
    parser.add_argument("--export_mp3", action="store_true", help="Export MP3 previews (requires ffmpeg)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    all_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                all_files.append(p)

    print(f"Found {len(all_files)} audio files in {folder}")
    total_exported = 0
    for p in all_files:
        try:
            metas = process_file(p, top_k=args.top_k, export_mp3=args.export_mp3)
            total_exported += len(metas) if metas else 0
            print(f"Exported {len(metas) if metas else 0} loop(s) for {p}")
        except Exception as e:
            print(f"Failed to process {p}: {e}")
    print(f"Done. Total loops exported: {total_exported}")


if __name__ == "__main__":
    main()
