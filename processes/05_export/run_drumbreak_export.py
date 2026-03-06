<<<<<<< HEAD
import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
import librosa
from export_top_loops_pro import export_top_loops

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")


def extract_drumbreak_candidates(y: np.ndarray, sr: int, beats: np.ndarray, bars_list=(1, 2, 4)):
    """Extract percussive-heavy loop candidates (drum breaks) using HPSS and beat timings.

    - Computes percussive component via HPSS
    - Builds bar-aligned segments (1/2/4 bars)
    - Scores by percussive RMS and percussive ratio vs full mix
    """
    # HPSS to separate percussive component
    try:
        y_h, y_p = librosa.effects.hpss(y)
    except Exception:
        y_h, y_p = np.zeros_like(y), y

    candidates = []
    beats_per_bar = 4

    if beats is None or len(beats) < 8:
        # Fallback: fixed windows of ~2 seconds (approx 2 bars at ~60 bpm)
        duration = len(y) / sr
        start = 0.0
        while start + 2.0 <= duration:
            end = start + 2.0
            seg = y[int(start*sr):int(end*sr)]
            seg_p = y_p[int(start*sr):int(end*sr)]
            if len(seg_p) == 0:
                start += 1.0
                continue
            rms_p = float(np.sqrt(np.mean(seg_p**2)))
            rms_full = float(np.sqrt(np.mean(seg**2))) if len(seg) else 1e-6
            perc_ratio = rms_p / (rms_full + 1e-6)
            score = (rms_p * 0.7) + (perc_ratio * 0.3)
            candidates.append({
                "start_time": start,
                "end_time": end,
                "bars": 0,
                "weight": perc_ratio,
                "score": score,
            })
            start += 1.0
        return candidates

    for bars in bars_list:
        span_beats = bars * beats_per_bar
        for i in range(0, len(beats) - span_beats):
            start_t = float(beats[i])
            end_t = float(beats[i + span_beats])
            seg = y[int(start_t*sr):int(end_t*sr)]
            seg_p = y_p[int(start_t*sr):int(end_t*sr)]
            if len(seg_p) == 0:
                continue
            rms_p = float(np.sqrt(np.mean(seg_p**2)))
            rms_full = float(np.sqrt(np.mean(seg**2))) if len(seg) else 1e-6
            perc_ratio = rms_p / (rms_full + 1e-6)
            score = (rms_p * 0.7) + (perc_ratio * 0.3)
            candidates.append({
                "start_time": start_t,
                "end_time": end_t,
                "bars": bars,
                "weight": perc_ratio,
                "score": score,
            })
    return candidates


def process_file(audio_path: Path, top_k: int, export_mp3: bool, out_root: Path):
    # Load audio (mono)
    y, sr = librosa.load(str(audio_path), sr=44100, mono=True)
    # Beat tracking
    try:
        bpm, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    except Exception:
        bpm, beats = None, None

    rows = extract_drumbreak_candidates(y, sr, beats)
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["track_id"] = audio_path.stem
    # Rank by score descending
    df = df.sort_values("score", ascending=False)

    outdir = out_root / audio_path.stem
    metas = export_top_loops(df, audio_path, outdir, top_k=top_k, sr=sr, export_mp3=export_mp3, add_tags=True)
    return metas


def main():
    parser = argparse.ArgumentParser(description="Run drumbreak export on a folder")
    parser.add_argument("--folder", required=True, help="Path to folder containing audio files")
    parser.add_argument("--top_k", type=int, default=5, help="Number of drumbreak loops to export per track")
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
    out_root = Path("data/exported_drumbreaks")
    out_root.mkdir(parents=True, exist_ok=True)

    for p in all_files:
        try:
            metas = process_file(p, top_k=args.top_k, export_mp3=args.export_mp3, out_root=out_root)
            total_exported += len(metas) if metas else 0
            print(f"Exported {len(metas) if metas else 0} drumbreak(s) for {p}")
        except Exception as e:
            print(f"Failed to process {p}: {e}")
    print(f"Done. Total drumbreak loops exported: {total_exported}")


if __name__ == "__main__":
    main()
=======
import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
import librosa
from export_top_loops_pro import export_top_loops

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")


def extract_drumbreak_candidates(y: np.ndarray, sr: int, beats: np.ndarray, bars_list=(1, 2, 4)):
    """Extract percussive-heavy loop candidates (drum breaks) using HPSS and beat timings.

    - Computes percussive component via HPSS
    - Builds bar-aligned segments (1/2/4 bars)
    - Scores by percussive RMS and percussive ratio vs full mix
    """
    # HPSS to separate percussive component
    try:
        y_h, y_p = librosa.effects.hpss(y)
    except Exception:
        y_h, y_p = np.zeros_like(y), y

    candidates = []
    beats_per_bar = 4

    if beats is None or len(beats) < 8:
        # Fallback: fixed windows of ~2 seconds (approx 2 bars at ~60 bpm)
        duration = len(y) / sr
        start = 0.0
        while start + 2.0 <= duration:
            end = start + 2.0
            seg = y[int(start*sr):int(end*sr)]
            seg_p = y_p[int(start*sr):int(end*sr)]
            if len(seg_p) == 0:
                start += 1.0
                continue
            rms_p = float(np.sqrt(np.mean(seg_p**2)))
            rms_full = float(np.sqrt(np.mean(seg**2))) if len(seg) else 1e-6
            perc_ratio = rms_p / (rms_full + 1e-6)
            score = (rms_p * 0.7) + (perc_ratio * 0.3)
            candidates.append({
                "start_time": start,
                "end_time": end,
                "bars": 0,
                "weight": perc_ratio,
                "score": score,
            })
            start += 1.0
        return candidates

    for bars in bars_list:
        span_beats = bars * beats_per_bar
        for i in range(0, len(beats) - span_beats):
            start_t = float(beats[i])
            end_t = float(beats[i + span_beats])
            seg = y[int(start_t*sr):int(end_t*sr)]
            seg_p = y_p[int(start_t*sr):int(end_t*sr)]
            if len(seg_p) == 0:
                continue
            rms_p = float(np.sqrt(np.mean(seg_p**2)))
            rms_full = float(np.sqrt(np.mean(seg**2))) if len(seg) else 1e-6
            perc_ratio = rms_p / (rms_full + 1e-6)
            score = (rms_p * 0.7) + (perc_ratio * 0.3)
            candidates.append({
                "start_time": start_t,
                "end_time": end_t,
                "bars": bars,
                "weight": perc_ratio,
                "score": score,
            })
    return candidates


def process_file(audio_path: Path, top_k: int, export_mp3: bool, out_root: Path):
    # Load audio (mono)
    y, sr = librosa.load(str(audio_path), sr=44100, mono=True)
    # Beat tracking
    try:
        bpm, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    except Exception:
        bpm, beats = None, None

    rows = extract_drumbreak_candidates(y, sr, beats)
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["track_id"] = audio_path.stem
    # Rank by score descending
    df = df.sort_values("score", ascending=False)

    outdir = out_root / audio_path.stem
    metas = export_top_loops(df, audio_path, outdir, top_k=top_k, sr=sr, export_mp3=export_mp3, add_tags=True)
    return metas


def main():
    parser = argparse.ArgumentParser(description="Run drumbreak export on a folder")
    parser.add_argument("--folder", required=True, help="Path to folder containing audio files")
    parser.add_argument("--top_k", type=int, default=5, help="Number of drumbreak loops to export per track")
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
    out_root = Path("data/exported_drumbreaks")
    out_root.mkdir(parents=True, exist_ok=True)

    for p in all_files:
        try:
            metas = process_file(p, top_k=args.top_k, export_mp3=args.export_mp3, out_root=out_root)
            total_exported += len(metas) if metas else 0
            print(f"Exported {len(metas) if metas else 0} drumbreak(s) for {p}")
        except Exception as e:
            print(f"Failed to process {p}: {e}")
    print(f"Done. Total drumbreak loops exported: {total_exported}")


if __name__ == "__main__":
    main()
>>>>>>> 9ccce6e0d927a6d32119a0ff7548b63d6b450e51
