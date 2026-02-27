import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import librosa
import numpy as np
import pandas as pd
import xgboost as xgb

from training.dataset_builder import generate_bar_aligned_candidates
from training.feature_extraction import extract_full_features

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")


def _parse_bars(value: str) -> List[int]:
    bars = [int(x.strip()) for x in value.split(",") if x.strip()]
    bars = [b for b in bars if b > 0]
    if not bars:
        raise ValueError("bars must contain at least one positive integer")
    return sorted(set(bars))


def _tempo_pct_values(min_pct: int, max_pct: int, step_pct: int, include_original: bool = True) -> List[int]:
    if step_pct <= 0:
        raise ValueError("tempo_step_pct must be > 0")
    if min_pct > max_pct:
        raise ValueError("tempo_min_pct must be <= tempo_max_pct")

    pcts = list(range(min_pct, max_pct + 1, step_pct))
    if include_original and 0 not in pcts:
        pcts.append(0)
    return sorted(set(pcts))


def _collect_audio_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for root, _, names in os.walk(folder):
        for name in names:
            p = Path(root) / name
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
    return sorted(files)


def _bpm_from_filename(path: Path) -> Optional[float]:
    """Extract BPM from trailing `_INTEGER` pattern in filename stem.

    Examples:
      - `break_87.wav` -> 87
      - `loop_130` -> 130
    """
    stem = path.stem
    m = re.search(r"_(\d{2,3})$", stem)
    if not m:
        return None
    bpm = float(m.group(1))
    if 40 <= bpm <= 240:
        return bpm
    return None


def _make_beat_grid_from_bpm(y: np.ndarray, sr: int, bpm: float) -> np.ndarray:
    """Build beat timestamps from known BPM and a likely first-onset anchor."""
    duration = len(y) / float(sr)
    if duration <= 0:
        return np.array([], dtype=float)

    beat_period = 60.0 / max(1e-6, bpm)

    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        start = float(onset_times[0]) if len(onset_times) else 0.0
    except Exception:
        start = 0.0

    start = max(0.0, min(start, duration))
    beats = np.arange(start, duration + 1e-9, beat_period, dtype=float)
    return beats


def _heuristic_score(row: Dict) -> float:
    # Drum-break oriented weak label in [0, 1]
    percussive = float(np.clip(row.get("percussive_ratio", 0.0), 0.0, 1.0))
    onset_density_norm = float(np.clip(row.get("onset_density", 0.0) / 12.0, 0.0, 1.0))
    onset_strength_norm = float(np.clip(row.get("onset_strength_mean", 0.0) / 5.0, 0.0, 1.0))
    rms_norm = float(np.clip(row.get("rms_mean", 0.0) / 0.2, 0.0, 1.0))
    boundary_drum = float(np.clip(row.get("boundary_drum_score", 0.0), 0.0, 1.0))
    kick_on_one = float(np.clip(row.get("kick_on_one_score", 0.0), 0.0, 1.0))
    snare_2_or_3 = float(np.clip(row.get("snare_two_or_three_score", 0.0), 0.0, 1.0))
    hihat_grid = float(np.clip(row.get("hihat_grid_score", 0.0), 0.0, 1.0))

    # Prefer loop-ish durations around 2 to 8s
    dur = float(row.get("duration_sec", 0.0))
    if dur < 1.0:
        dur_score = 0.0
    elif dur <= 8.0:
        dur_score = 1.0
    elif dur <= 12.0:
        dur_score = max(0.0, 1.0 - (dur - 8.0) / 4.0)
    else:
        dur_score = 0.0

    score = (
        0.25 * percussive
        + 0.15 * onset_density_norm
        + 0.10 * onset_strength_norm
        + 0.08 * rms_norm
        + 0.07 * dur_score
        + 0.20 * boundary_drum
        + 0.08 * kick_on_one
        + 0.05 * snare_2_or_3
        + 0.02 * hihat_grid
    )
    return float(np.clip(score, 0.0, 1.0))


def _normalize_per_track(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    def _scale(g: pd.DataFrame) -> pd.DataFrame:
        v = g[score_col].to_numpy(dtype=float)
        lo, hi = np.min(v), np.max(v)
        if hi - lo < 1e-8:
            g[out_col] = 0.5
        else:
            g[out_col] = (v - lo) / (hi - lo)
        return g

    return df.groupby("track_id", group_keys=False).apply(_scale)


def _collect_candidate_rows(
    audio_files: List[Path],
    bars: Sequence[int] = (4, 8),
    sr: int = 44100,
    source_kind: str = "drumbreak",
    tempo_augment: bool = False,
    tempo_min_pct: int = -10,
    tempo_max_pct: int = 10,
    tempo_step_pct: int = 1,
) -> pd.DataFrame:
    rows = []
    tempo_pcts = _tempo_pct_values(
        tempo_min_pct,
        tempo_max_pct,
        tempo_step_pct,
        include_original=True,
    )

    for i, audio_path in enumerate(audio_files, start=1):
        try:
            y, file_sr = librosa.load(str(audio_path), sr=sr, mono=True)
            filename_bpm = _bpm_from_filename(audio_path)

            variant_count = 0
            total_candidates = 0
            for pct in tempo_pcts:
                if pct != 0 and not tempo_augment:
                    continue

                tempo_factor = 1.0 + (pct / 100.0)
                if tempo_factor <= 0:
                    continue

                if pct == 0:
                    y_variant = y
                else:
                    # CDJ-like tempo fader simulation: ±tempo changes in percent.
                    y_variant = librosa.effects.time_stretch(y, rate=tempo_factor)

                if filename_bpm is not None:
                    bpm = float(filename_bpm) * tempo_factor
                    beats = _make_beat_grid_from_bpm(y_variant, file_sr, bpm)
                else:
                    bpm, beats = librosa.beat.beat_track(y=y_variant, sr=file_sr, units="time")

                track_id = audio_path.stem if pct == 0 else f"{audio_path.stem}_tempo_{pct:+d}pct"
                cands = generate_bar_aligned_candidates(track_id, beats, bpm, bars)

                for cand in cands:
                    feats = extract_full_features(y_variant, file_sr, cand.start_time, cand.end_time)
                    if not feats:
                        continue
                    row = {
                        "track_id": track_id,
                        "start_time": cand.start_time,
                        "end_time": cand.end_time,
                        "bars": cand.bars,
                        "bpm": cand.bpm,
                        "source_kind": source_kind,
                        "tempo_shift_pct": pct,
                        "tempo_factor": tempo_factor,
                        "bpm_source": "filename" if filename_bpm is not None else "beat_track",
                    }
                    row.update(feats)
                    row["weak_score"] = _heuristic_score(row)
                    rows.append(row)

                variant_count += 1
                total_candidates += len(cands)

            print(
                f"[{i}/{len(audio_files)}] processed {audio_path.name} -> {total_candidates} candidates "
                f"across {variant_count} tempo variant(s) ({source_kind})"
            )
        except Exception as exc:
            print(f"[{i}/{len(audio_files)}] failed {audio_path.name}: {exc}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_training_rows(
    audio_files: List[Path],
    bars: Sequence[int] = (4, 8),
    sr: int = 44100,
    tempo_augment: bool = False,
    tempo_min_pct: int = -10,
    tempo_max_pct: int = 10,
    tempo_step_pct: int = 1,
) -> pd.DataFrame:
    df = _collect_candidate_rows(
        audio_files,
        bars=bars,
        sr=sr,
        source_kind="drumbreak",
        tempo_augment=tempo_augment,
        tempo_min_pct=tempo_min_pct,
        tempo_max_pct=tempo_max_pct,
        tempo_step_pct=tempo_step_pct,
    )
    if df.empty:
        return pd.DataFrame()

    df = _normalize_per_track(df, "weak_score", "target")
    # Keep positive examples in upper range so negatives can be mixed in later.
    df["target"] = 0.40 + 0.60 * df["target"]
    df["sample_weight"] = 1.0
    return df


def build_hard_negative_rows(
    audio_files: List[Path],
    bars: Sequence[int] = (4, 8),
    sr: int = 44100,
    neg_per_track: int = 24,
) -> pd.DataFrame:
    df = _collect_candidate_rows(
        audio_files,
        bars=bars,
        sr=sr,
        source_kind="fullmix",
        tempo_augment=False,
    )
    if df.empty:
        return pd.DataFrame()

    hard_negs = []
    for _, group in df.groupby("track_id"):
        g = group.sort_values("weak_score", ascending=True)
        # Mine least drum-break-like segments as hard negatives.
        hard_negs.append(g.head(max(1, neg_per_track)))

    neg = pd.concat(hard_negs, ignore_index=True)
    inv = 1.0 - np.clip(neg["weak_score"].to_numpy(dtype=float), 0.0, 1.0)
    neg["target"] = 0.20 * inv
    neg["sample_weight"] = 1.25
    return neg


def train_model(
    df: pd.DataFrame,
    model_path: Path,
    feature_list_path: Path,
    random_state: int = 42,
    use_sample_weights: bool = True,
) -> None:
    non_feature_cols = {
        "track_id",
        "start_time",
        "end_time",
        "bars",
        "bpm",
        "bpm_source",
        "weak_score",
        "target",
        "source_kind",
        "sample_weight",
    }
    features = [c for c in df.columns if c not in non_feature_cols]

    X = df[features]
    y = df["target"]
    sample_weight = df["sample_weight"] if (use_sample_weights and "sample_weight" in df.columns) else None

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
    )
    model.fit(X, y, sample_weight=sample_weight)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    feature_list_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_path))
    feature_list_path.write_text(json.dumps(features, indent=2), encoding="utf-8")

    preds = model.predict(X)
    mae = float(np.mean(np.abs(preds - y.to_numpy(dtype=float))))
    print(f"Training complete. Rows: {len(df)}, Features: {len(features)}, Train MAE: {mae:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved features: {feature_list_path}")


def main():
    parser = argparse.ArgumentParser(description="Train loop-ranker model from a folder of drum breaks")
    parser.add_argument("--folder", required=True, help="Path to folder containing drum-break audio files")
    parser.add_argument(
        "--bars",
        default="4,8",
        help="Comma-separated bar lengths for candidate slicing (default: 4,8)",
    )
    parser.add_argument(
        "--tempo_augment",
        action="store_true",
        help="Apply CDJ-style tempo augmentation to drum breaks",
    )
    parser.add_argument("--tempo_min_pct", type=int, default=-10, help="Minimum tempo shift percent for augmentation")
    parser.add_argument("--tempo_max_pct", type=int, default=10, help="Maximum tempo shift percent for augmentation")
    parser.add_argument("--tempo_step_pct", type=int, default=1, help="Tempo shift step percent for augmentation")
    parser.add_argument(
        "--full_music_folder",
        default=None,
        help="Optional path to full-song audio used to mine hard negatives for fine-tuning",
    )
    parser.add_argument("--neg_per_track", type=int, default=24, help="Hard negatives to mine per full-song track")
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on number of source files (useful for experiments, e.g. 50)",
    )
    parser.add_argument(
        "--filename_bpm_only",
        action="store_true",
        help="Use only files ending with _INTEGER BPM (e.g., name_128.wav)",
    )
    parser.add_argument("--model_out", default="training/models/loop_ranker.json", help="Output path for trained XGBoost model")
    parser.add_argument("--features_out", default="training/models/features.json", help="Output path for feature list JSON")
    parser.add_argument("--rows_out", default="training/models/training_rows.csv", help="Optional CSV dump of generated training rows")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    audio_files = _collect_audio_files(folder)
    if not audio_files:
        raise RuntimeError(f"No audio files found under: {folder}")

    if args.filename_bpm_only:
        audio_files = [p for p in audio_files if _bpm_from_filename(p) is not None]
        if not audio_files:
            raise RuntimeError("No files matched trailing _INTEGER BPM pattern")

    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("max_files must be > 0")
        audio_files = audio_files[: args.max_files]

    bars = _parse_bars(args.bars)

    print(f"Found {len(audio_files)} source files")
    df_pos = build_training_rows(
        audio_files,
        bars=bars,
        tempo_augment=args.tempo_augment,
        tempo_min_pct=args.tempo_min_pct,
        tempo_max_pct=args.tempo_max_pct,
        tempo_step_pct=args.tempo_step_pct,
    )
    if df_pos.empty:
        raise RuntimeError("Could not build any positive training rows from input files")

    df = df_pos

    full_music_folder: Optional[Path] = Path(args.full_music_folder) if args.full_music_folder else None
    if full_music_folder:
        if not full_music_folder.exists():
            raise FileNotFoundError(f"Full-music folder not found: {full_music_folder}")
        full_music_files = _collect_audio_files(full_music_folder)
        print(f"Found {len(full_music_files)} full-song files for hard-negative mining")
        if full_music_files:
            df_neg = build_hard_negative_rows(full_music_files, bars=bars, neg_per_track=args.neg_per_track)
            if not df_neg.empty:
                df = pd.concat([df_pos, df_neg], ignore_index=True)
                print(
                    f"Mixed dataset -> positives: {len(df_pos)}, negatives: {len(df_neg)}, total: {len(df)}"
                )
            else:
                print("No hard negatives were generated; training on positives only")
        else:
            print("No full-song audio found; training on positives only")

    if df.empty:
        raise RuntimeError("Could not build any training rows from input files")

    rows_out = Path(args.rows_out)
    rows_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(rows_out, index=False)
    print(f"Saved training rows: {rows_out}")

    train_model(df, Path(args.model_out), Path(args.features_out))


if __name__ == "__main__":
    main()
