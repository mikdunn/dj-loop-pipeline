import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import xgboost as xgb

from training.dataset_builder import generate_bar_aligned_candidates
from training.feature_extraction import extract_full_features

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")


def parse_bars(value: str) -> List[int]:
    bars = [int(x.strip()) for x in value.split(",") if x.strip()]
    bars = [b for b in bars if b > 0]
    if not bars:
        raise ValueError("bars must contain at least one positive integer")
    return sorted(set(bars))


def collect_audio_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for root, _, names in os.walk(folder):
        for name in names:
            p = Path(root) / name
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
    return sorted(files)


def heuristic_target(row: Dict) -> float:
    """Weak target in [0,1] representing drum-break suitability."""
    percussive = float(np.clip(row.get("percussive_ratio", 0.0), 0.0, 1.0))
    onset_density_norm = float(np.clip(row.get("onset_density", 0.0) / 12.0, 0.0, 1.0))
    onset_strength_norm = float(np.clip(row.get("onset_strength_mean", 0.0) / 5.0, 0.0, 1.0))
    rms_norm = float(np.clip(row.get("rms_mean", 0.0) / 0.2, 0.0, 1.0))

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
        0.40 * percussive
        + 0.25 * onset_density_norm
        + 0.15 * onset_strength_norm
        + 0.10 * rms_norm
        + 0.10 * dur_score
    )
    return float(np.clip(score, 0.0, 1.0))


def build_eval_rows(audio_files: List[Path], bars: Sequence[int] = (4, 8), sr: int = 44100) -> pd.DataFrame:
    rows: List[Dict] = []

    for i, audio_path in enumerate(audio_files, start=1):
        try:
            y, file_sr = librosa.load(str(audio_path), sr=sr, mono=True)
            bpm, beats = librosa.beat.beat_track(y=y, sr=file_sr, units="time")
            cands = generate_bar_aligned_candidates(audio_path.stem, beats, bpm, bars)

            for cand in cands:
                feats = extract_full_features(y, file_sr, cand.start_time, cand.end_time)
                if not feats:
                    continue
                row = {
                    "track_id": audio_path.stem,
                    "start_time": cand.start_time,
                    "end_time": cand.end_time,
                    "bars": cand.bars,
                    "bpm": cand.bpm,
                }
                row.update(feats)
                row["target"] = heuristic_target(row)
                rows.append(row)

            print(f"[{i}/{len(audio_files)}] processed {audio_path.name} -> {len(cands)} candidates")
        except Exception as exc:
            print(f"[{i}/{len(audio_files)}] failed {audio_path.name}: {exc}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _dcg(rels: np.ndarray) -> float:
    if rels.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2.0 ** rels - 1.0) / discounts))


def model_metrics(df: pd.DataFrame, pred_col: str, top_k: int, random_seed: int = 42) -> Dict[str, float]:
    track_topk_targets = []
    track_random_targets = []
    track_ndcg = []

    rng = np.random.default_rng(random_seed)

    for _, group in df.groupby("track_id"):
        g = group.copy()
        if g.empty:
            continue

        k = min(top_k, len(g))
        g_sorted = g.sort_values(pred_col, ascending=False)
        topk_target = float(g_sorted["target"].head(k).mean())
        track_topk_targets.append(topk_target)

        sample_ix = rng.choice(len(g), size=k, replace=False)
        random_target = float(g.iloc[sample_ix]["target"].mean())
        track_random_targets.append(random_target)

        rel_pred = g_sorted["target"].to_numpy(dtype=float)[:k]
        rel_best = g.sort_values("target", ascending=False)["target"].to_numpy(dtype=float)[:k]
        idcg = _dcg(rel_best)
        ndcg = (_dcg(rel_pred) / idcg) if idcg > 0 else 0.0
        track_ndcg.append(float(ndcg))

    pred = df[pred_col].to_numpy(dtype=float)
    target = df["target"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(pred - target)))
    pearson = float(pd.Series(pred).corr(pd.Series(target), method="pearson"))
    spearman = float(pd.Series(pred).corr(pd.Series(target), method="spearman"))

    mean_topk = float(np.mean(track_topk_targets)) if track_topk_targets else 0.0
    mean_random = float(np.mean(track_random_targets)) if track_random_targets else 0.0
    uplift = ((mean_topk - mean_random) / (mean_random + 1e-8)) * 100.0
    mean_ndcg = float(np.mean(track_ndcg)) if track_ndcg else 0.0

    return {
        "rows": int(len(df)),
        "tracks": int(df["track_id"].nunique()),
        "top_k": int(top_k),
        "mae": mae,
        "pearson": pearson if np.isfinite(pearson) else 0.0,
        "spearman": spearman if np.isfinite(spearman) else 0.0,
        "mean_track_topk_target": mean_topk,
        "mean_track_random_topk_target": mean_random,
        "uplift_vs_random_pct": float(uplift),
        "mean_track_ndcg_at_k": mean_ndcg,
    }


def add_model_predictions(
    df: pd.DataFrame,
    model_path: Path,
    feature_list_path: Path,
    pred_col: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    features = json.loads(feature_list_path.read_text(encoding="utf-8"))
    missing = [f for f in features if f not in df.columns]
    for m in missing:
        df[m] = 0.0

    X = df[features]
    df[pred_col] = model.predict(X)

    return df, {
        "feature_count": len(features),
        "missing_features_filled_zero": len(missing),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare loop-ranking models on a validation folder")
    parser.add_argument("--folder", required=True, help="Validation audio folder")
    parser.add_argument("--model_a", required=True, help="Path to first model (e.g. breaks-only)")
    parser.add_argument("--features_a", required=True, help="Feature JSON for first model")
    parser.add_argument("--model_b", default=None, help="Path to second model (e.g. mixed fine-tuned)")
    parser.add_argument("--features_b", default=None, help="Feature JSON for second model")
    parser.add_argument("--bars", default="4,8", help="Comma-separated bar lengths for candidate generation (default: 4,8)")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k segments per track for quality metrics")
    parser.add_argument("--rows_out", default="training/models/eval_rows.csv", help="CSV containing candidates, targets, and predictions")
    parser.add_argument("--metrics_out", default="training/models/eval_metrics.json", help="JSON output for aggregate metrics")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Validation folder not found: {folder}")

    model_a = Path(args.model_a)
    features_a = Path(args.features_a)
    if not model_a.exists() or not features_a.exists():
        raise FileNotFoundError("model_a/features_a not found")

    if bool(args.model_b) != bool(args.features_b):
        raise ValueError("Provide both --model_b and --features_b, or neither")

    bars = parse_bars(args.bars)

    audio_files = collect_audio_files(folder)
    if not audio_files:
        raise RuntimeError(f"No audio files found under: {folder}")

    print(f"Found {len(audio_files)} validation files")
    df = build_eval_rows(audio_files, bars=bars)
    if df.empty:
        raise RuntimeError("No evaluation rows generated")

    df, meta_a = add_model_predictions(df, model_a, features_a, "pred_a")
    metrics = {
        "model_a": {
            "name": str(model_a),
            "feature_info": meta_a,
            "metrics": model_metrics(df, "pred_a", top_k=args.top_k),
        }
    }

    if args.model_b and args.features_b:
        model_b = Path(args.model_b)
        features_b = Path(args.features_b)
        if not model_b.exists() or not features_b.exists():
            raise FileNotFoundError("model_b/features_b not found")

        df, meta_b = add_model_predictions(df, model_b, features_b, "pred_b")
        metrics["model_b"] = {
            "name": str(model_b),
            "feature_info": meta_b,
            "metrics": model_metrics(df, "pred_b", top_k=args.top_k),
        }

        a_topk = metrics["model_a"]["metrics"]["mean_track_topk_target"]
        b_topk = metrics["model_b"]["metrics"]["mean_track_topk_target"]
        metrics["comparison"] = {
            "delta_topk_target_b_minus_a": float(b_topk - a_topk),
            "relative_topk_target_change_pct": float(((b_topk - a_topk) / (a_topk + 1e-8)) * 100.0),
        }

    rows_out = Path(args.rows_out)
    rows_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(rows_out, index=False)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== Evaluation Summary ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved rows: {rows_out}")
    print(f"Saved metrics: {metrics_out}")


if __name__ == "__main__":
    main()
