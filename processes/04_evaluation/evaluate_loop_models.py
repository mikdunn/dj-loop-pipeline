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
            try:
                bpm, beats = librosa.beat.beat_track(y=y, sr=file_sr, units="time")
                cands = generate_bar_aligned_candidates(audio_path.stem, beats, bpm, bars)
            except Exception:
                bpm, beats = 120.0, np.array([], dtype=float)
                cands = []

            # Fallback for very short/beat-sparse clips: fixed windows.
            if not cands:
                duration = len(y) / float(file_sr) if file_sr > 0 else 0.0
                win_sec = 2.0
                stride_sec = 1.0
                bpm_arr = np.asarray(bpm, dtype=float).ravel()
                bpm_value = float(bpm_arr[0]) if bpm_arr.size > 0 and np.isfinite(bpm_arr[0]) else 120.0
                starts = np.arange(0.0, max(0.0, duration - win_sec) + 1e-9, stride_sec, dtype=float)
                cands = [
                    {
                        "start_time": float(s),
                        "end_time": float(s + win_sec),
                        "bars": 0,
                        "bpm": bpm_value,
                    }
                    for s in starts
                ]

            for cand in cands:
                if hasattr(cand, "start_time"):
                    start_time = float(cand.start_time)
                    end_time = float(cand.end_time)
                    bars_val = int(cand.bars)
                    bpm_val = float(cand.bpm)
                else:
                    start_time = float(cand["start_time"])
                    end_time = float(cand["end_time"])
                    bars_val = int(cand["bars"])
                    bpm_val = float(cand["bpm"])

                feats = extract_full_features(y, file_sr, start_time, end_time)
                if not feats:
                    continue
                row = {
                    "track_id": audio_path.stem,
                    "start_time": start_time,
                    "end_time": end_time,
                    "bars": bars_val,
                    "bpm": bpm_val,
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


def _normalize_01(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)


def add_composite_predictions(
    df: pd.DataFrame,
    base_pred_col: str,
    out_col: str,
    weight_model_score: float,
    weight_periodicity: float,
    weight_drum_alignment: float,
) -> pd.DataFrame:
    for c in ("periodicity_score", "drum_alignment_score"):
        if c not in df.columns:
            df[c] = 0.0

    model_vals = _normalize_01(df[base_pred_col].to_numpy(dtype=float))
    periodicity_vals = np.clip(df["periodicity_score"].to_numpy(dtype=float), 0.0, 1.0)
    drum_vals = np.clip(df["drum_alignment_score"].to_numpy(dtype=float), 0.0, 1.0)

    w_model = max(0.0, float(weight_model_score))
    w_periodicity = max(0.0, float(weight_periodicity))
    w_drum = max(0.0, float(weight_drum_alignment))
    w_sum = w_model + w_periodicity + w_drum
    if w_sum <= 1e-12:
        w_model, w_periodicity, w_drum = 1.0, 0.0, 0.0
        w_sum = 1.0

    w_model /= w_sum
    w_periodicity /= w_sum
    w_drum /= w_sum

    df[out_col] = w_model * model_vals + w_periodicity * periodicity_vals + w_drum * drum_vals
    return df


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
    parser.add_argument("--disable_composite_eval", action="store_true", help="Skip composite reranking evaluation")
    parser.add_argument("--weight_model_score", type=float, default=0.75, help="Composite weight for normalized model score")
    parser.add_argument("--weight_periodicity", type=float, default=0.15, help="Composite weight for periodicity_score")
    parser.add_argument("--weight_drum_alignment", type=float, default=0.10, help="Composite weight for drum_alignment_score")
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

    if not bool(args.disable_composite_eval):
        df = add_composite_predictions(
            df,
            base_pred_col="pred_a",
            out_col="pred_a_composite",
            weight_model_score=float(args.weight_model_score),
            weight_periodicity=float(args.weight_periodicity),
            weight_drum_alignment=float(args.weight_drum_alignment),
        )
        metrics["model_a"]["metrics_composite"] = model_metrics(df, "pred_a_composite", top_k=args.top_k)
        metrics["model_a"]["composite_vs_raw"] = {
            "delta_topk_target": float(
                metrics["model_a"]["metrics_composite"]["mean_track_topk_target"]
                - metrics["model_a"]["metrics"]["mean_track_topk_target"]
            ),
            "delta_ndcg_at_k": float(
                metrics["model_a"]["metrics_composite"]["mean_track_ndcg_at_k"]
                - metrics["model_a"]["metrics"]["mean_track_ndcg_at_k"]
            ),
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

        if not bool(args.disable_composite_eval):
            df = add_composite_predictions(
                df,
                base_pred_col="pred_b",
                out_col="pred_b_composite",
                weight_model_score=float(args.weight_model_score),
                weight_periodicity=float(args.weight_periodicity),
                weight_drum_alignment=float(args.weight_drum_alignment),
            )
            metrics["model_b"]["metrics_composite"] = model_metrics(df, "pred_b_composite", top_k=args.top_k)
            metrics["model_b"]["composite_vs_raw"] = {
                "delta_topk_target": float(
                    metrics["model_b"]["metrics_composite"]["mean_track_topk_target"]
                    - metrics["model_b"]["metrics"]["mean_track_topk_target"]
                ),
                "delta_ndcg_at_k": float(
                    metrics["model_b"]["metrics_composite"]["mean_track_ndcg_at_k"]
                    - metrics["model_b"]["metrics"]["mean_track_ndcg_at_k"]
                ),
            }

        a_topk = metrics["model_a"]["metrics"]["mean_track_topk_target"]
        b_topk = metrics["model_b"]["metrics"]["mean_track_topk_target"]
        metrics["comparison"] = {
            "delta_topk_target_b_minus_a": float(b_topk - a_topk),
            "relative_topk_target_change_pct": float(((b_topk - a_topk) / (a_topk + 1e-8)) * 100.0),
        }

        if not bool(args.disable_composite_eval):
            a_topk_comp = metrics["model_a"]["metrics_composite"]["mean_track_topk_target"]
            b_topk_comp = metrics["model_b"]["metrics_composite"]["mean_track_topk_target"]
            metrics["comparison"]["delta_topk_target_b_minus_a_composite"] = float(b_topk_comp - a_topk_comp)
            metrics["comparison"]["relative_topk_target_change_pct_composite"] = float(
                ((b_topk_comp - a_topk_comp) / (a_topk_comp + 1e-8)) * 100.0
            )

    metrics["composite_eval"] = {
        "enabled": not bool(args.disable_composite_eval),
        "weights_requested": {
            "model_score": float(args.weight_model_score),
            "periodicity": float(args.weight_periodicity),
            "drum_alignment": float(args.weight_drum_alignment),
        },
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
