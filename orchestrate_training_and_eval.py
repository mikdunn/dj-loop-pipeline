import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

import train_from_drumbreaks as trainer
import evaluate_loop_models as evaluator


def run_pipeline(
    drumbreak_folder: Path,
    validation_folder: Path,
    output_dir: Path,
    full_music_folder: Optional[Path] = None,
    neg_per_track: int = 24,
    bars: Sequence[int] = (4, 8),
    tempo_augment: bool = False,
    tempo_min_pct: int = -10,
    tempo_max_pct: int = 10,
    tempo_step_pct: int = 1,
    top_k: int = 5,
    seed: int = 42,
) -> Path:
    if not drumbreak_folder.exists():
        raise FileNotFoundError(f"Drum-break folder not found: {drumbreak_folder}")
    if not validation_folder.exists():
        raise FileNotFoundError(f"Validation folder not found: {validation_folder}")
    if full_music_folder and not full_music_folder.exists():
        raise FileNotFoundError(f"Full-music folder not found: {full_music_folder}")

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    rows_dir = output_dir / "rows"
    eval_dir = output_dir / "eval"
    models_dir.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1) Train breaks-only model
    # -----------------------------
    break_files = trainer._collect_audio_files(drumbreak_folder)
    if not break_files:
        raise RuntimeError(f"No drum-break audio files found under: {drumbreak_folder}")

    print(f"[1/4] Building positive rows from {len(break_files)} drum-break files")
    df_pos = trainer.build_training_rows(
        break_files,
        bars=bars,
        tempo_augment=tempo_augment,
        tempo_min_pct=tempo_min_pct,
        tempo_max_pct=tempo_max_pct,
        tempo_step_pct=tempo_step_pct,
    )
    if df_pos.empty:
        raise RuntimeError("Could not build positive training rows")

    pos_rows_path = rows_dir / "training_rows_breaks_only.csv"
    df_pos.to_csv(pos_rows_path, index=False)

    model_breaks = models_dir / "loop_ranker_breaks_only.json"
    features_breaks = models_dir / "features_breaks_only.json"

    print("[2/4] Training breaks-only model")
    trainer.train_model(
        df=df_pos,
        model_path=model_breaks,
        feature_list_path=features_breaks,
        random_state=seed,
        use_sample_weights=True,
    )

    # -----------------------------
    # 2) Train mixed model (optional)
    # -----------------------------
    model_mixed = None
    features_mixed = None
    df_mixed = None

    if full_music_folder is not None:
        full_files = trainer._collect_audio_files(full_music_folder)
        print(f"[3/4] Mining hard negatives from {len(full_files)} full-song files")
        if full_files:
            df_neg = trainer.build_hard_negative_rows(
                audio_files=full_files,
                bars=bars,
                neg_per_track=neg_per_track,
            )
            if not df_neg.empty:
                df_mixed = pd.concat([df_pos, df_neg], ignore_index=True)
            else:
                print("No hard negatives were generated; mixed model will train on positives only")
                df_mixed = df_pos.copy()
        else:
            print("No full-song files found; mixed model will train on positives only")
            df_mixed = df_pos.copy()

        mixed_rows_path = rows_dir / "training_rows_mixed.csv"
        df_mixed.to_csv(mixed_rows_path, index=False)

        model_mixed = models_dir / "loop_ranker_mixed.json"
        features_mixed = models_dir / "features_mixed.json"

        print("[3/4] Training mixed model")
        trainer.train_model(
            df=df_mixed,
            model_path=model_mixed,
            feature_list_path=features_mixed,
            random_state=seed,
            use_sample_weights=True,
        )
    else:
        print("[3/4] Skipping mixed-model training (no --full_music_folder provided)")

    # -----------------------------
    # 3) Evaluate and compare
    # -----------------------------
    val_files = evaluator.collect_audio_files(validation_folder)
    if not val_files:
        raise RuntimeError(f"No validation audio files found under: {validation_folder}")

    print(f"[4/4] Building evaluation rows from {len(val_files)} validation files")
    eval_df = evaluator.build_eval_rows(val_files, bars=bars)
    if eval_df.empty:
        raise RuntimeError("No evaluation rows generated")

    eval_df, meta_a = evaluator.add_model_predictions(
        eval_df,
        model_path=model_breaks,
        feature_list_path=features_breaks,
        pred_col="pred_breaks_only",
    )

    metrics = {
        "run_info": {
            "drumbreak_folder": str(drumbreak_folder),
            "validation_folder": str(validation_folder),
            "full_music_folder": str(full_music_folder) if full_music_folder else None,
            "neg_per_track": int(neg_per_track),
            "bars": [int(b) for b in bars],
            "tempo_augment": bool(tempo_augment),
            "tempo_min_pct": int(tempo_min_pct),
            "tempo_max_pct": int(tempo_max_pct),
            "tempo_step_pct": int(tempo_step_pct),
            "top_k": int(top_k),
            "seed": int(seed),
        },
        "model_breaks_only": {
            "model_path": str(model_breaks),
            "features_path": str(features_breaks),
            "feature_info": meta_a,
            "metrics": evaluator.model_metrics(eval_df, pred_col="pred_breaks_only", top_k=top_k),
        },
    }

    if model_mixed and features_mixed:
        eval_df, meta_b = evaluator.add_model_predictions(
            eval_df,
            model_path=model_mixed,
            feature_list_path=features_mixed,
            pred_col="pred_mixed",
        )

        metrics["model_mixed"] = {
            "model_path": str(model_mixed),
            "features_path": str(features_mixed),
            "feature_info": meta_b,
            "metrics": evaluator.model_metrics(eval_df, pred_col="pred_mixed", top_k=top_k),
        }

        a_topk = metrics["model_breaks_only"]["metrics"]["mean_track_topk_target"]
        b_topk = metrics["model_mixed"]["metrics"]["mean_track_topk_target"]
        metrics["comparison"] = {
            "delta_topk_target_mixed_minus_breaks": float(b_topk - a_topk),
            "relative_topk_target_change_pct": float(((b_topk - a_topk) / (a_topk + 1e-8)) * 100.0),
        }

    eval_rows_path = eval_dir / "eval_rows.csv"
    eval_df.to_csv(eval_rows_path, index=False)

    metrics_path = eval_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "paths": {
                    "models_dir": str(models_dir),
                    "rows_dir": str(rows_dir),
                    "eval_rows": str(eval_rows_path),
                    "eval_metrics": str(metrics_path),
                },
                "metrics_preview": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nRun complete!")
    print(f"Output dir: {output_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Rows: {eval_rows_path}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate training and evaluation: breaks-only, mixed fine-tune, and side-by-side comparison"
    )
    parser.add_argument("--drumbreak_folder", required=True, help="Folder containing drum-break files")
    parser.add_argument("--validation_folder", required=True, help="Folder used for evaluation")
    parser.add_argument(
        "--full_music_folder",
        default=None,
        help="Optional full-song folder used to mine hard negatives for mixed training",
    )
    parser.add_argument(
        "--bars",
        default="4,8",
        help="Comma-separated bar lengths used in training/evaluation candidates (default: 4,8)",
    )
    parser.add_argument(
        "--tempo_augment",
        action="store_true",
        help="Apply CDJ-style tempo augmentation for drum-break positives",
    )
    parser.add_argument("--tempo_min_pct", type=int, default=-10, help="Minimum tempo shift percent for augmentation")
    parser.add_argument("--tempo_max_pct", type=int, default=10, help="Maximum tempo shift percent for augmentation")
    parser.add_argument("--tempo_step_pct", type=int, default=1, help="Tempo shift step percent for augmentation")
    parser.add_argument("--neg_per_track", type=int, default=24, help="Hard negatives to mine per full-song track")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k loops per track for quality metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output run directory. Default: training/runs/<timestamp>",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("training/runs") / timestamp
    bars = trainer._parse_bars(args.bars)

    run_pipeline(
        drumbreak_folder=Path(args.drumbreak_folder),
        validation_folder=Path(args.validation_folder),
        output_dir=output_dir,
        full_music_folder=Path(args.full_music_folder) if args.full_music_folder else None,
        neg_per_track=args.neg_per_track,
        bars=bars,
        tempo_augment=args.tempo_augment,
        tempo_min_pct=args.tempo_min_pct,
        tempo_max_pct=args.tempo_max_pct,
        tempo_step_pct=args.tempo_step_pct,
        top_k=args.top_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
