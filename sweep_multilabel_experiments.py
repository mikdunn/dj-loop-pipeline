import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from train_multilabel_loop_classifier import (
    build_dataset,
    fit_and_evaluate_multilabel,
)


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_optional_int_list(s: str) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for part in (s or "").split(","):
        token = part.strip().lower()
        if not token or token == "none" or token == "0":
            out.append(None)
        else:
            out.append(int(token))
    return out


def parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def run_sweep(
    df: pd.DataFrame,
    out_leaderboard_json: Path,
    out_leaderboard_csv: Path,
    calibration_min_threshold: float,
    calibration_max_threshold: float,
    calibration_step: float,
    low_support_cutoff: int,
    n_estimators_grid: List[int],
    max_depth_grid: List[Optional[int]],
    min_samples_leaf_grid: List[int],
    low_support_floor_grid: List[float],
    val_size: float,
    test_size: float,
    random_state: int,
    label_source: str,
) -> None:
    experiments: List[Dict] = []
    exp_id = 0

    for n_estimators in n_estimators_grid:
        for max_depth in max_depth_grid:
            for min_leaf in min_samples_leaf_grid:
                for floor in low_support_floor_grid:
                    exp_id += 1
                    bundle, report = fit_and_evaluate_multilabel(
                        df=df,
                        calibration_min_threshold=calibration_min_threshold,
                        calibration_max_threshold=calibration_max_threshold,
                        calibration_step=calibration_step,
                        low_support_cutoff=low_support_cutoff,
                        low_support_floor=float(floor),
                        val_size=val_size,
                        test_size=test_size,
                        random_state=random_state,
                        n_estimators=int(n_estimators),
                        max_depth=max_depth,
                        min_samples_leaf=int(min_leaf),
                        label_source=label_source,
                    )

                    _ = bundle  # bundle not persisted in sweep; report is the tracked artifact

                    row = {
                        "experiment_id": exp_id,
                        "n_estimators": int(n_estimators),
                        "max_depth": None if max_depth is None else int(max_depth),
                        "min_samples_leaf": int(min_leaf),
                        "low_support_floor": float(floor),
                        "val_micro_f1_calibrated": float(report["val_metrics"]["micro_f1_calibrated"]),
                        "val_macro_f1_calibrated": float(report["val_metrics"]["macro_f1_calibrated"]),
                        "test_micro_f1_calibrated": float(report["test_metrics"]["micro_f1_calibrated"]),
                        "test_macro_f1_calibrated": float(report["test_metrics"]["macro_f1_calibrated"]),
                        "test_micro_f1_default": float(report["test_metrics"]["micro_f1_default"]),
                        "test_macro_f1_default": float(report["test_metrics"]["macro_f1_default"]),
                    }
                    experiments.append(row)

                    print(
                        f"[exp {exp_id}] n_estimators={n_estimators}, max_depth={max_depth}, "
                        f"min_samples_leaf={min_leaf}, low_support_floor={floor} -> "
                        f"test_micro_cal={row['test_micro_f1_calibrated']:.4f}, "
                        f"test_macro_cal={row['test_macro_f1_calibrated']:.4f}"
                    )

    if not experiments:
        raise RuntimeError("No experiments were run")

    df_leaderboard = pd.DataFrame(experiments)
    df_leaderboard = df_leaderboard.sort_values(
        by=["test_macro_f1_calibrated", "test_micro_f1_calibrated"],
        ascending=[False, False],
    ).reset_index(drop=True)

    out_leaderboard_json.parent.mkdir(parents=True, exist_ok=True)
    out_leaderboard_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_experiments": int(len(df_leaderboard)),
        "ranking": "test_macro_f1_calibrated desc, then test_micro_f1_calibrated desc",
        "best": df_leaderboard.iloc[0].to_dict(),
        "experiments": df_leaderboard.to_dict(orient="records"),
    }
    out_leaderboard_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    df_leaderboard.to_csv(out_leaderboard_csv, index=False)

    print(f"Saved leaderboard JSON: {out_leaderboard_json}")
    print(f"Saved leaderboard CSV: {out_leaderboard_csv}")
    print("Best config:")
    print(df_leaderboard.iloc[0].to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep multi-label loop model configs and output leaderboard")
    parser.add_argument("--folder", required=True, help="Root folder containing loop files")
    parser.add_argument("--min_tag_count", type=int, default=8, help="Minimum occurrences required to keep a tag")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on scanned files")
    parser.add_argument("--dataset_out", default="training/models/loop_multilabel_dataset.csv", help="Dataset CSV output")

    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--calibration_min_threshold", type=float, default=0.12)
    parser.add_argument("--calibration_max_threshold", type=float, default=0.8)
    parser.add_argument("--calibration_step", type=float, default=0.02)
    parser.add_argument("--low_support_cutoff", type=int, default=6)

    parser.add_argument("--n_estimators_grid", default="400,600,900")
    parser.add_argument("--max_depth_grid", default="none,18")
    parser.add_argument("--min_samples_leaf_grid", default="1,2,4")
    parser.add_argument("--low_support_floor_grid", default="0.20,0.24,0.28")

    parser.add_argument("--leaderboard_json", default="training/models/loop_multilabel_leaderboard.json")
    parser.add_argument("--leaderboard_csv", default="training/models/loop_multilabel_leaderboard.csv")
    parser.add_argument(
        "--label_source",
        choices=("filename", "audio", "hybrid"),
        default="audio",
        help="Weak-label source used when building the dataset.",
    )

    args = parser.parse_args()

    root = Path(args.folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    df = build_dataset(
        root,
        min_tag_count=int(args.min_tag_count),
        max_files=args.max_files,
        label_source=args.label_source,
    )
    if df.empty:
        raise RuntimeError("No labeled loop rows found for sweep")

    ds_out = Path(args.dataset_out)
    ds_out.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out["tags"] = df_out["tags"].apply(lambda x: ",".join(x))
    df_out.to_csv(ds_out, index=False)
    print(f"Saved dataset: {ds_out} (rows={len(df_out)})")

    val_size = max(0.05, min(0.45, float(args.val_size)))
    test_size = max(0.05, min(0.45, float(args.test_size)))
    if val_size + test_size >= 0.8:
        val_size = 0.2
        test_size = 0.2

    cal_min = max(0.0, min(1.0, float(args.calibration_min_threshold)))
    cal_max = max(0.0, min(1.0, float(args.calibration_max_threshold)))
    if cal_max < cal_min:
        cal_max = cal_min
    cal_step = max(0.005, min(0.2, float(args.calibration_step)))

    n_estimators_grid = [max(100, v) for v in parse_int_list(args.n_estimators_grid)]
    max_depth_grid = parse_optional_int_list(args.max_depth_grid)
    min_samples_leaf_grid = [max(1, v) for v in parse_int_list(args.min_samples_leaf_grid)]
    low_support_floor_grid = [max(cal_min, min(cal_max, v)) for v in parse_float_list(args.low_support_floor_grid)]

    if not n_estimators_grid or not min_samples_leaf_grid or not low_support_floor_grid:
        raise ValueError("One or more grids are empty; check grid CLI arguments")

    run_sweep(
        df=df,
        out_leaderboard_json=Path(args.leaderboard_json),
        out_leaderboard_csv=Path(args.leaderboard_csv),
        calibration_min_threshold=cal_min,
        calibration_max_threshold=cal_max,
        calibration_step=cal_step,
        low_support_cutoff=max(1, int(args.low_support_cutoff)),
        n_estimators_grid=n_estimators_grid,
        max_depth_grid=max_depth_grid,
        min_samples_leaf_grid=min_samples_leaf_grid,
        low_support_floor_grid=low_support_floor_grid,
        val_size=val_size,
        test_size=test_size,
        random_state=int(args.random_state),
        label_source=args.label_source,
    )


if __name__ == "__main__":
    main()
