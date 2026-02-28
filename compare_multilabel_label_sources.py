import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from train_multilabel_loop_classifier import build_dataset, fit_and_evaluate_multilabel


LABEL_SOURCES = ("filename", "audio", "hybrid")


def run_comparison(
    folder: Path,
    min_tag_count: int,
    max_files: Optional[int],
    out_dir: Path,
    calibration_min_threshold: float,
    calibration_max_threshold: float,
    calibration_step: float,
    low_support_cutoff: int,
    low_support_floor: float,
    val_size: float,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_leaf: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []
    payload: Dict[str, Dict] = {}

    for label_source in LABEL_SOURCES:
        print(f"\n=== Running label_source={label_source} ===")
        df = build_dataset(
            folder,
            min_tag_count=min_tag_count,
            max_files=max_files,
            label_source=label_source,
        )
        if df.empty:
            print(f"No rows for label_source={label_source}; skipping")
            payload[label_source] = {
                "status": "empty_dataset",
                "n_rows": 0,
            }
            continue

        ds_out = out_dir / f"loop_multilabel_dataset_{label_source}.csv"
        df_out = df.copy()
        df_out["tags"] = df_out["tags"].apply(lambda x: ",".join(x))
        df_out.to_csv(ds_out, index=False)

        _, report = fit_and_evaluate_multilabel(
            df=df,
            calibration_min_threshold=calibration_min_threshold,
            calibration_max_threshold=calibration_max_threshold,
            calibration_step=calibration_step,
            low_support_cutoff=low_support_cutoff,
            low_support_floor=low_support_floor,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            label_source=label_source,
        )

        report_out = out_dir / f"loop_multilabel_report_{label_source}.json"
        report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

        summary_row = {
            "label_source": label_source,
            "n_rows": int(report["n_rows"]),
            "n_classes": int(report["n_classes"]),
            "val_micro_f1_calibrated": float(report["val_metrics"]["micro_f1_calibrated"]),
            "val_macro_f1_calibrated": float(report["val_metrics"]["macro_f1_calibrated"]),
            "test_micro_f1_calibrated": float(report["test_metrics"]["micro_f1_calibrated"]),
            "test_macro_f1_calibrated": float(report["test_metrics"]["macro_f1_calibrated"]),
            "test_micro_f1_default": float(report["test_metrics"]["micro_f1_default"]),
            "test_macro_f1_default": float(report["test_metrics"]["macro_f1_default"]),
        }
        summary_rows.append(summary_row)

        payload[label_source] = {
            "status": "ok",
            "dataset_csv": str(ds_out),
            "report_json": str(report_out),
            "summary": summary_row,
            "classes": report.get("classes", []),
        }

    if summary_rows:
        leaderboard = pd.DataFrame(summary_rows).sort_values(
            by=["test_macro_f1_calibrated", "test_micro_f1_calibrated"],
            ascending=[False, False],
        ).reset_index(drop=True)
        leaderboard_csv = out_dir / "label_source_comparison.csv"
        leaderboard.to_csv(leaderboard_csv, index=False)

        payload["leaderboard"] = {
            "ranking": "test_macro_f1_calibrated desc, then test_micro_f1_calibrated desc",
            "csv": str(leaderboard_csv),
            "best": leaderboard.iloc[0].to_dict(),
            "rows": leaderboard.to_dict(orient="records"),
        }

        print("\nComparison complete. Best label source:")
        print(payload["leaderboard"]["best"])
    else:
        payload["leaderboard"] = {
            "ranking": "none",
            "csv": None,
            "best": None,
            "rows": [],
        }

    summary_json = out_dir / "label_source_comparison.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved summary JSON: {summary_json}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multilabel weak-label sources: filename vs audio vs hybrid")
    parser.add_argument("--folder", required=True, help="Root folder containing loop files")
    parser.add_argument("--out_dir", default="training/models", help="Output directory for comparison artifacts")
    parser.add_argument("--min_tag_count", type=int, default=8, help="Minimum occurrences required to keep a tag")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on scanned files")

    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--calibration_min_threshold", type=float, default=0.12)
    parser.add_argument("--calibration_max_threshold", type=float, default=0.8)
    parser.add_argument("--calibration_step", type=float, default=0.02)
    parser.add_argument("--low_support_cutoff", type=int, default=6)
    parser.add_argument("--low_support_floor", type=float, default=0.24)

    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--max_depth", type=int, default=0, help="0 means None")
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

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

    run_comparison(
        folder=root,
        min_tag_count=max(1, int(args.min_tag_count)),
        max_files=args.max_files,
        out_dir=Path(args.out_dir),
        calibration_min_threshold=cal_min,
        calibration_max_threshold=cal_max,
        calibration_step=cal_step,
        low_support_cutoff=max(1, int(args.low_support_cutoff)),
        low_support_floor=max(cal_min, min(cal_max, float(args.low_support_floor))),
        val_size=val_size,
        test_size=test_size,
        random_state=int(args.random_state),
        n_estimators=max(100, int(args.n_estimators)),
        max_depth=None if int(args.max_depth) <= 0 else int(args.max_depth),
        min_samples_leaf=max(1, int(args.min_samples_leaf)),
    )


if __name__ == "__main__":
    main()
