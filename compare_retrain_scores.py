import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def read_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pick_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "mode",
        "model_family",
        "leakage_mode",
        "n_rows",
        "n_classes",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        # graph-quality metrics (present in newer reports)
        "block_contrast",
        "spectral_gap",
        "mean_similarity",
        "hit_at_5",
        "hit_at_10",
        "nmi",
        "ari",
        "silhouette",
        "assortativity",
        # optional mask metrics for future image-style audio masks
        "mask_iou_mean",
        "mask_dice_mean",
        "laplacian_energy",
    ]
    return {k: report.get(k) for k in keys}


def safe_delta(a: Any, b: Any) -> Any:
    if pd.notna(a) and pd.notna(b):
        return float(a - b)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare previous matrix scores against new retrain metrics")
    ap.add_argument("--prev_metrics_csv", default="training/models/contact_map_compare/sharpen_mode_metrics.csv")
    ap.add_argument("--baseline_report", default="training/models/retrain_compare/contact_train_report_none.json")
    ap.add_argument("--new_report", default="training/models/retrain_compare/contact_train_report_self_tuned_knn.json")
    ap.add_argument("--out_csv", default="training/models/retrain_compare/retrain_score_comparison.csv")
    ap.add_argument("--out_json", default="training/models/retrain_compare/retrain_score_comparison.json")
    args = ap.parse_args()

    prev_csv = Path(args.prev_metrics_csv)
    if not prev_csv.exists():
        raise FileNotFoundError(f"Missing previous metrics CSV: {prev_csv}")

    prev = pd.read_csv(prev_csv)
    prev_pick = prev[prev["mode"].isin(["none", "self_tuned_knn"])].copy()

    baseline = pick_metrics(read_report(Path(args.baseline_report)))
    new = pick_metrics(read_report(Path(args.new_report)))

    if baseline.get("mode") is None:
        baseline["mode"] = "none"
    if new.get("mode") is None:
        new["mode"] = "self_tuned_knn"
    if baseline.get("model_family") is None:
        baseline["model_family"] = "rf"
    if new.get("model_family") is None:
        new["model_family"] = "rf"
    if baseline.get("leakage_mode") is None:
        baseline["leakage_mode"] = "transductive"
    if new.get("leakage_mode") is None:
        new["leakage_mode"] = "transductive"

    retrain_rows = pd.DataFrame([baseline, new])

    merged = prev_pick.merge(retrain_rows, on="mode", how="outer", suffixes=("_prev", "_new"))

    # delta of new-rule vs baseline on retrain metrics
    baseline_row = retrain_rows[retrain_rows["mode"] == "none"].iloc[0]
    new_row = retrain_rows[retrain_rows["mode"] == "self_tuned_knn"].iloc[0]

    summary = {
        "previous_matrix_metrics_file": str(prev_csv),
        "baseline_report": str(Path(args.baseline_report)),
        "new_report": str(Path(args.new_report)),
        "baseline_config": {
            "mode": baseline.get("mode"),
            "model_family": baseline.get("model_family"),
            "leakage_mode": baseline.get("leakage_mode"),
        },
        "new_config": {
            "mode": new.get("mode"),
            "model_family": new.get("model_family"),
            "leakage_mode": new.get("leakage_mode"),
        },
        "deltas_new_minus_baseline": {
            "accuracy": safe_delta(new_row.get("accuracy"), baseline_row.get("accuracy")),
            "macro_f1": safe_delta(new_row.get("macro_f1"), baseline_row.get("macro_f1")),
            "weighted_f1": safe_delta(new_row.get("weighted_f1"), baseline_row.get("weighted_f1")),
            "block_contrast": safe_delta(new_row.get("block_contrast"), baseline_row.get("block_contrast")),
            "spectral_gap": safe_delta(new_row.get("spectral_gap"), baseline_row.get("spectral_gap")),
        },
        "matrix_metric_rows": prev_pick.to_dict(orient="records"),
        "retrain_rows": retrain_rows.to_dict(orient="records"),
    }

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved comparison CSV: {out_csv}")
    print(f"Saved comparison JSON: {out_json}")
    print("Delta (new - baseline):", summary["deltas_new_minus_baseline"])


if __name__ == "__main__":
    main()
