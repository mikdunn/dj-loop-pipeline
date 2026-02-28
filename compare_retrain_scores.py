import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def read_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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

    baseline = read_report(Path(args.baseline_report))
    new = read_report(Path(args.new_report))

    retrain_rows = pd.DataFrame([
        {
            "mode": baseline.get("mode", "none"),
            "accuracy": baseline.get("accuracy"),
            "macro_f1": baseline.get("macro_f1"),
            "weighted_f1": baseline.get("weighted_f1"),
            "n_rows": baseline.get("n_rows"),
            "n_classes": baseline.get("n_classes"),
        },
        {
            "mode": new.get("mode", "self_tuned_knn"),
            "accuracy": new.get("accuracy"),
            "macro_f1": new.get("macro_f1"),
            "weighted_f1": new.get("weighted_f1"),
            "n_rows": new.get("n_rows"),
            "n_classes": new.get("n_classes"),
        },
    ])

    merged = prev_pick.merge(retrain_rows, on="mode", how="outer", suffixes=("_prev", "_new"))

    # delta of new-rule vs baseline on retrain metrics
    baseline_row = retrain_rows[retrain_rows["mode"] == "none"].iloc[0]
    new_row = retrain_rows[retrain_rows["mode"] == "self_tuned_knn"].iloc[0]

    summary = {
        "previous_matrix_metrics_file": str(prev_csv),
        "baseline_report": str(Path(args.baseline_report)),
        "new_report": str(Path(args.new_report)),
        "deltas_new_minus_baseline": {
            "accuracy": float(new_row["accuracy"] - baseline_row["accuracy"]),
            "macro_f1": float(new_row["macro_f1"] - baseline_row["macro_f1"]),
            "weighted_f1": float(new_row["weighted_f1"] - baseline_row["weighted_f1"]),
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
