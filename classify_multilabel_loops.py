import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from train_multilabel_loop_classifier import collect_audio_files, extract_loop_features


def classify_folder(
    model_path: Path,
    folder: Path,
    out_csv: Path,
    min_prob: float = 0.35,
    use_calibrated_thresholds: bool = True,
) -> None:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols: List[str] = bundle["feature_cols"]
    classes: List[str] = bundle["classes"]
    thresholds: Dict[str, float] = bundle.get("thresholds", {})

    files = collect_audio_files(folder)
    rows: List[Dict] = []

    for i, p in enumerate(files, start=1):
        feats = extract_loop_features(p)
        if feats is None:
            continue

        x = pd.DataFrame([feats])
        for col in feature_cols:
            if col not in x.columns:
                x[col] = 0.0
        x = x[feature_cols]

        probs = model.predict_proba(x)
        probs = np.asarray(probs).reshape(-1)

        def _threshold_for(cls: str) -> float:
            if use_calibrated_thresholds and cls in thresholds:
                return float(thresholds[cls])
            return float(min_prob)

        tags = [classes[j] for j, pr in enumerate(probs) if pr >= _threshold_for(classes[j])]
        if not tags:
            best_idx = int(np.argmax(probs))
            tags = [classes[best_idx]]

        row = {
            "file": str(p),
            "predicted_tags": ",".join(tags),
        }
        for cls, pr in zip(classes, probs):
            row[f"prob_{cls}"] = float(pr)
        rows.append(row)

        if i % 200 == 0:
            print(f"classified {i}/{len(files)} files")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved predictions: {out_csv}")
    print(f"Rows: {len(df)}")


def main():
    parser = argparse.ArgumentParser(description="Classify loops with multi-label loop tag classifier")
    parser.add_argument("--model", default="training/models/loop_multilabel_classifier.joblib", help="Path to model bundle")
    parser.add_argument("--folder", required=True, help="Loop folder to classify")
    parser.add_argument("--out", default="training/models/loop_multilabel_predictions.csv", help="Output CSV")
    parser.add_argument("--min_prob", type=float, default=0.35, help="Per-tag probability threshold")
    parser.add_argument(
        "--disable_calibrated_thresholds",
        action="store_true",
        help="Ignore calibrated per-label thresholds in the model bundle and use --min_prob globally.",
    )
    args = parser.parse_args()

    classify_folder(
        model_path=Path(args.model),
        folder=Path(args.folder),
        out_csv=Path(args.out),
        min_prob=max(0.0, min(1.0, args.min_prob)),
        use_calibrated_thresholds=not args.disable_calibrated_thresholds,
    )


if __name__ == "__main__":
    main()
