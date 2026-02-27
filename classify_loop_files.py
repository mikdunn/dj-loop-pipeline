import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from train_loop_file_classifier import collect_audio_files, extract_loop_features


def classify_folder(model_path: Path, folder: Path, out_csv: Path, min_conf: float = 0.0) -> None:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols: List[str] = bundle["feature_cols"]

    files = collect_audio_files(folder)
    rows: List[Dict] = []

    for i, p in enumerate(files, start=1):
        feats = extract_loop_features(p)
        if feats is None:
            continue

        x = pd.DataFrame([feats])
        for c in feature_cols:
            if c not in x.columns:
                x[c] = 0.0
        x = x[feature_cols]

        probs = model.predict_proba(x)[0]
        classes = model.classes_
        idx = int(np.argmax(probs))
        pred = str(classes[idx])
        conf = float(probs[idx])

        if conf < min_conf:
            pred = "uncertain"

        row = {
            "file": str(p),
            "predicted_label": pred,
            "confidence": conf,
        }
        for cls, pr in zip(classes, probs):
            row[f"prob_{cls}"] = float(pr)
        rows.append(row)

        if i % 100 == 0:
            print(f"classified {i}/{len(files)} files")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved predictions: {out_csv}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print("Predicted label counts:")
        print(df["predicted_label"].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Classify loop audio files with trained loop classifier")
    parser.add_argument("--model", default="training/models/loop_file_classifier.joblib", help="Trained model bundle")
    parser.add_argument("--folder", required=True, help="Folder with loop audio files")
    parser.add_argument("--out", default="training/models/loop_file_predictions.csv", help="Output CSV")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Confidence threshold for uncertain label")
    args = parser.parse_args()

    classify_folder(
        model_path=Path(args.model),
        folder=Path(args.folder),
        out_csv=Path(args.out),
        min_conf=max(0.0, min(1.0, args.min_conf)),
    )


if __name__ == "__main__":
    main()
