import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from train_drum_sound_classifier import collect_audio_files, extract_one_shot_features


def classify_folder(model_path: Path, folder: Path, output_csv: Path, confidence_threshold: float = 0.0) -> None:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols: List[str] = bundle["feature_cols"]

    files = collect_audio_files(folder)
    rows: List[Dict] = []

    for i, p in enumerate(files, start=1):
        feats = extract_one_shot_features(p)
        if feats is None:
            continue

        x = pd.DataFrame([feats])
        for col in feature_cols:
            if col not in x.columns:
                x[col] = 0.0
        x = x[feature_cols]

        probs = model.predict_proba(x)[0]
        classes = model.classes_
        best_idx = int(np.argmax(probs))
        pred = str(classes[best_idx])
        conf = float(probs[best_idx])

        if conf < confidence_threshold:
            pred = "uncertain"

        row = {
            "file": str(p),
            "predicted_label": pred,
            "confidence": conf,
        }
        for cls, pr in zip(classes, probs):
            row[f"prob_{cls}"] = float(pr)
        rows.append(row)

        if i % 300 == 0:
            print(f"classified {i}/{len(files)} files")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved predictions: {output_csv}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print("Predicted label counts:")
        print(df["predicted_label"].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Classify drum sounds in a folder with a trained classifier")
    parser.add_argument("--model", default="training/models/drum_sound_classifier.joblib", help="Path to classifier model bundle")
    parser.add_argument("--folder", required=True, help="Folder with audio files to classify")
    parser.add_argument("--out", default="training/models/drum_sound_predictions.csv", help="Output CSV for predictions")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Confidence threshold below which label becomes 'uncertain'")
    args = parser.parse_args()

    classify_folder(
        model_path=Path(args.model),
        folder=Path(args.folder),
        output_csv=Path(args.out),
        confidence_threshold=max(0.0, min(1.0, args.min_conf)),
    )


if __name__ == "__main__":
    main()
