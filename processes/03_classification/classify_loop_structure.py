import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from analyze_loop_structure import analyze_file, collect_audio_files
from train_loop_structure_classifier import row_from_analysis


def classify_folder(
    folder: Path,
    model_path: Path,
    output_csv: Path,
    output_json: Path,
    drum_model: Optional[Path],
    sr: int,
    min_conf: float,
    structure_threshold: float,
    max_files: Optional[int],
) -> None:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols: List[str] = bundle["feature_cols"]

    drum_bundle = None
    if drum_model is not None and drum_model.exists():
        drum_bundle = joblib.load(drum_model)

    files = collect_audio_files(folder)
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    rows: List[Dict] = []
    details: List[Dict] = []

    for i, p in enumerate(files, start=1):
        result = analyze_file(
            p,
            drum_bundle=drum_bundle,
            sr=sr,
            min_conf=min_conf,
            structure_threshold=structure_threshold,
        )
        if result is None:
            continue

        base_row = row_from_analysis(result)
        x = pd.DataFrame([base_row])
        for c in feature_cols:
            if c not in x.columns:
                x[c] = 0.0
        x = x[feature_cols]

        probs = model.predict_proba(x)[0]
        classes = list(model.classes_)
        best_idx = int(np.argmax(probs))
        pred = str(classes[best_idx])
        conf = float(probs[best_idx])

        row = {
            "file": base_row["file"],
            "predicted_structure_class": pred,
            "confidence": conf,
            "chosen_dividers": base_row.get("chosen_dividers", 0),
            "structure_pattern": base_row.get("structure_pattern", ""),
            "structure_score": base_row.get("structure_score", 0.0),
            "repetition_score": base_row.get("repetition_score", 0.0),
            "half_similarity": base_row.get("half_similarity", 0.0),
            "superimpose_similarity": base_row.get("superimpose_similarity", 0.0),
        }
        for c, pr in zip(classes, probs):
            row[f"prob_{c}"] = float(pr)

        rows.append(row)

        details.append(
            {
                "file": base_row["file"],
                "predicted_structure_class": pred,
                "confidence": conf,
                "analysis": result,
            }
        )

        if i % 120 == 0:
            print(f"classified {i}/{len(files)} files")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)

    payload = {
        "n_files": int(len(details)),
        "folder": str(folder),
        "model": str(model_path),
        "details": details,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved structure predictions CSV: {output_csv}")
    print(f"Saved structure predictions JSON: {output_json}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print("Predicted class counts:")
        print(df["predicted_structure_class"].value_counts())


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify loop structure classes")
    parser.add_argument("--folder", required=True, help="Folder of loops to classify")
    parser.add_argument("--model", default="training/models/loop_structure_classifier.joblib", help="Structure classifier model")
    parser.add_argument("--drum_model", default="training/models/drum_sound_classifier.joblib", help="Optional drum model bundle")
    parser.add_argument("--out_csv", default="training/models/loop_structure_predictions.csv")
    parser.add_argument("--out_json", default="training/models/loop_structure_predictions.json")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--min_conf", type=float, default=0.25)
    parser.add_argument("--structure_threshold", type=float, default=0.82)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    classify_folder(
        folder=folder,
        model_path=Path(args.model),
        output_csv=Path(args.out_csv),
        output_json=Path(args.out_json),
        drum_model=Path(args.drum_model) if args.drum_model else None,
        sr=max(8000, int(args.sr)),
        min_conf=max(0.0, min(1.0, float(args.min_conf))),
        structure_threshold=max(0.5, min(0.99, float(args.structure_threshold))),
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
