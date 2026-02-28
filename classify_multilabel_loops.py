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
    contact_model_path: Path = None,
    contact_min_prob: float = 0.35,
    contact_use_calibrated_thresholds: bool = True,
    contact_prefix: str = "contact",
) -> None:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols: List[str] = bundle["feature_cols"]
    classes: List[str] = bundle["classes"]
    thresholds: Dict[str, float] = bundle.get("thresholds", {})

    contact_bundle = None
    contact_model = None
    contact_feature_cols: List[str] = []
    contact_classes: List[str] = []
    contact_thresholds: Dict[str, float] = {}
    if contact_model_path is not None:
        contact_bundle = joblib.load(contact_model_path)
        contact_model = contact_bundle["model"]
        contact_feature_cols = contact_bundle["feature_cols"]
        contact_classes = contact_bundle["classes"]
        contact_thresholds = contact_bundle.get("thresholds", {})

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

        if contact_bundle is not None and contact_model is not None:
            x_contact = pd.DataFrame([feats])
            for col in contact_feature_cols:
                if col not in x_contact.columns:
                    x_contact[col] = 0.0
            x_contact = x_contact[contact_feature_cols]

            c_probs = np.asarray(contact_model.predict_proba(x_contact)).reshape(-1)

            def _contact_threshold_for(cls: str) -> float:
                if contact_use_calibrated_thresholds and cls in contact_thresholds:
                    return float(contact_thresholds[cls])
                return float(contact_min_prob)

            c_tags = [contact_classes[j] for j, pr in enumerate(c_probs) if pr >= _contact_threshold_for(contact_classes[j])]
            if not c_tags:
                c_tags = [contact_classes[int(np.argmax(c_probs))]]

            row[f"predicted_{contact_prefix}_tags"] = ",".join(c_tags)
            for cls, pr in zip(contact_classes, c_probs):
                row[f"prob_{contact_prefix}_{cls}"] = float(pr)

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
    parser.add_argument("--contact_model", default=None, help="Optional contact-map model bundle (.joblib)")
    parser.add_argument("--contact_min_prob", type=float, default=0.35, help="Fallback probability threshold for contact head")
    parser.add_argument(
        "--contact_disable_calibrated_thresholds",
        action="store_true",
        help="Ignore calibrated per-label thresholds for contact head and use --contact_min_prob globally.",
    )
    parser.add_argument("--contact_prefix", default="contact", help="Column prefix for contact head outputs")
    args = parser.parse_args()

    classify_folder(
        model_path=Path(args.model),
        folder=Path(args.folder),
        out_csv=Path(args.out),
        min_prob=max(0.0, min(1.0, args.min_prob)),
        use_calibrated_thresholds=not args.disable_calibrated_thresholds,
        contact_model_path=Path(args.contact_model) if args.contact_model else None,
        contact_min_prob=max(0.0, min(1.0, args.contact_min_prob)),
        contact_use_calibrated_thresholds=not args.contact_disable_calibrated_thresholds,
        contact_prefix=str(args.contact_prefix).strip() or "contact",
    )


if __name__ == "__main__":
    main()
