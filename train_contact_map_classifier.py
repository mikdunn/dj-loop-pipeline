import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import joblib
import pandas as pd

from train_multilabel_loop_classifier import extract_loop_features, fit_and_evaluate_multilabel


def parse_contact_tags(raw_tags: str, contact_bin: Optional[str]) -> List[str]:
    tags: List[str] = []

    if isinstance(raw_tags, str) and raw_tags.strip():
        tags.extend([t.strip() for t in raw_tags.split(",") if t.strip()])

    if isinstance(contact_bin, str) and contact_bin.strip():
        tags.append(contact_bin.strip())

    return sorted(set(tags))


def build_contact_dataset(
    labels_csv: Path,
    min_tag_count: int = 8,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    labels = pd.read_csv(labels_csv)
    if "file" not in labels.columns:
        raise ValueError("Labels CSV must contain a 'file' column")

    # Accept contact_tags and/or contact_bin
    if "contact_tags" not in labels.columns and "contact_bin" not in labels.columns:
        raise ValueError("Labels CSV must contain 'contact_tags' and/or 'contact_bin' columns")

    rows: List[Dict] = []
    tag_counter: Dict[str, int] = {}

    work = labels
    if max_files is not None and max_files > 0:
        work = labels.head(int(max_files)).copy()

    for i, row in work.iterrows():
        file_path = Path(str(row["file"]))
        tags = parse_contact_tags(
            raw_tags=str(row["contact_tags"]) if "contact_tags" in work.columns else "",
            contact_bin=str(row["contact_bin"]) if "contact_bin" in work.columns else None,
        )
        if not tags:
            continue

        feats = extract_loop_features(file_path)
        if feats is None:
            continue

        for t in tags:
            tag_counter[t] = tag_counter.get(t, 0) + 1

        out_row = {"file": str(file_path), "tags": tags}
        out_row.update(feats)
        rows.append(out_row)

        if (i + 1) % 200 == 0:
            print(f"processed {i + 1}/{len(work)} labeled rows")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    keep_tags: Set[str] = {t for t, c in tag_counter.items() if c >= int(min_tag_count)}
    if not keep_tags:
        return pd.DataFrame()

    df["tags"] = df["tags"].apply(lambda ts: [t for t in ts if t in keep_tags])
    df = df[df["tags"].apply(len) > 0].reset_index(drop=True)
    return df


def train_contact_classifier(
    df: pd.DataFrame,
    model_out: Path,
    report_out: Path,
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
    bundle, report_payload = fit_and_evaluate_multilabel(
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
        label_source="contact_map",
    )

    bundle["head_name"] = "contact_map"

    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, model_out)
    report_out.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"Saved contact model: {model_out}")
    print(f"Saved contact report: {report_out}")
    print("Test Macro F1 (calibrated):", round(report_payload["test_metrics"]["macro_f1_calibrated"], 4))



def main() -> None:
    parser = argparse.ArgumentParser(description="Train contact-map multilabel classifier from contact_map_labels.csv")
    parser.add_argument("--labels_csv", required=True, help="Path to contact_map_labels.csv")
    parser.add_argument("--dataset_out", default="training/models/contact_map/contact_multilabel_dataset.csv")
    parser.add_argument("--model_out", default="training/models/contact_map/contact_multilabel_classifier.joblib")
    parser.add_argument("--report_out", default="training/models/contact_map/contact_multilabel_report.json")
    parser.add_argument("--min_tag_count", type=int, default=8)
    parser.add_argument("--max_files", type=int, default=None)

    parser.add_argument("--calibration_min_threshold", type=float, default=0.12)
    parser.add_argument("--calibration_max_threshold", type=float, default=0.8)
    parser.add_argument("--calibration_step", type=float, default=0.02)
    parser.add_argument("--low_support_cutoff", type=int, default=6)
    parser.add_argument("--low_support_floor", type=float, default=0.24)

    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--max_depth", type=int, default=0, help="0 means None")
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    args = parser.parse_args()

    df = build_contact_dataset(
        labels_csv=Path(args.labels_csv),
        min_tag_count=max(1, int(args.min_tag_count)),
        max_files=args.max_files,
    )
    if df.empty:
        raise RuntimeError("No usable labeled rows found from contact labels CSV")

    ds_out = Path(args.dataset_out)
    ds_out.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out["tags"] = df_out["tags"].apply(lambda x: ",".join(x))
    df_out.to_csv(ds_out, index=False)
    print(f"Saved contact dataset: {ds_out} (rows={len(df_out)})")

    cal_min = max(0.0, min(1.0, float(args.calibration_min_threshold)))
    cal_max = max(0.0, min(1.0, float(args.calibration_max_threshold)))
    if cal_max < cal_min:
        cal_max = cal_min
    cal_step = max(0.005, min(0.2, float(args.calibration_step)))
    low_support_floor = max(cal_min, min(cal_max, float(args.low_support_floor)))

    val_size = max(0.05, min(0.45, float(args.val_size)))
    test_size = max(0.05, min(0.45, float(args.test_size)))
    if val_size + test_size >= 0.8:
        val_size = 0.2
        test_size = 0.2

    train_contact_classifier(
        df=df,
        model_out=Path(args.model_out),
        report_out=Path(args.report_out),
        calibration_min_threshold=cal_min,
        calibration_max_threshold=cal_max,
        calibration_step=cal_step,
        low_support_cutoff=max(1, int(args.low_support_cutoff)),
        low_support_floor=low_support_floor,
        val_size=val_size,
        test_size=test_size,
        random_state=int(args.random_state),
        n_estimators=max(100, int(args.n_estimators)),
        max_depth=None if int(args.max_depth) <= 0 else int(args.max_depth),
        min_samples_leaf=max(1, int(args.min_samples_leaf)),
    )


if __name__ == "__main__":
    main()
