import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from analyze_loop_structure import analyze_file, collect_audio_files


KNOWN_DRUM_LABELS = ["kick", "snare", "hihat", "clap", "tom", "perc", "uncertain", "unavailable"]


def canonical_structure_class(pattern: str, n_dividers: int, divider_labels: Optional[List[str]] = None) -> str:
    p = (pattern or "").strip().upper()
    if not p or len(p) < max(2, n_dividers):
        return "unknown"

    chars = list(p[:n_dividers])
    unique = len(set(chars))
    labels = [str(x).lower() for x in (divider_labels or []) if str(x).strip()]

    def _dominant_label() -> str:
        if not labels:
            return "unknown"
        counts: Dict[str, int] = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        top_label, top_count = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0]
        if top_count / max(1, len(labels)) >= 0.6:
            return top_label
        return "mixed"

    if unique == 1:
        return f"uniform_{_dominant_label()}"

    if n_dividers % 2 == 0:
        half = n_dividers // 2
        first = chars[:half]
        second = chars[half : half * 2]

        if first == second:
            return f"front_back_repeat_{_dominant_label()}"

        # strict alternation ABAB... style
        alt_ok = all(chars[i] == chars[i % 2] for i in range(n_dividers)) and len(set(chars[:2])) == 2
        if alt_ok:
            return f"alternating_{_dominant_label()}"

        # grouped halves like AABB (4) / AAAABBBB (8)
        if len(set(first)) == 1 and len(set(second)) == 1 and first[0] != second[0]:
            return f"front_back_contrast_{_dominant_label()}"

    if unique >= max(4, n_dividers // 2):
        return f"through_composed_{_dominant_label()}"

    return f"mixed_repeat_{_dominant_label()}"


def row_from_analysis(result: Dict) -> Dict:
    row: Dict = {
        "file": str(result.get("file", "")),
        "chosen_dividers": int(result.get("chosen_dividers", 0)),
        "duration_sec": float(result.get("duration_sec", 0.0)),
        "tempo_est": float(result.get("tempo_est", 120.0)),
        "structure_score": float(result.get("structure_score", 0.0)),
        "repetition_score": float(result.get("repetition_score", 0.0)),
        "half_similarity": float(result.get("half_similarity", 0.0)),
        "superimpose_similarity": float(result.get("superimpose_similarity", 0.0)),
        "score_4_dividers": float(result.get("score_4_dividers", 0.0)),
        "score_8_dividers": float(result.get("score_8_dividers", 0.0)),
        "offset_samples": float(result.get("offset_samples", 0.0)),
    }

    preds = result.get("divider_drum_predictions", []) or []
    labels = [str(p.get("predicted_label", "uncertain")) for p in preds]
    confs = [float(p.get("confidence", 0.0)) for p in preds]

    row["divider_count"] = float(len(preds))
    row["divider_conf_mean"] = float(np.mean(confs)) if confs else 0.0
    row["divider_conf_std"] = float(np.std(confs)) if confs else 0.0

    for lbl in KNOWN_DRUM_LABELS:
        c = labels.count(lbl)
        row[f"drum_count_{lbl}"] = float(c)
        row[f"drum_ratio_{lbl}"] = float(c / max(1, len(labels)))

    pattern = str(result.get("structure_pattern", ""))
    row["structure_pattern"] = pattern
    pat = [c for c in pattern]
    row["pattern_unique_count"] = float(len(set(pat))) if pat else 0.0
    if len(pat) > 1:
        transitions = sum(1 for i in range(1, len(pat)) if pat[i] != pat[i - 1])
        row["pattern_transition_ratio"] = float(transitions / (len(pat) - 1))
    else:
        row["pattern_transition_ratio"] = 0.0
    row["structure_class"] = canonical_structure_class(
        pattern,
        int(result.get("chosen_dividers", 0)),
        divider_labels=labels,
    )

    return row


def build_dataset(
    folder: Path,
    drum_model: Optional[Path],
    sr: int,
    min_conf: float,
    structure_threshold: float,
    max_files: Optional[int],
    min_class_count: int,
) -> pd.DataFrame:
    drum_bundle = None
    if drum_model is not None and drum_model.exists():
        drum_bundle = joblib.load(drum_model)

    files = collect_audio_files(folder)
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    rows: List[Dict] = []
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

        row = row_from_analysis(result)
        rows.append(row)

        if i % 120 == 0:
            print(f"processed {i}/{len(files)} files")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    counts = df["structure_class"].value_counts()
    keep = counts[counts >= int(min_class_count)].index.tolist()
    df = df[df["structure_class"].isin(keep)].reset_index(drop=True)
    return df


def train_classifier(df: pd.DataFrame, model_out: Path, report_out: Path) -> None:
    feature_cols = [
        c
        for c in df.columns
        if c not in {"file", "structure_pattern", "structure_class"}
        and not c.startswith("drum_count_")
        and not c.startswith("drum_ratio_")
    ]

    x = df[feature_cols]
    y = df["structure_class"]

    if y.nunique() < 2:
        raise RuntimeError(
            "Need at least 2 structure classes after filtering. "
            "Try raising --max_files, lowering --min_class_count, or retraining drum model for better divider diversity."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=700,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": clf,
            "feature_cols": feature_cols,
            "labels": labels,
            "task": "loop_structure_classification",
        },
        model_out,
    )

    payload = {
        "n_rows": int(len(df)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "classes": labels,
        "class_counts": y.value_counts().to_dict(),
        "report": report,
        "confusion_matrix": cm.tolist(),
    }
    report_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved model: {model_out}")
    print(f"Saved report: {report_out}")
    print("Weighted F1:", round(float(report.get("weighted avg", {}).get("f1-score", 0.0)), 4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train loop-structure classifier from repetition/superimposition analysis")
    parser.add_argument("--folder", required=True, help="Folder of loop files")
    parser.add_argument("--drum_model", default="training/models/drum_sound_classifier.joblib", help="Optional drum model bundle")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--min_conf", type=float, default=0.25)
    parser.add_argument("--structure_threshold", type=float, default=0.82)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--min_class_count", type=int, default=12, help="Min samples per structure class")
    parser.add_argument("--dataset_out", default="training/models/loop_structure_dataset.csv")
    parser.add_argument("--model_out", default="training/models/loop_structure_classifier.joblib")
    parser.add_argument("--report_out", default="training/models/loop_structure_classifier_report.json")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    drum_model = Path(args.drum_model) if args.drum_model else None

    df = build_dataset(
        folder=folder,
        drum_model=drum_model,
        sr=max(8000, int(args.sr)),
        min_conf=max(0.0, min(1.0, float(args.min_conf))),
        structure_threshold=max(0.5, min(0.99, float(args.structure_threshold))),
        max_files=args.max_files,
        min_class_count=max(2, int(args.min_class_count)),
    )

    if df.empty:
        raise RuntimeError("No usable structure rows found. Try raising --max_files or lowering --min_class_count.")

    ds_out = Path(args.dataset_out)
    ds_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ds_out, index=False)
    print(f"Saved dataset: {ds_out} (rows={len(df)})")

    train_classifier(df, Path(args.model_out), Path(args.report_out))


if __name__ == "__main__":
    main()
