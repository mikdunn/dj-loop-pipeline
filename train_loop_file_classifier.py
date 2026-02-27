import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")

# Priority order: first match wins for a single-label classifier.
LOOP_LABEL_PATTERNS = [
    ("dnb", r"\bdnb\b|drum\s*&\s*bass|drum\s*n\s*bass"),
    ("breaks", r"breakbeat|\bbreaks?\b|break-y|breaky"),
    ("house", r"\bhouse\b"),
    ("techno", r"\btechno\b"),
    ("disco", r"\bdisco\b"),
    ("downbeat", r"\bdownbeat\b"),
    ("triphop", r"trip\s*hop"),
    ("metal", r"\bmetal\b"),
    ("punk", r"\bpunk\b"),
    ("rock", r"\brock\b"),
    ("funk", r"\bfunk\b|funk-y|funky"),
    ("jazz", r"\bjazz\b|jazz-y|jazzy"),
    ("soul", r"\bsoul\b"),
    ("groove", r"\bgroove\b|groove-y|groovey"),
    ("march", r"\bmarch\b|marching"),
    ("chilled", r"\bchilled\b|\bchill\b|spacey|laid\s*back"),
    ("electronic", r"electronic|electronica"),
]


def infer_loop_label(path: Path) -> Optional[str]:
    s = path.name.lower()
    for label, pat in LOOP_LABEL_PATTERNS:
        if re.search(pat, s):
            return label
    return None


def collect_audio_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                out.append(p)
    return sorted(out)


def extract_loop_features(file_path: Path, sr: int = 22050, max_seconds: float = 12.0) -> Optional[Dict[str, float]]:
    try:
        y, fs = librosa.load(str(file_path), sr=sr, mono=True)
    except Exception:
        return None

    if y is None or len(y) < 4096:
        return None

    y = y[: int(max_seconds * fs)]

    # Core time/frequency features
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=fs)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=fs)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=fs, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    onset_env = librosa.onset.onset_strength(y=y, sr=fs)
    try:
        tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=fs, aggregate=np.median)[0])
    except Exception:
        tempo = 120.0

    try:
        y_h, y_p = librosa.effects.hpss(y)
        e_h = float(np.mean(y_h ** 2))
        e_p = float(np.mean(y_p ** 2))
        perc_ratio = e_p / (e_h + e_p + 1e-8)
    except Exception:
        perc_ratio = 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=20)

    # Pulse clarity via onset autocorrelation
    if onset_env.size > 0:
        ac = np.correlate(onset_env, onset_env, mode="full")
        ac = ac[ac.size // 2:]
        ac = ac / (ac[0] + 1e-8)
        pulse_clarity = float(np.max(ac[1: min(len(ac), 120)])) if len(ac) > 2 else 0.0
    else:
        pulse_clarity = 0.0

    feats: Dict[str, float] = {
        "duration_sec": float(len(y) / fs),
        "tempo_est": float(tempo if np.isfinite(tempo) else 120.0),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "flatness_mean": float(np.mean(flatness)),
        "onset_strength_mean": float(np.mean(onset_env)) if onset_env.size else 0.0,
        "onset_strength_std": float(np.std(onset_env)) if onset_env.size else 0.0,
        "percussive_ratio": float(perc_ratio),
        "pulse_clarity": float(pulse_clarity),
    }

    for i in range(mfcc.shape[0]):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    return feats


def build_dataset(root: Path, min_per_class: int = 10, max_files: Optional[int] = None) -> pd.DataFrame:
    files = collect_audio_files(root)
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    rows: List[Dict] = []
    for i, p in enumerate(files, start=1):
        label = infer_loop_label(p)
        if label is None:
            continue
        feats = extract_loop_features(p)
        if feats is None:
            continue
        row = {"file": str(p), "label": label}
        row.update(feats)
        rows.append(row)

        if i % 100 == 0:
            print(f"processed {i}/{len(files)} files")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    counts = df["label"].value_counts()
    keep_labels = counts[counts >= min_per_class].index.tolist()
    return df[df["label"].isin(keep_labels)].reset_index(drop=True)


def train_classifier(df: pd.DataFrame, model_out: Path, report_out: Path) -> None:
    feature_cols = [c for c in df.columns if c not in {"file", "label"}]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=700,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    labels = sorted(y.unique().tolist())
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": clf, "feature_cols": feature_cols, "labels": labels}, model_out)

    payload = {
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_counts": y.value_counts().to_dict(),
        "labels": labels,
        "report": report,
        "confusion_matrix": cm.tolist(),
    }
    report_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved model: {model_out}")
    print(f"Saved report: {report_out}")
    print("Class counts:")
    print(y.value_counts())
    print("Weighted F1:", round(report.get("weighted avg", {}).get("f1-score", 0.0), 4))


def main():
    parser = argparse.ArgumentParser(description="Train a loop-file classifier from known constructed loop files")
    parser.add_argument("--folder", required=True, help="Root folder containing loop audio files")
    parser.add_argument("--min_per_class", type=int, default=10, help="Minimum class frequency to keep")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on scanned files")
    parser.add_argument("--dataset_out", default="training/models/loop_classifier_dataset.csv", help="Output dataset CSV")
    parser.add_argument("--model_out", default="training/models/loop_file_classifier.joblib", help="Output trained model bundle")
    parser.add_argument("--report_out", default="training/models/loop_file_classifier_report.json", help="Output metrics JSON")
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    df = build_dataset(root, min_per_class=args.min_per_class, max_files=args.max_files)
    if df.empty:
        raise RuntimeError("No loop samples with valid labels were found")

    ds_out = Path(args.dataset_out)
    ds_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ds_out, index=False)
    print(f"Saved dataset: {ds_out} (rows={len(df)})")

    train_classifier(df, Path(args.model_out), Path(args.report_out))


if __name__ == "__main__":
    main()
