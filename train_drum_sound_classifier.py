import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")

# Order matters: first match wins
LABEL_PATTERNS: List[Tuple[str, str]] = [
    ("kick", r"(?i)(^|[^a-z])kick([^a-z]|$)|\bbd\b|808\s*kick"),
    ("snare", r"(?i)(^|[^a-z])snare([^a-z]|$)|\bsd\b"),
    ("hihat", r"(?i)\bhat\b|hihat|hi[-_ ]hat|\bhh\b|openhat|closedhat"),
    ("clap", r"(?i)(^|[^a-z])clap([^a-z]|$)"),
    ("tom", r"(?i)(^|[^a-z])tom([^a-z]|$)"),
    ("perc", r"(?i)perc|percussion|rim|cowbell|shaker|tamb"),
]


def infer_label(path: Path) -> Optional[str]:
    s = str(path).lower()
    for label, pat in LABEL_PATTERNS:
        if re.search(pat, s):
            return label
    return None


def collect_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
    return sorted(files)


def extract_one_shot_features(file_path: Path, sr: int = 22050, max_seconds: float = 1.0) -> Optional[Dict[str, float]]:
    try:
        y, fs = librosa.load(str(file_path), sr=sr, mono=True)
    except Exception:
        return None

    if y is None or len(y) < 512:
        return None

    max_len = int(max_seconds * fs)
    y = y[:max_len]

    # Trim silence around one-shots
    try:
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception:
        pass

    if len(y) < 512:
        return None

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=fs)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=fs)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=fs, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    onset_env = librosa.onset.onset_strength(y=y, sr=fs)
    onset_count = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=fs))

    mel = librosa.feature.melspectrogram(y=y, sr=fs, n_mels=64)
    log_mel = librosa.power_to_db(mel + 1e-10)
    mfcc = librosa.feature.mfcc(S=log_mel, sr=fs, n_mfcc=13)

    # Sub-band energies for drum timbre
    stft = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=fs, n_fft=2048)

    def band_energy(fmin: float, fmax: float) -> float:
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return 0.0
        return float(np.mean(stft[mask, :]))

    low_e = band_energy(20, 180)
    mid_e = band_energy(180, 2500)
    high_e = band_energy(5000, 12000)
    total_e = low_e + mid_e + high_e + 1e-8

    feats: Dict[str, float] = {
        "duration_sec": float(len(y) / fs),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "flatness_mean": float(np.mean(flatness)),
        "onset_strength_mean": float(np.mean(onset_env)) if onset_env.size else 0.0,
        "onset_strength_std": float(np.std(onset_env)) if onset_env.size else 0.0,
        "onset_count": float(onset_count),
        "low_band_ratio": float(low_e / total_e),
        "mid_band_ratio": float(mid_e / total_e),
        "high_band_ratio": float(high_e / total_e),
    }

    for i in range(mfcc.shape[0]):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    return feats


def build_dataset(root: Path, min_per_class: int = 20, max_files: Optional[int] = None) -> pd.DataFrame:
    files = collect_audio_files(root)
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    rows: List[Dict] = []
    for i, p in enumerate(files, start=1):
        label = infer_label(p)
        if label is None:
            continue

        feats = extract_one_shot_features(p)
        if feats is None:
            continue

        row = {"file": str(p), "label": label}
        row.update(feats)
        rows.append(row)

        if i % 200 == 0:
            print(f"processed {i}/{len(files)} files")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    counts = df["label"].value_counts()
    keep = counts[counts >= min_per_class].index.tolist()
    df = df[df["label"].isin(keep)].reset_index(drop=True)
    return df


def train_classifier(df: pd.DataFrame, model_out: Path, report_out: Path) -> None:
    feature_cols = [c for c in df.columns if c not in {"file", "label"}]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    labels_sorted = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": clf, "feature_cols": feature_cols, "labels": labels_sorted}, model_out)

    payload = {
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_counts": y.value_counts().to_dict(),
        "report": report_dict,
        "labels": labels_sorted,
        "confusion_matrix": cm.tolist(),
    }
    report_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved model: {model_out}")
    print(f"Saved report: {report_out}")
    print("Class counts:")
    print(y.value_counts())
    print("Weighted F1:", round(report_dict.get("weighted avg", {}).get("f1-score", 0.0), 4))


def main():
    parser = argparse.ArgumentParser(description="Train a drum-sound classifier from one-shot sample folders")
    parser.add_argument("--folder", required=True, help="Root folder with drum samples")
    parser.add_argument("--min_per_class", type=int, default=20, help="Min samples required to keep a class")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on total files scanned")
    parser.add_argument("--dataset_out", default="training/models/drum_sound_dataset.csv", help="Path to save extracted training dataset CSV")
    parser.add_argument("--model_out", default="training/models/drum_sound_classifier.joblib", help="Path to save trained classifier bundle")
    parser.add_argument("--report_out", default="training/models/drum_sound_classifier_report.json", help="Path to save metrics report JSON")
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    df = build_dataset(root, min_per_class=args.min_per_class, max_files=args.max_files)
    if df.empty:
        raise RuntimeError("No labeled samples found. Check filename/folder naming patterns.")

    dataset_out = Path(args.dataset_out)
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_out, index=False)
    print(f"Saved dataset: {dataset_out} (rows={len(df)})")

    train_classifier(df, Path(args.model_out), Path(args.report_out))


if __name__ == "__main__":
    main()
