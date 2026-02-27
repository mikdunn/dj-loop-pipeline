import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")

TAG_PATTERNS = {
    "dnb": r"\bdnb\b|drum\s*&\s*bass|drum\s*n\s*bass",
    "breaks": r"breakbeat|\bbreaks?\b|break-y|breaky",
    "house": r"\bhouse\b",
    "techno": r"\btechno\b",
    "disco": r"\bdisco\b",
    "downbeat": r"\bdownbeat\b",
    "triphop": r"trip\s*hop",
    "metal": r"\bmetal\b",
    "punk": r"\bpunk\b",
    "rock": r"\brock\b",
    "funk": r"\bfunk\b|funk-y|funky",
    "jazz": r"\bjazz\b|jazz-y|jazzy",
    "soul": r"\bsoul\b",
    "groove": r"\bgroove\b|groove-y|groovey",
    "march": r"\bmarch\b|marching",
    "chilled": r"\bchilled\b|\bchill\b|spacey|laid\s*back",
    "electronic": r"electronic|electronica",
}


def collect_audio_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                out.append(p)
    return sorted(out)


def infer_tags(path: Path) -> List[str]:
    s = path.name.lower()
    tags: List[str] = []
    for tag, pat in TAG_PATTERNS.items():
        if re.search(pat, s):
            tags.append(tag)
    # common normalization
    if "dnb" in tags and "breaks" not in tags:
        tags.append("breaks")
    return sorted(set(tags))


def extract_loop_features(file_path: Path, sr: int = 22050, max_seconds: float = 12.0) -> Optional[Dict[str, float]]:
    try:
        y, fs = librosa.load(str(file_path), sr=sr, mono=True)
    except Exception:
        return None

    if y is None or len(y) < 4096:
        return None

    y = y[: int(max_seconds * fs)]

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

    if onset_env.size > 0:
        ac = np.correlate(onset_env, onset_env, mode="full")
        ac = ac[ac.size // 2:]
        ac = ac / (ac[0] + 1e-8)
        pulse_clarity = float(np.max(ac[1:min(len(ac), 120)])) if len(ac) > 2 else 0.0
    else:
        pulse_clarity = 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=20)

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


def build_dataset(root: Path, min_tag_count: int = 8, max_files: Optional[int] = None) -> pd.DataFrame:
    files = collect_audio_files(root)
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    rows: List[Dict] = []
    tag_counter: Dict[str, int] = {}

    for i, p in enumerate(files, start=1):
        tags = infer_tags(p)
        if not tags:
            continue

        feats = extract_loop_features(p)
        if feats is None:
            continue

        for t in tags:
            tag_counter[t] = tag_counter.get(t, 0) + 1

        row = {"file": str(p), "tags": tags}
        row.update(feats)
        rows.append(row)

        if i % 200 == 0:
            print(f"processed {i}/{len(files)} files")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    keep_tags: Set[str] = {t for t, c in tag_counter.items() if c >= min_tag_count}
    if not keep_tags:
        return pd.DataFrame()

    def _filter_tags(ts: List[str]) -> List[str]:
        return [t for t in ts if t in keep_tags]

    df["tags"] = df["tags"].apply(_filter_tags)
    df = df[df["tags"].apply(len) > 0].reset_index(drop=True)
    return df


def calibrate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str],
    min_thr: float = 0.10,
    max_thr: float = 0.85,
    step: float = 0.02,
    low_support_cutoff: int = 6,
    low_support_floor: float = 0.24,
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, float]]:
    """Calibrate one threshold per label by maximizing per-label F1 on validation set.

    Policy guardrails:
    - thresholds are searched only within [min_thr, max_thr]
    - for labels with support < low_support_cutoff, threshold is floored at low_support_floor
    """
    thresholds: Dict[str, float] = {}
    best_f1s: Dict[str, float] = {}

    candidate_thresholds = np.arange(min_thr, max_thr + 1e-9, step)
    y_pred = np.zeros_like(y_true)

    for i, label in enumerate(classes):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        support = int(np.sum(yt))

        # If no positives in validation for this label, keep conservative threshold.
        if support == 0:
            thresholds[label] = 0.50
            best_f1s[label] = 0.0
            y_pred[:, i] = (yp >= 0.50).astype(int)
            continue

        best_thr = 0.50
        best_f1 = -1.0
        for thr in candidate_thresholds:
            pred = (yp >= float(thr)).astype(int)
            f1 = float(f1_score(yt, pred, average="binary", zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        effective_thr = float(best_thr)
        if support < low_support_cutoff:
            effective_thr = max(effective_thr, float(low_support_floor))

        # Always keep threshold in configured bounds.
        effective_thr = max(float(min_thr), min(float(max_thr), effective_thr))

        thresholds[label] = effective_thr
        y_pred[:, i] = (yp >= effective_thr).astype(int)
        best_f1s[label] = float(f1_score(yt, y_pred[:, i], average="binary", zero_division=0))

    return thresholds, y_pred, best_f1s


def iterative_multilabel_split_indices(
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate iterative multi-label stratification.

    Returns (train_indices, test_indices), trying to preserve per-label prevalence.
    """
    n = int(y.shape[0])
    if n <= 1:
        idx = np.arange(n, dtype=int)
        return idx, np.array([], dtype=int)

    target_test = int(round(n * float(test_size)))
    target_test = max(1, min(n - 1, target_test))

    rng = np.random.default_rng(int(random_state))
    remaining = set(range(n))
    test_indices: List[int] = []

    y_int = y.astype(int)
    label_remaining = y_int.sum(axis=0).astype(float)
    label_target_test = np.round(label_remaining * float(test_size)).astype(float)
    label_in_test = np.zeros(y_int.shape[1], dtype=float)

    while len(test_indices) < target_test and len(remaining) > 0:
        need = label_target_test - label_in_test
        positive_needs = np.where(need > 1e-9)[0]

        if positive_needs.size == 0:
            break

        rarity_order = sorted(
            positive_needs.tolist(),
            key=lambda j: (label_remaining[j], -need[j]),
        )
        chosen_label = rarity_order[0]

        candidates = [i for i in remaining if y_int[i, chosen_label] == 1]
        if not candidates:
            label_target_test[chosen_label] = label_in_test[chosen_label]
            continue

        best_score = None
        best_candidates: List[int] = []
        for i in candidates:
            yi = y_int[i]
            score = float(np.sum(np.maximum(need, 0.0) * yi / (label_remaining + 1e-6)))
            if best_score is None or score > best_score + 1e-12:
                best_score = score
                best_candidates = [i]
            elif abs(score - float(best_score)) <= 1e-12:
                best_candidates.append(i)

        selected = int(rng.choice(best_candidates))
        test_indices.append(selected)
        remaining.remove(selected)

        yi = y_int[selected].astype(float)
        label_in_test += yi
        label_remaining = np.maximum(0.0, label_remaining - yi)

    if len(test_indices) < target_test and len(remaining) > 0:
        remaining_list = list(remaining)
        rng.shuffle(remaining_list)
        fill_count = target_test - len(test_indices)
        fill = remaining_list[:fill_count]
        test_indices.extend(fill)
        remaining.difference_update(fill)

    test_idx = np.array(sorted(test_indices), dtype=int)
    train_idx = np.array(sorted(remaining), dtype=int)
    return train_idx, test_idx


def multilabel_train_val_test_indices(
    y: np.ndarray,
    val_size: float,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if y.shape[0] < 5:
        raise ValueError("Need at least 5 rows for train/val/test split")

    test_size = max(0.05, min(0.45, float(test_size)))
    val_size = max(0.05, min(0.45, float(val_size)))
    if val_size + test_size >= 0.8:
        val_size = 0.2
        test_size = 0.2

    train_val_idx, test_idx = iterative_multilabel_split_indices(
        y,
        test_size=test_size,
        random_state=random_state,
    )

    rel_val = float(val_size) / max(1e-6, (1.0 - float(test_size)))
    rel_val = max(0.05, min(0.5, rel_val))

    y_train_val = y[train_val_idx]
    train_idx_rel, val_idx_rel = iterative_multilabel_split_indices(
        y_train_val,
        test_size=rel_val,
        random_state=random_state + 17,
    )

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    return train_idx, val_idx, test_idx


def predict_with_thresholds(y_prob: np.ndarray, classes: List[str], thresholds: Dict[str, float]) -> np.ndarray:
    y_pred = np.zeros_like(y_prob, dtype=int)
    for i, label in enumerate(classes):
        thr = float(thresholds.get(label, 0.5))
        y_pred[:, i] = (y_prob[:, i] >= thr).astype(int)
    return y_pred


def build_label_metrics(y_true: np.ndarray, y_pred_default: np.ndarray, y_pred_cal: np.ndarray, classes: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(classes):
        out[label] = {
            "support": int(np.sum(y_true[:, i])),
            "f1_default": float(f1_score(y_true[:, i], y_pred_default[:, i], average="binary", zero_division=0)),
            "f1_calibrated": float(f1_score(y_true[:, i], y_pred_cal[:, i], average="binary", zero_division=0)),
        }
    return out


def fit_and_evaluate_multilabel(
    df: pd.DataFrame,
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
) -> Tuple[Dict, Dict]:
    feature_cols = [c for c in df.columns if c not in {"file", "tags"}]
    x = df[feature_cols]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["tags"])
    classes = mlb.classes_.tolist()

    train_idx, val_idx, test_idx = multilabel_train_val_test_indices(
        y,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    x_train = x.iloc[train_idx]
    x_val = x.iloc[val_idx]
    x_test = x.iloc[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    base = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        random_state=int(random_state),
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=int(min_samples_leaf),
    )
    clf = OneVsRestClassifier(base)
    clf.fit(x_train, y_train)

    y_prob_val = np.asarray(clf.predict_proba(x_val))
    y_prob_test = np.asarray(clf.predict_proba(x_test))

    y_pred_val_default = (y_prob_val >= 0.5).astype(int)
    y_pred_test_default = (y_prob_test >= 0.5).astype(int)

    thresholds, y_pred_val_cal, _ = calibrate_thresholds(
        y_val,
        y_prob_val,
        classes,
        min_thr=calibration_min_threshold,
        max_thr=calibration_max_threshold,
        step=calibration_step,
        low_support_cutoff=low_support_cutoff,
        low_support_floor=low_support_floor,
    )
    y_pred_test_cal = predict_with_thresholds(y_prob_test, classes, thresholds)

    val_micro_default = float(f1_score(y_val, y_pred_val_default, average="micro", zero_division=0))
    val_macro_default = float(f1_score(y_val, y_pred_val_default, average="macro", zero_division=0))
    val_micro_cal = float(f1_score(y_val, y_pred_val_cal, average="micro", zero_division=0))
    val_macro_cal = float(f1_score(y_val, y_pred_val_cal, average="macro", zero_division=0))

    test_micro_default = float(f1_score(y_test, y_pred_test_default, average="micro", zero_division=0))
    test_macro_default = float(f1_score(y_test, y_pred_test_default, average="macro", zero_division=0))
    test_micro_cal = float(f1_score(y_test, y_pred_test_cal, average="micro", zero_division=0))
    test_macro_cal = float(f1_score(y_test, y_pred_test_cal, average="macro", zero_division=0))

    report_payload = {
        "n_rows": int(len(df)),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "n_classes": int(len(classes)),
        "classes": classes,
        "random_state": int(random_state),
        "split_strategy": "iterative_multilabel_stratified",
        "val_metrics": {
            "micro_f1_default": val_micro_default,
            "macro_f1_default": val_macro_default,
            "micro_f1_calibrated": val_micro_cal,
            "macro_f1_calibrated": val_macro_cal,
        },
        "test_metrics": {
            "micro_f1_default": test_micro_default,
            "macro_f1_default": test_macro_default,
            "micro_f1_calibrated": test_micro_cal,
            "macro_f1_calibrated": test_macro_cal,
        },
        "label_metrics_test": build_label_metrics(y_test, y_pred_test_default, y_pred_test_cal, classes),
        "thresholds": thresholds,
        "model_params": {
            "n_estimators": int(n_estimators),
            "max_depth": None if max_depth is None else int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
        },
        "calibration_policy": {
            "min_threshold": float(calibration_min_threshold),
            "max_threshold": float(calibration_max_threshold),
            "step": float(calibration_step),
            "low_support_cutoff": int(low_support_cutoff),
            "low_support_floor": float(low_support_floor),
        },
    }

    bundle = {
        "model": clf,
        "feature_cols": feature_cols,
        "classes": classes,
        "thresholds": thresholds,
        "calibration_policy": report_payload["calibration_policy"],
        "split_strategy": "iterative_multilabel_stratified",
        "random_state": int(random_state),
        "model_params": report_payload["model_params"],
    }
    return bundle, report_payload


def train_multilabel(
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
        df,
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
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, model_out)
    report_out.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"Saved model: {model_out}")
    print(f"Saved report: {report_out}")
    print("Validation Micro F1 (default=0.5):", round(report_payload["val_metrics"]["micro_f1_default"], 4))
    print("Validation Macro F1 (default=0.5):", round(report_payload["val_metrics"]["macro_f1_default"], 4))
    print("Validation Micro F1 (calibrated):", round(report_payload["val_metrics"]["micro_f1_calibrated"], 4))
    print("Validation Macro F1 (calibrated):", round(report_payload["val_metrics"]["macro_f1_calibrated"], 4))
    print("Test Micro F1 (default=0.5):", round(report_payload["test_metrics"]["micro_f1_default"], 4))
    print("Test Macro F1 (default=0.5):", round(report_payload["test_metrics"]["macro_f1_default"], 4))
    print("Test Micro F1 (calibrated):", round(report_payload["test_metrics"]["micro_f1_calibrated"], 4))
    print("Test Macro F1 (calibrated):", round(report_payload["test_metrics"]["macro_f1_calibrated"], 4))


def main():
    parser = argparse.ArgumentParser(description="Train multi-label loop tag classifier from full loop folders")
    parser.add_argument("--folder", required=True, help="Root folder containing loop files")
    parser.add_argument("--min_tag_count", type=int, default=8, help="Minimum occurrences required to keep a tag")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on scanned files")
    parser.add_argument("--dataset_out", default="training/models/loop_multilabel_dataset.csv", help="Dataset CSV output")
    parser.add_argument("--model_out", default="training/models/loop_multilabel_classifier.joblib", help="Model bundle output")
    parser.add_argument("--report_out", default="training/models/loop_multilabel_report.json", help="Metrics JSON output")
    parser.add_argument("--calibration_min_threshold", type=float, default=0.12, help="Minimum threshold allowed during calibration")
    parser.add_argument("--calibration_max_threshold", type=float, default=0.80, help="Maximum threshold allowed during calibration")
    parser.add_argument("--calibration_step", type=float, default=0.02, help="Threshold search step size")
    parser.add_argument("--low_support_cutoff", type=int, default=6, help="Validation positives below this are treated as low-support labels")
    parser.add_argument("--low_support_floor", type=float, default=0.24, help="Minimum threshold enforced for low-support labels")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for splitting and model")
    parser.add_argument("--n_estimators", type=int, default=600, help="RandomForest n_estimators")
    parser.add_argument("--max_depth", type=int, default=0, help="RandomForest max_depth (0 means None)")
    parser.add_argument("--min_samples_leaf", type=int, default=2, help="RandomForest min_samples_leaf")
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    df = build_dataset(root, min_tag_count=args.min_tag_count, max_files=args.max_files)
    if df.empty:
        raise RuntimeError("No labeled loop rows found for training")

    ds_out = Path(args.dataset_out)
    ds_out.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    df_to_save["tags"] = df_to_save["tags"].apply(lambda x: ",".join(x))
    df_to_save.to_csv(ds_out, index=False)
    print(f"Saved dataset: {ds_out} (rows={len(df_to_save)})")

    cal_min = max(0.0, min(1.0, float(args.calibration_min_threshold)))
    cal_max = max(0.0, min(1.0, float(args.calibration_max_threshold)))
    if cal_max < cal_min:
        cal_max = cal_min
    cal_step = max(0.005, min(0.2, float(args.calibration_step)))
    low_support_cutoff = max(1, int(args.low_support_cutoff))
    low_support_floor = max(cal_min, min(cal_max, float(args.low_support_floor)))
    val_size = max(0.05, min(0.45, float(args.val_size)))
    test_size = max(0.05, min(0.45, float(args.test_size)))
    if val_size + test_size >= 0.8:
        val_size = 0.2
        test_size = 0.2
    random_state = int(args.random_state)
    n_estimators = max(100, int(args.n_estimators))
    max_depth = None if int(args.max_depth) <= 0 else int(args.max_depth)
    min_samples_leaf = max(1, int(args.min_samples_leaf))

    train_multilabel(
        df,
        Path(args.model_out),
        Path(args.report_out),
        calibration_min_threshold=cal_min,
        calibration_max_threshold=cal_max,
        calibration_step=cal_step,
        low_support_cutoff=low_support_cutoff,
        low_support_floor=low_support_floor,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )


if __name__ == "__main__":
    main()
