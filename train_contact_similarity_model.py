import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

EXTS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")


def collect_audio_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dp, _, fs in os.walk(root):
        for n in fs:
            p = Path(dp) / n
            if p.suffix.lower() in EXTS:
                out.append(p)
    return sorted(out)


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-8)


def extract_pattern_sequence(file_path: Path, sr: int = 22050, max_seconds: float = 20.0, max_frames: int = 450) -> Optional[np.ndarray]:
    try:
        y, fs = librosa.load(str(file_path), sr=sr, mono=True)
    except Exception:
        return None

    if y is None or len(y) < 4096:
        return None

    y = y[: int(max_seconds * fs)]
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=fs, hop_length=hop)
    if onset.size < 8:
        return None

    onset_z = _z(onset)
    onset_delta = np.diff(onset_z, prepend=onset_z[0])

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=fs, n_fft=2048)
    low_mask = (freqs >= 20.0) & (freqs <= 180.0)
    low_env = _z(np.mean(S[low_mask, :], axis=0)) if np.any(low_mask) else np.zeros_like(onset_z)

    t = min(len(onset_z), len(onset_delta), len(low_env))
    X = np.vstack([onset_z[:t], onset_delta[:t], low_env[:t]]).astype(np.float32)

    if X.shape[1] > max_frames:
        idx = np.linspace(0, X.shape[1] - 1, max_frames).astype(int)
        X = X[:, idx]

    return X


def pair_distance_overlap(Xi: np.ndarray, Xj: np.ndarray) -> Tuple[float, float]:
    try:
        D, wp = librosa.sequence.dtw(X=Xi, Y=Xj, metric="cosine", subseq=True)
    except Exception:
        return float("inf"), 0.0

    if D.size == 0:
        return float("inf"), 0.0

    end_i, end_j = wp[-1]
    path_len = max(1, len(wp))
    dist = float(D[end_i, end_j] / path_len)
    overlap = float(min(1.0, path_len / max(Xi.shape[1], Xj.shape[1])))
    return dist, overlap


def build_raw_similarity(feats: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(feats)
    dist = np.zeros((n, n), dtype=float)
    ov = np.zeros((n, n), dtype=float)
    for i in range(n):
        ov[i, i] = 1.0
        for j in range(i + 1, n):
            d, o = pair_distance_overlap(feats[i], feats[j])
            dist[i, j] = dist[j, i] = d
            ov[i, j] = ov[j, i] = o

    vals = dist[np.isfinite(dist) & (dist > 0)]
    temp = float(np.median(vals)) if vals.size else 1.0
    temp = max(1e-6, temp)
    sim = np.exp(-dist / temp) * np.clip(ov, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim, dist, ov


def self_tuned_knn(dist: np.ndarray, ov: np.ndarray, k: int = 7, mutual: bool = True) -> np.ndarray:
    n = dist.shape[0]

    def kth_positive(row: np.ndarray, k_: int) -> float:
        v = np.sort(row[np.isfinite(row) & (row > 0)])
        if v.size == 0:
            return 1.0
        idx = min(max(0, k_ - 1), v.size - 1)
        return float(max(1e-8, v[idx]))

    k = max(2, int(k))
    sig = np.array([kth_positive(dist[i], k) for i in range(n)], dtype=float)
    denom = np.outer(sig, sig) + 1e-12
    sim = np.exp(-(dist ** 2) / denom) * np.clip(ov, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)

    knn = np.zeros((n, n), dtype=bool)
    for i in range(n):
        r = sim[i].copy()
        r[i] = -np.inf
        nn = np.argsort(r)[::-1][:k]
        knn[i, nn] = True

    mask = (knn & knn.T) if mutual else (knn | knn.T)
    out = sim * mask.astype(float)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 1.0)
    return out


def sinkhorn(sim: np.ndarray, iters: int = 25, eps: float = 1e-8) -> np.ndarray:
    x = np.maximum(sim.copy(), eps)
    for _ in range(max(1, int(iters))):
        x = x / (np.sum(x, axis=1, keepdims=True) + eps)
        x = x / (np.sum(x, axis=0, keepdims=True) + eps)
    x = 0.5 * (x + x.T)
    x = x / (np.max(x) + eps)
    np.fill_diagonal(x, 1.0)
    return x


def sharpen(sim_raw: np.ndarray, dist: np.ndarray, ov: np.ndarray, mode: str) -> np.ndarray:
    m = mode.strip().lower()
    if m == "none":
        return sim_raw.copy()
    if m == "self_tuned_knn":
        return self_tuned_knn(dist, ov, k=7, mutual=True)
    if m == "sinkhorn":
        return sinkhorn(sim_raw, iters=25, eps=1e-8)
    raise ValueError("mode must be one of: none, self_tuned_knn, sinkhorn")


def fiedler_labels(sim: np.ndarray, n_bins: int = 4) -> np.ndarray:
    n = sim.shape[0]
    deg = np.sum(sim, axis=1)
    D = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    L = np.eye(n) - D @ sim @ D
    _, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1] if n >= 2 else np.zeros(n)

    q = np.linspace(0, 1, max(2, int(n_bins)) + 1)
    edges = np.quantile(fiedler, q)
    y = np.zeros(n, dtype=int)
    for i, v in enumerate(fiedler):
        y[i] = int(np.searchsorted(edges[1:-1], v, side="right"))
    return y


def extract_loop_features(file_path: Path, sr: int = 22050, max_seconds: float = 12.0) -> Optional[dict]:
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

    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13)

    out = {
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
    }
    for i in range(mfcc.shape[0]):
        out[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        out[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
    return out


def run_train(folder: Path, out_dir: Path, mode: str, max_files: int, n_bins: int, random_state: int = 42) -> Path:
    files = collect_audio_files(folder)
    if max_files > 0:
        files = files[:max_files]
    if len(files) < 12:
        raise RuntimeError("Need at least 12 files for train/eval")

    feats_seq: List[np.ndarray] = []
    keep_files: List[Path] = []
    for p in files:
        X = extract_pattern_sequence(p)
        if X is not None:
            feats_seq.append(X)
            keep_files.append(p)

    if len(keep_files) < 12:
        raise RuntimeError("Too few valid files after sequence extraction")

    sim_raw, dist, ov = build_raw_similarity(feats_seq)
    sim = sharpen(sim_raw, dist, ov, mode=mode)
    y = fiedler_labels(sim, n_bins=n_bins)

    rows = []
    for p, yy in zip(keep_files, y):
        f = extract_loop_features(p)
        if f is None:
            continue
        r = {"file": str(p), "label": int(yy)}
        r.update(f)
        rows.append(r)

    df = pd.DataFrame(rows)
    if df["label"].nunique() < 2:
        raise RuntimeError("Label bins collapsed to one class; increase max_files or adjust mode")

    feature_cols = [c for c in df.columns if c not in {"file", "label"}]
    X = df[feature_cols]
    yv = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        yv,
        test_size=0.25,
        random_state=random_state,
        stratify=yv,
    )

    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    report = {
        "mode": mode,
        "n_rows": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "random_state": int(random_state),
        "max_files": int(max_files),
        "n_bins": int(n_bins),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = out_dir / f"contact_train_dataset_{mode}.csv"
    model_path = out_dir / f"contact_rf_model_{mode}.joblib"
    report_path = out_dir / f"contact_train_report_{mode}.json"

    df.to_csv(dataset_csv, index=False)
    joblib.dump({"model": clf, "feature_cols": feature_cols, "mode": mode, "n_bins": n_bins}, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved dataset: {dataset_csv}")
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(f"macro_f1={report['macro_f1']:.4f}, weighted_f1={report['weighted_f1']:.4f}, accuracy={report['accuracy']:.4f}")

    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train contact-similarity classifier with selectable sharpening mode")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out_dir", default="training/models/retrain_compare")
    ap.add_argument("--mode", choices=("none", "self_tuned_knn", "sinkhorn"), default="self_tuned_knn")
    ap.add_argument("--max_files", type=int, default=120)
    ap.add_argument("--n_bins", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    run_train(
        folder=Path(args.folder),
        out_dir=Path(args.out_dir),
        mode=args.mode,
        max_files=max(12, int(args.max_files)),
        n_bins=max(2, int(args.n_bins)),
        random_state=int(args.random_state),
    )


if __name__ == "__main__":
    main()
