import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")


def collect_audio_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                out.append(p)
    return sorted(out)


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return (x - mu) / (sd + 1e-8)


def extract_pattern_sequence(
    file_path: Path,
    sr: int = 22050,
    max_seconds: float = 20.0,
    hop_length: int = 512,
    max_frames: int = 600,
) -> Optional[np.ndarray]:
    """Extract compact rhythmic-pattern sequence (d x t) for alignment.

    Features are intentionally universal and mostly rhythm-focused:
    - onset strength envelope
    - onset delta (transient change)
    - low-band energy envelope
    - tempogram summary channel
    """
    try:
        y, fs = librosa.load(str(file_path), sr=sr, mono=True)
    except Exception:
        return None

    if y is None or len(y) < 4096:
        return None

    y = y[: int(max_seconds * fs)]

    onset = librosa.onset.onset_strength(y=y, sr=fs, hop_length=hop_length)
    if onset.size < 8:
        return None

    onset = _safe_zscore(onset)
    onset_delta = np.diff(onset, prepend=onset[0])

    # Low-frequency rhythmic energy (kick-ish region).
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=fs, n_fft=2048)
    low_mask = (freqs >= 20.0) & (freqs <= 180.0)
    if np.any(low_mask):
        low_env = _safe_zscore(np.mean(S[low_mask, :], axis=0))
    else:
        low_env = np.zeros_like(onset)

    # Tempogram summarized as dominant periodicity strength per frame.
    tg = librosa.feature.tempogram(onset_envelope=onset, sr=fs, hop_length=hop_length)
    if tg.ndim == 2 and tg.shape[1] > 0:
        tg_max = _safe_zscore(np.max(tg, axis=0))
    else:
        tg_max = np.zeros_like(onset)

    t = min(onset.size, onset_delta.size, low_env.size, tg_max.size)
    X = np.vstack([onset[:t], onset_delta[:t], low_env[:t], tg_max[:t]]).astype(np.float32)

    # Downsample in time if needed for pairwise DTW tractability.
    if X.shape[1] > max_frames:
        idx = np.linspace(0, X.shape[1] - 1, max_frames).astype(int)
        X = X[:, idx]

    return X


def pair_overlap_similarity(
    Xi: np.ndarray,
    Xj: np.ndarray,
    metric: str = "cosine",
) -> Tuple[float, float, float]:
    """Return (distance, overlap_fraction, stretch_ratio).

    - distance: normalized DTW cost (lower is better)
    - overlap_fraction: matched-path length / max(Ti, Tj), useful for subsequence matching
    - stretch_ratio: implied global stretch from DTW path (j over i)
    """
    try:
        D, wp = librosa.sequence.dtw(X=Xi, Y=Xj, metric=metric, subseq=True)
    except Exception:
        return float("inf"), 0.0, 1.0

    if D.size == 0:
        return float("inf"), 0.0, 1.0

    end_i, end_j = wp[-1]
    # subseq path may not start at (0,0)
    start_i, start_j = wp[0]

    path_len = max(1, len(wp))
    dist = float(D[end_i, end_j] / path_len)

    Ti = max(1, Xi.shape[1])
    Tj = max(1, Xj.shape[1])

    overlap_fraction = float(min(1.0, path_len / max(Ti, Tj)))

    delta_i = max(1, abs(int(end_i) - int(start_i)) + 1)
    delta_j = max(1, abs(int(end_j) - int(start_j)) + 1)
    stretch_ratio = float(delta_j / delta_i)

    return dist, overlap_fraction, stretch_ratio


def build_similarity_matrix(
    features: List[np.ndarray],
    distance_metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(features)
    dist = np.zeros((n, n), dtype=np.float64)
    ov = np.zeros((n, n), dtype=np.float64)
    stretch = np.ones((n, n), dtype=np.float64)

    for i in range(n):
        dist[i, i] = 0.0
        ov[i, i] = 1.0
        stretch[i, i] = 1.0
        for j in range(i + 1, n):
            d, o, s = pair_overlap_similarity(features[i], features[j], metric=distance_metric)
            dist[i, j] = dist[j, i] = d
            ov[i, j] = ov[j, i] = o
            stretch[i, j] = s
            stretch[j, i] = 1.0 / max(1e-8, s)

    finite_vals = dist[np.isfinite(dist) & (dist > 0)]
    temp = float(np.median(finite_vals)) if finite_vals.size else 1.0
    temp = max(1e-6, temp)

    # Similarity increases with lower DTW distance and stronger overlap.
    sim = np.exp(-dist / temp) * np.clip(ov, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim, dist, stretch


def _kth_positive_distance(dist_row: np.ndarray, k: int) -> float:
    vals = dist_row[np.isfinite(dist_row) & (dist_row > 0)]
    if vals.size == 0:
        return 1.0
    vals = np.sort(vals)
    idx = min(max(0, int(k) - 1), vals.size - 1)
    return float(max(1e-8, vals[idx]))


def self_tuned_knn_affinity(
    dist: np.ndarray,
    overlap: np.ndarray,
    k: int = 7,
    mutual_knn: bool = True,
) -> np.ndarray:
    """Self-tuning affinity (Zelnik-Manor style) + optional mutual-kNN sparsification."""
    n = dist.shape[0]
    k = max(2, int(k))

    sigma = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sigma[i] = _kth_positive_distance(dist[i], k=k)

    denom = np.outer(sigma, sigma) + 1e-12
    sim = np.exp(-(dist ** 2) / denom) * np.clip(overlap, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)

    # kNN graph mask
    knn_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        nn_idx = np.argsort(row)[::-1][:k]
        knn_mask[i, nn_idx] = True

    if mutual_knn:
        mask = knn_mask & knn_mask.T
    else:
        mask = knn_mask | knn_mask.T

    out = sim * mask.astype(np.float64)
    np.fill_diagonal(out, 1.0)
    return out


def sinkhorn_doubly_stochastic(sim: np.ndarray, n_iter: int = 25, eps: float = 1e-8) -> np.ndarray:
    """Approximate doubly-stochastic normalization of similarity matrix."""
    x = np.asarray(sim, dtype=np.float64).copy()
    x = np.maximum(x, eps)

    for _ in range(max(1, int(n_iter))):
        x = x / (np.sum(x, axis=1, keepdims=True) + eps)
        x = x / (np.sum(x, axis=0, keepdims=True) + eps)

    x = 0.5 * (x + x.T)
    np.fill_diagonal(x, np.maximum(1.0, np.diag(x)))
    x = x / (np.max(x) + eps)
    return x


def sharpen_similarity(
    sim_raw: np.ndarray,
    dist: np.ndarray,
    overlap: np.ndarray,
    mode: str,
    self_tune_k: int,
    mutual_knn: bool,
    sinkhorn_iters: int,
    sinkhorn_eps: float,
) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "none":
        out = sim_raw.copy()
    elif mode == "self_tuned_knn":
        out = self_tuned_knn_affinity(
            dist=dist,
            overlap=overlap,
            k=max(2, int(self_tune_k)),
            mutual_knn=bool(mutual_knn),
        )
    elif mode == "sinkhorn":
        out = sinkhorn_doubly_stochastic(
            sim=sim_raw,
            n_iter=max(1, int(sinkhorn_iters)),
            eps=max(1e-12, float(sinkhorn_eps)),
        )
    else:
        raise ValueError("Unsupported sharpen mode. Expected one of: none, self_tuned_knn, sinkhorn")

    out = np.maximum(0.0, out)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 1.0)
    return out


def fiedler_order(sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = sim.shape[0]
    deg = np.sum(sim, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    L = np.eye(n) - D_inv_sqrt @ sim @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L)
    if n >= 2:
        fiedler = eigvecs[:, 1]
    else:
        fiedler = np.zeros(n, dtype=np.float64)

    order = np.argsort(fiedler)
    return order, eigvals, eigvecs


def spectral_bin_labels(fiedler: np.ndarray, n_bins: int) -> List[str]:
    n_bins = max(2, int(n_bins))
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(fiedler, qs)

    labels: List[str] = []
    for v in fiedler:
        idx = int(np.searchsorted(edges[1:-1], v, side="right"))
        labels.append(f"cm_bin_{idx}")
    return labels


def eigen_sign_multilabels(eigvecs: np.ndarray, n_components: int = 3) -> List[List[str]]:
    n = eigvecs.shape[0]
    k = min(max(1, n_components), max(1, eigvecs.shape[1] - 1))
    tags: List[List[str]] = [[] for _ in range(n)]

    # skip eigenvector 0 (constant mode), use next k vectors
    for j in range(1, k + 1):
        v = eigvecs[:, j]
        med = float(np.median(v))
        for i in range(n):
            side = "pos" if v[i] >= med else "neg"
            tags[i].append(f"lap_e{j}_{side}")
    return tags


def maybe_save_heatmap(sim: np.ndarray, order: np.ndarray, out_png: Path, title: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib not available; skipped heatmap"

    sim_ord = sim[np.ix_(order, order)]
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(sim_ord, aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Files (Fiedler order)")
    plt.ylabel("Files (Fiedler order)")
    plt.colorbar(label="similarity")
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()
    return None


def build_nearest_neighbor_rows(
    sim: np.ndarray,
    dist: np.ndarray,
    stretch: np.ndarray,
    files: List[Path],
    top_k: int,
) -> List[Dict]:
    rows: List[Dict] = []
    n = sim.shape[0]
    k = max(1, int(top_k))

    for i in range(n):
        candidates: List[Tuple[int, float]] = []
        for j in range(n):
            if i == j:
                continue
            candidates.append((j, float(sim[i, j])))
        candidates.sort(key=lambda t: t[1], reverse=True)

        for rank, (j, s) in enumerate(candidates[:k], start=1):
            rows.append(
                {
                    "file": str(files[i]),
                    "neighbor_rank": int(rank),
                    "neighbor_file": str(files[j]),
                    "similarity": float(s),
                    "distance": float(dist[i, j]),
                    "stretch_ratio_neighbor_over_file": float(stretch[i, j]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DTW overlap contact map, Laplacian ordering, and pseudo-labels from audio files"
    )
    parser.add_argument("--folder", required=True, help="Root folder containing audio files")
    parser.add_argument("--out_dir", default="training/models/contact_map", help="Output directory")
    parser.add_argument("--max_files", type=int, default=200, help="Cap file count for pairwise computation")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--max_seconds", type=float, default=20.0)
    parser.add_argument("--max_frames", type=int, default=600)
    parser.add_argument("--distance_metric", default="cosine", help="DTW frame metric")
    parser.add_argument("--n_bins", type=int, default=4, help="Number of contact-map bins from Fiedler vector")
    parser.add_argument("--n_eigen_tags", type=int, default=3, help="How many non-trivial Laplacian eigenvectors to convert into sign-tags")
    parser.add_argument(
        "--sharpen_mode",
        choices=("none", "self_tuned_knn", "sinkhorn"),
        default="self_tuned_knn",
        help="Similarity normalization/sharpening mode before Laplacian ordering.",
    )
    parser.add_argument("--self_tune_k", type=int, default=7, help="k for self-tuned local scale / kNN graph")
    parser.add_argument(
        "--disable_mutual_knn",
        action="store_true",
        help="Use union kNN graph instead of mutual-kNN when sharpen_mode=self_tuned_knn.",
    )
    parser.add_argument("--sinkhorn_iters", type=int, default=25, help="Sinkhorn row/col normalization iterations")
    parser.add_argument("--sinkhorn_eps", type=float, default=1e-8, help="Numerical epsilon for Sinkhorn")
    parser.add_argument("--top_k_neighbors", type=int, default=10, help="How many nearest neighbors to save per file")
    parser.add_argument("--top_pairs_limit", type=int, default=500, help="How many globally top similar pairs to save")
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    files = collect_audio_files(root)
    if args.max_files and args.max_files > 0:
        files = files[: int(args.max_files)]

    if len(files) < 3:
        raise RuntimeError("Need at least 3 audio files to build a meaningful contact map")

    print(f"Collected files: {len(files)}")

    feats: List[np.ndarray] = []
    keep_files: List[Path] = []
    for i, p in enumerate(files, start=1):
        X = extract_pattern_sequence(
            p,
            sr=int(args.sr),
            max_seconds=float(args.max_seconds),
            max_frames=int(args.max_frames),
        )
        if X is None:
            continue
        feats.append(X)
        keep_files.append(p)
        if i % 50 == 0:
            print(f"feature-extracted {i}/{len(files)}")

    if len(keep_files) < 3:
        raise RuntimeError("Too few valid audio files after feature extraction")

    print(f"Building pairwise matrix for {len(keep_files)} files...")
    sim_raw, dist, stretch = build_similarity_matrix(feats, distance_metric=str(args.distance_metric))

    # Rebuild overlap proxy from raw sim and dist scale approximation for self-tuned mode.
    # Since sim_raw = exp(-dist/temp)*overlap, we recover overlap approximately in [0,1].
    finite_vals = dist[np.isfinite(dist) & (dist > 0)]
    temp = float(np.median(finite_vals)) if finite_vals.size else 1.0
    temp = max(1e-6, temp)
    overlap_est = np.clip(sim_raw / np.exp(-dist / temp), 0.0, 1.0)
    overlap_est[~np.isfinite(overlap_est)] = 0.0
    np.fill_diagonal(overlap_est, 1.0)

    sim = sharpen_similarity(
        sim_raw=sim_raw,
        dist=dist,
        overlap=overlap_est,
        mode=str(args.sharpen_mode),
        self_tune_k=max(2, int(args.self_tune_k)),
        mutual_knn=not args.disable_mutual_knn,
        sinkhorn_iters=max(1, int(args.sinkhorn_iters)),
        sinkhorn_eps=max(1e-12, float(args.sinkhorn_eps)),
    )

    order, eigvals, eigvecs = fiedler_order(sim)
    fiedler = eigvecs[:, 1] if eigvecs.shape[1] > 1 else np.zeros(len(keep_files), dtype=np.float64)

    cm_bins = spectral_bin_labels(fiedler, n_bins=int(args.n_bins))
    eigen_tags = eigen_sign_multilabels(eigvecs, n_components=int(args.n_eigen_tags))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_raw_csv = out_dir / "contact_similarity_matrix_raw.csv"
    sim_csv = out_dir / "contact_similarity_matrix.csv"
    dist_csv = out_dir / "contact_distance_matrix.csv"
    stretch_csv = out_dir / "contact_stretch_ratio_matrix.csv"
    labels_csv = out_dir / "contact_map_labels.csv"
    order_csv = out_dir / "contact_map_order.csv"
    top_pairs_csv = out_dir / "contact_top_pairs.csv"
    nearest_neighbors_csv = out_dir / "contact_nearest_neighbors.csv"
    summary_json = out_dir / "contact_map_summary.json"
    heatmap_png = out_dir / "contact_similarity_heatmap.png"
    heatmap_raw_png = out_dir / "contact_similarity_heatmap_raw.png"

    names = [str(p) for p in keep_files]
    pd.DataFrame(sim_raw, index=names, columns=names).to_csv(sim_raw_csv)
    pd.DataFrame(sim, index=names, columns=names).to_csv(sim_csv)
    pd.DataFrame(dist, index=names, columns=names).to_csv(dist_csv)
    pd.DataFrame(stretch, index=names, columns=names).to_csv(stretch_csv)

    labels_rows: List[Dict] = []
    for i, p in enumerate(keep_files):
        tags = [cm_bins[i]] + eigen_tags[i]
        labels_rows.append(
            {
                "file": str(p),
                "contact_bin": cm_bins[i],
                "contact_tags": ",".join(sorted(set(tags))),
                "fiedler_value": float(fiedler[i]),
            }
        )
    pd.DataFrame(labels_rows).to_csv(labels_csv, index=False)

    order_rows = [{"rank": int(k), "file": str(keep_files[idx]), "index": int(idx)} for k, idx in enumerate(order)]
    pd.DataFrame(order_rows).to_csv(order_csv, index=False)

    hm_warn = maybe_save_heatmap(
        sim=sim,
        order=order,
        out_png=heatmap_png,
        title="DTW overlap contact map (Fiedler-ordered)",
    )
    hm_raw_warn = None
    if str(args.sharpen_mode).lower() != "none":
        hm_raw_warn = maybe_save_heatmap(
            sim=sim_raw,
            order=order,
            out_png=heatmap_raw_png,
            title="DTW overlap contact map (raw)",
        )

    # top overlaps (excluding self)
    top_pairs: List[Dict] = []
    n = sim.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            top_pairs.append(
                {
                    "file_i": str(keep_files[i]),
                    "file_j": str(keep_files[j]),
                    "similarity": float(sim[i, j]),
                    "distance": float(dist[i, j]),
                    "stretch_ratio_j_over_i": float(stretch[i, j]),
                }
            )
    top_pairs = sorted(top_pairs, key=lambda r: r["similarity"], reverse=True)

    top_pairs_limit = max(1, int(args.top_pairs_limit))
    top_pairs_df = pd.DataFrame(top_pairs[:top_pairs_limit])
    top_pairs_df.to_csv(top_pairs_csv, index=False)

    nn_rows = build_nearest_neighbor_rows(
        sim=sim,
        dist=dist,
        stretch=stretch,
        files=keep_files,
        top_k=max(1, int(args.top_k_neighbors)),
    )
    pd.DataFrame(nn_rows).to_csv(nearest_neighbors_csv, index=False)

    summary = {
        "n_files": int(len(keep_files)),
        "params": {
            "sr": int(args.sr),
            "max_seconds": float(args.max_seconds),
            "max_frames": int(args.max_frames),
            "distance_metric": str(args.distance_metric),
            "sharpen_mode": str(args.sharpen_mode),
            "self_tune_k": int(args.self_tune_k),
            "mutual_knn": bool(not args.disable_mutual_knn),
            "sinkhorn_iters": int(args.sinkhorn_iters),
            "sinkhorn_eps": float(args.sinkhorn_eps),
            "n_bins": int(args.n_bins),
            "n_eigen_tags": int(args.n_eigen_tags),
        },
        "artifacts": {
            "similarity_raw_csv": str(sim_raw_csv),
            "similarity_csv": str(sim_csv),
            "distance_csv": str(dist_csv),
            "stretch_ratio_csv": str(stretch_csv),
            "labels_csv": str(labels_csv),
            "order_csv": str(order_csv),
            "top_pairs_csv": str(top_pairs_csv),
            "nearest_neighbors_csv": str(nearest_neighbors_csv),
            "heatmap_png": str(heatmap_png) if hm_warn is None else None,
            "heatmap_raw_png": str(heatmap_raw_png) if hm_raw_warn is None and str(args.sharpen_mode).lower() != "none" else None,
        },
        "laplacian_eigenvalues": [float(v) for v in eigvals[: min(10, len(eigvals))]],
        "top_overlaps": top_pairs[: min(50, len(top_pairs))],
    }
    if hm_warn:
        summary["warnings"] = [hm_warn]
    if hm_raw_warn:
        summary.setdefault("warnings", []).append(hm_raw_warn)

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved labels: {labels_csv}")
    print(f"Saved top pairs: {top_pairs_csv}")
    print(f"Saved nearest neighbors: {nearest_neighbors_csv}")
    print(f"Saved summary: {summary_json}")
    if hm_warn:
        print(hm_warn)


if __name__ == "__main__":
    main()
