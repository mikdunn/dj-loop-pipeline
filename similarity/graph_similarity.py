from typing import List, Tuple

import librosa
import numpy as np


def pair_distance_overlap(Xi: np.ndarray, Xj: np.ndarray) -> Tuple[float, float]:
    try:
        D, wp = librosa.sequence.dtw(X=Xi, Y=Xj, metric="cosine", subseq=True)
    except Exception:
        return float("inf"), 0.0

    if D.size == 0:
        return float("inf"), 0.0

    end_i, end_j = wp[-1]
    end_i = int(np.clip(end_i, 0, D.shape[0] - 1))
    end_j = int(np.clip(end_j, 0, D.shape[1] - 1))
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


def _self_tuned_knn(dist: np.ndarray, ov: np.ndarray, k: int = 7, mutual: bool = True) -> np.ndarray:
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


def _sinkhorn(sim: np.ndarray, iters: int = 25, eps: float = 1e-8) -> np.ndarray:
    x = np.maximum(sim.copy(), eps)
    for _ in range(max(1, int(iters))):
        x = x / (np.sum(x, axis=1, keepdims=True) + eps)
        x = x / (np.sum(x, axis=0, keepdims=True) + eps)
    x = 0.5 * (x + x.T)
    x = x / (np.max(x) + eps)
    np.fill_diagonal(x, 1.0)
    return x


def sharpen_similarity(sim_raw: np.ndarray, dist: np.ndarray, ov: np.ndarray, mode: str) -> np.ndarray:
    m = mode.strip().lower()
    if m == "none":
        return sim_raw.copy()
    if m == "self_tuned_knn":
        return _self_tuned_knn(dist, ov, k=7, mutual=True)
    if m == "sinkhorn":
        return _sinkhorn(sim_raw, iters=25, eps=1e-8)
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
