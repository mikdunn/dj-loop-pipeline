from typing import Dict

import numpy as np


def _fiedler_order(sim: np.ndarray) -> np.ndarray:
    n = sim.shape[0]
    deg = np.sum(sim, axis=1)
    D = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    L = np.eye(n) - D @ sim @ D
    _, eigvecs = np.linalg.eigh(L)
    f = eigvecs[:, 1] if n >= 2 else np.zeros(n)
    return np.argsort(f)


def _block_contrast(sim: np.ndarray, order: np.ndarray, frac: float = 0.12) -> float:
    S = sim[np.ix_(order, order)]
    n = S.shape[0]
    b = max(2, int(n * frac))
    if n < 4:
        return 0.0

    diag_vals = []
    for i in range(0, n - b + 1, b):
        diag_vals.append(float(np.mean(S[i : i + b, i : i + b])))
    diag_mean = float(np.mean(diag_vals)) if diag_vals else 0.0

    off = S.copy()
    for i in range(0, n - b + 1, b):
        off[i : i + b, i : i + b] = np.nan
    off_mean = float(np.nanmean(off)) if np.isfinite(np.nanmean(off)) else 0.0
    return float(diag_mean - off_mean)


def _spectral_gap(sim: np.ndarray) -> float:
    deg = np.sum(sim, axis=1)
    D = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    L = np.eye(sim.shape[0]) - D @ sim @ D
    ew = np.linalg.eigvalsh(L)
    return float(ew[1] - ew[0]) if ew.size > 1 else 0.0


def compute_graph_quality(sim: np.ndarray) -> Dict[str, float]:
    order = _fiedler_order(sim)
    tri = np.triu_indices_from(sim, k=1)
    mean_similarity = float(np.mean(sim[tri])) if tri[0].size else 0.0
    return {
        "block_contrast": _block_contrast(sim, order),
        "spectral_gap": _spectral_gap(sim),
        "mean_similarity": mean_similarity,
    }
