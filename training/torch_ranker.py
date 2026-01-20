from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StandardScaler:
    """Tiny, dependency-free standard scaler.

    Stored as float32 for compactness; transform outputs float32.
    """

    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={x.shape}")
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        self.mean_ = mean.astype(np.float32)
        self.scale_ = std.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler is not fitted")
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / self.scale_

    def to_dict(self) -> Dict[str, Any]:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler is not fitted")
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StandardScaler":
        mean = np.asarray(d["mean"], dtype=np.float32)
        scale = np.asarray(d["scale"], dtype=np.float32)
        return cls(mean_=mean, scale_=scale)


def _torch_import():
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the torch model. Install 'torch' first."
        ) from e
    return torch, nn


def build_mlp(
    in_features: int,
    hidden_sizes: Sequence[int] = (256, 128),
    dropout: float = 0.10,
):
    torch, nn = _torch_import()

    layers: List[Any] = []
    prev = int(in_features)
    for h in hidden_sizes:
        h = int(h)
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def save_checkpoint(
    path: str,
    *,
    model,
    features: List[str],
    scaler: Optional[StandardScaler] = None,
    meta: Optional[Dict[str, Any]] = None,
):
    torch, _ = _torch_import()

    ckpt: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "features": list(features),
        "meta": meta or {},
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.to_dict()

    torch.save(ckpt, path)


def load_checkpoint(path: str):
    torch, _ = _torch_import()

    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError("Not a valid checkpoint: missing state_dict")
    features = ckpt.get("features")
    scaler_dict = ckpt.get("scaler")
    scaler = StandardScaler.from_dict(scaler_dict) if scaler_dict else None
    meta = ckpt.get("meta") or {}
    return ckpt["state_dict"], features, scaler, meta
