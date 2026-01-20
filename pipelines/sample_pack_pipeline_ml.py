import json
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import pandas as pd

from training.dataset_builder import compute_candidate_weight, generate_bar_aligned_candidates
from training.feature_extraction import build_track_feature_cache, extract_full_features


class _XgbRegressorWrapper:
    def __init__(self, model_path: str, feature_list_path: str):
        try:
            import xgboost as xgb
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "XGBoost is required for .json models. Install 'xgboost' or use a .pt torch model."
            ) from e

        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        with open(feature_list_path, encoding="utf-8") as f:
            self.features = json.load(f)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features])


class _TorchRegressorWrapper:
    def __init__(self, model_path: str, feature_list_path: Optional[str] = None):
        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for .pt/.pth models. Install 'torch'."
            ) from e

        from training.torch_ranker import build_mlp, load_checkpoint

        state_dict, features_from_ckpt, scaler, meta = load_checkpoint(model_path)

        features: List[str]
        if feature_list_path:
            with open(feature_list_path, encoding="utf-8") as f:
                features = json.load(f)
        elif features_from_ckpt:
            features = list(features_from_ckpt)
        else:
            raise ValueError("Torch model requires a features list (either in checkpoint or features.json)")

        hidden_sizes = meta.get("hidden_sizes") or [256, 128]
        dropout = float(meta.get("dropout", 0.10))

        self.features = features
        self.scaler = scaler
        self.torch = torch

        self.model = build_mlp(len(self.features), hidden_sizes=hidden_sizes, dropout=dropout)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].to_numpy(dtype=np.float32, copy=True)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scaler is not None:
            x = self.scaler.transform(x)

        with self.torch.no_grad():
            xb = self.torch.from_numpy(x)
            pred = self.model(xb).view(-1)
            return pred.cpu().numpy()

class LoopPipelineML:
    def __init__(self, model_path: str, feature_list_path: Optional[str] = None):
        """ML loop pipeline.

        Backwards compatible with the original signature:
            LoopPipelineML("training/models/loop_ranker.json", "training/models/features.json")

        Additionally supports PyTorch models:
            LoopPipelineML("training/models/loop_ranker.pt", "training/models/features.json")
        """

        suffix = Path(model_path).suffix.lower()
        if suffix == ".json":
            if not feature_list_path:
                raise ValueError("feature_list_path is required for XGBoost models")
            self._model = _XgbRegressorWrapper(model_path, feature_list_path)
        elif suffix in {".pt", ".pth"}:
            self._model = _TorchRegressorWrapper(model_path, feature_list_path)
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def process_track(
        self,
        audio_path,
        timestamps=None,
        bars_options=None,
        *,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        min_score_prob: Optional[float] = None,
        min_rel_score: Optional[float] = None,
    ):
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        cache = build_track_feature_cache(y, sr)
        bpm, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")

        # Fallback when beat tracking fails
        bpm = float(bpm)
        if not np.isfinite(bpm) or bpm <= 0:
            bpm = 120.0
        if beats is None or len(beats) < (max(bars_options or [1]) * 4 + 1):
            dur_s = float(len(y)) / float(sr)
            spb = 60.0 / bpm
            beats = np.arange(0.0, dur_s + 1e-6, spb, dtype=np.float32)
        if bars_options is None:
            bars_options = [1, 2, 4, 8, 16]
        cands = generate_bar_aligned_candidates(audio_path.stem, beats, bpm, list(bars_options))
        data = []
        for cand in cands:
            feats = extract_full_features(y, sr, cand.start_time, cand.end_time, cache=cache)
            if feats:
                row = feats.copy()
                row.update({
                    "track_id": audio_path.stem,
                    "start_time": cand.start_time,
                    "end_time": cand.end_time,
                    "bars": cand.bars,
                    "weight": compute_candidate_weight(cand.center_time, timestamps or [], bpm)
                })
                data.append(row)
        df = pd.DataFrame(data)
        df["score"] = self._model.predict(df)

        # Rank best -> worst
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        # Per-track probability score (useful as a cutoff that is less sensitive to absolute scale)
        if len(df) > 0:
            s = df["score"].to_numpy(dtype=np.float32)
            s = s - float(np.max(s))
            p = np.exp(s)
            p = p / (float(np.sum(p)) + 1e-8)
            df["score_prob"] = p
            df["score_rank"] = np.arange(1, len(df) + 1)
        else:
            df["score_prob"] = np.array([], dtype=np.float32)
            df["score_rank"] = np.array([], dtype=np.int32)

        # Apply quality cutoffs
        if min_score is not None:
            df = df[df["score"] >= float(min_score)]
        if min_score_prob is not None and "score_prob" in df.columns:
            df = df[df["score_prob"] >= float(min_score_prob)]
        if min_rel_score is not None and len(df) > 0:
            best = float(df["score"].iloc[0])
            df = df[df["score"] >= best + float(min_rel_score)]

        if top_k is not None:
            df = df.head(int(top_k))

        return df.reset_index(drop=True)
