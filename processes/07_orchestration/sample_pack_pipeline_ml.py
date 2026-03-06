import json
import xgboost as xgb
import pandas as pd
import librosa
import numpy as np
from training.feature_extraction import extract_full_features
from training.dataset_builder import generate_bar_aligned_candidates, compute_candidate_weight


def _normalize_01(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)


class LoopPipelineML:
    def __init__(
        self,
        model_path,
        feature_list_path,
        adaptive_4_vs_8: bool = False,
        adaptive_margin: float = 0.03,
        use_composite_rerank: bool = True,
        weight_model_score: float = 0.75,
        weight_periodicity: float = 0.15,
        weight_drum_alignment: float = 0.10,
    ):
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        with open(feature_list_path) as f:
            self.features = json.load(f)

        self.adaptive_4_vs_8 = bool(adaptive_4_vs_8)
        self.adaptive_margin = float(adaptive_margin)
        self.use_composite_rerank = bool(use_composite_rerank)

        w_model = max(0.0, float(weight_model_score))
        w_periodicity = max(0.0, float(weight_periodicity))
        w_drum = max(0.0, float(weight_drum_alignment))
        w_sum = w_model + w_periodicity + w_drum
        if w_sum <= 1e-12:
            w_model, w_periodicity, w_drum = 1.0, 0.0, 0.0
            w_sum = 1.0

        self.weight_model_score = float(w_model / w_sum)
        self.weight_periodicity = float(w_periodicity / w_sum)
        self.weight_drum_alignment = float(w_drum / w_sum)

    def process_track(self, audio_path, timestamps=None):
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        bpm, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        cands = generate_bar_aligned_candidates(
            audio_path.stem,
            beats,
            bpm,
            [4, 8],
            adaptive_4_vs_8=self.adaptive_4_vs_8,
            adaptive_margin=self.adaptive_margin,
        )
        data = []
        for cand in cands:
            feats = extract_full_features(y, sr, cand.start_time, cand.end_time)
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
        if df.empty:
            return df

        for f in self.features:
            if f not in df.columns:
                df[f] = 0.0

        for c in ("periodicity_score", "drum_alignment_score"):
            if c not in df.columns:
                df[c] = 0.0

        df["score"] = self.model.predict(df[self.features])

        if not self.use_composite_rerank:
            return df.sort_values("score", ascending=False)

        model_component = _normalize_01(df["score"].to_numpy(dtype=float))
        periodicity_component = np.clip(df["periodicity_score"].to_numpy(dtype=float), 0.0, 1.0)
        drum_component = np.clip(df["drum_alignment_score"].to_numpy(dtype=float), 0.0, 1.0)

        df["beatmatch_composite_score"] = (
            self.weight_model_score * model_component
            + self.weight_periodicity * periodicity_component
            + self.weight_drum_alignment * drum_component
        )
        return df.sort_values("beatmatch_composite_score", ascending=False)
