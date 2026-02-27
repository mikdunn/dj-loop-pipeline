import json
import xgboost as xgb
import pandas as pd
import librosa
from training.feature_extraction import extract_full_features
from training.dataset_builder import generate_bar_aligned_candidates, compute_candidate_weight

class LoopPipelineML:
    def __init__(self, model_path, feature_list_path):
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        with open(feature_list_path) as f:
            self.features = json.load(f)

    def process_track(self, audio_path, timestamps=None):
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        bpm, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        cands = generate_bar_aligned_candidates(audio_path.stem, beats, bpm, [4,8])
        data = []
        for cand in cands:
            feats = extract_full_features(y,sr,cand.start_time,cand.end_time)
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
        df["score"] = self.model.predict(df[self.features])
        return df.sort_values("score",ascending=False)
