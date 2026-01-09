import numpy as np
import librosa
import openl3

def extract_openl3_embedding(y, sr, embedding_size=512, input_repr="mel256"):
    emb, _ = openl3.get_audio_embedding(
        y,
        sr,
        input_repr=input_repr,
        embedding_size=embedding_size,
        content_type="music"
    )
    return np.mean(emb, axis=0)

def extract_full_features(y, sr, start_time, end_time):
    start_s = int(start_time * sr)
    end_s = int(end_time * sr)
    segment = y[start_s:end_s]
    if len(segment) == 0:
        return {}
    feats = {}
    # Energy / RMS
    rms = librosa.feature.rms(y=segment)[0]
    feats["rms_mean"] = float(np.mean(rms))
    # Rhythm / onset strength
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    feats["onset_mean"] = float(np.mean(onset_env))
    # OpenL3 embedding
    emb = extract_openl3_embedding(segment, sr)
    for i, val in enumerate(emb):
        feats[f"openl3_{i}"] = float(val)
    return feats
