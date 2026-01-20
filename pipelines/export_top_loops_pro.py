import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import pandas as pd

def normalize_audio(y, target=-14.0):
    rms = np.sqrt(np.mean(y**2))
    gain = 10**((target - 20*np.log10(rms + 1e-8))/20)
    return y * gain

def export_top_loops(
    df,
    audio_file,
    outdir,
    top_k=5,
    sr=44100,
    export_mp3=False,
    *,
    min_score: float | None = None,
    min_score_prob: float | None = None,
    min_rel_score: float | None = None,
):
    y, _ = librosa.load(audio_file, sr=sr, mono=True)
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    metas = []

    # Ensure best->worst ordering if a score column exists.
    if isinstance(df, pd.DataFrame) and "score" in df.columns:
        df = df.sort_values("score", ascending=False)

    # Apply optional cutoffs (works even if caller didn't filter).
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        if min_score is not None and "score" in df.columns:
            df = df[df["score"] >= float(min_score)]
        if min_score_prob is not None and "score_prob" in df.columns:
            df = df[df["score_prob"] >= float(min_score_prob)]
        if min_rel_score is not None and "score" in df.columns and len(df) > 0:
            best = float(df["score"].iloc[0])
            df = df[df["score"] >= best + float(min_rel_score)]

    if export_mp3:
        try:
            from pydub import AudioSegment  # type: ignore
        except Exception as e:
            raise ImportError(
                "MP3 export requested but 'pydub' is not installed. Install pydub (and ffmpeg) or set export_mp3=False."
            ) from e

    for i, row in df.head(top_k).iterrows():
        seg = y[int(row["start_time"]*sr):int(row["end_time"]*sr)]
        seg = normalize_audio(seg)
        wav_path = outdir / f"{row['track_id']}_loop{i}.wav"
        sf.write(wav_path, seg, sr)
        if export_mp3:
            seg16 = (seg * 32767).astype(np.int16)
            audio_seg = AudioSegment(
                seg16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
            audio_seg.export(str(wav_path.with_suffix(".mp3")), format="mp3", bitrate="192k")
        metas.append(row.to_dict())
    pd.DataFrame(metas).to_csv(outdir/"loops_metadata.csv", index=False)
    return metas
