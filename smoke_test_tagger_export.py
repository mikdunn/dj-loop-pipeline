import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from export_top_loops_pro import export_top_loops

SR = 44100
DURATION = 2.0
FREQ = 440.0

# Create test audio directory and file
audio_dir = Path("data/test_audio")
export_dir = Path("data/test_export")
audio_dir.mkdir(parents=True, exist_ok=True)
export_dir.mkdir(parents=True, exist_ok=True)

# Generate simple sine tone
samples = np.arange(int(SR * DURATION))
y = 0.2 * np.sin(2 * np.pi * FREQ * (samples / SR))

wav_path = audio_dir / "test_tone.wav"
sf.write(wav_path, y, SR)

# Create a single loop candidate from 0.0 to 1.0 seconds
row = {
    "track_id": "test_tone",
    "start_time": 0.0,
    "end_time": 1.0,
    "bars": 1,
    "weight": 1.0,
}

df = pd.DataFrame([row])

# Run export with tagging enabled (will only use local tags; Spotify optional)
metas = export_top_loops(df, wav_path, export_dir / "test_tone", top_k=1, sr=SR, export_mp3=False, add_tags=True)

print("Exported", len(metas), "loop(s)")
print("Metadata CSV:", (export_dir / "test_tone" / "loops_metadata.csv").as_posix())
