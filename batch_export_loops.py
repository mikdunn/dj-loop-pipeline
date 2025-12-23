from pathlib import Path
from pipelines.sample_pack_pipeline_ml import LoopPipelineML
from pipelines.export_top_loops_pro import export_top_loops

pipeline = LoopPipelineML("training/models/loop_ranker.json","training/models/features.json")

RAW_AUDIO_DIR = Path("data/raw_audio")
OUTPUT_DIR = Path("data/exported_loops")
TOP_K = 5

for file in RAW_AUDIO_DIR.glob("*.wav"):
    df_ranked = pipeline.process_track(file)
    export_top_loops(df_ranked, file, OUTPUT_DIR/file.stem, top_k=TOP_K, export_mp3=True)
