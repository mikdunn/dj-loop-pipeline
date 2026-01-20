from __future__ import annotations

from pathlib import Path

from pipelines.drum_breaks import export_top_drum_breaks_from_ranked_candidates
from pipelines.sample_pack_pipeline_ml import LoopPipelineML

# Choose your model backend (.json for XGBoost, .pt for PyTorch)
MODEL_PATH = "training/models/loop_ranker.pt"
FEATURES_PATH = "training/models/features.json"  # optional for .pt but recommended

RAW_AUDIO_DIR = Path("data/raw_audio")
OUTPUT_DIR = Path("data/exported_breaks")
TOP_K_BREAKS = 3
SLICE_HITS = True

# Cutoffs (pick ONE style, or combine):
MIN_SCORE = None  # absolute model score threshold
MIN_SCORE_PROB = 0.05  # per-track softmax probability threshold (scale-invariant)
MIN_REL_SCORE = None  # keep only scores >= best + MIN_REL_SCORE


def main() -> int:
    pipeline = LoopPipelineML(MODEL_PATH, FEATURES_PATH)

    for file in RAW_AUDIO_DIR.glob("*.wav"):
        df_ranked = pipeline.process_track(
            file,
            top_k=None,
            min_score=MIN_SCORE,
            min_score_prob=MIN_SCORE_PROB,
            min_rel_score=MIN_REL_SCORE,
        )
        if len(df_ranked) == 0:
            continue
        export_top_drum_breaks_from_ranked_candidates(
            df_ranked=df_ranked,
            audio_file=file,
            outdir=OUTPUT_DIR,
            top_k=TOP_K_BREAKS,
            slice_hits=SLICE_HITS,
        )

    print(f"Done. Exported drum breaks to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
