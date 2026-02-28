# Training from a Drum-Break Folder

This project now includes a weakly supervised trainer:

- Script: `train_from_drumbreaks.py`
- Input: folder containing drum-break audio files (`.wav`, `.mp3`, `.flac`, `.aiff`, `.m4a`)
- Outputs (default):
  - `training/models/loop_ranker.json`
  - `training/models/features.json`
  - `training/models/training_rows.csv`

## Example

`python train_from_drumbreaks.py --folder "C:\Users\dunnm\Downloads\Drum Breaks"`

By default, training candidates are generated as **4-bar and 8-bar loops** (`--bars "4,8"`).

## CDJ-style BPM augmentation (tempo fader simulation)

You can augment drum-break training by creating tempo-shifted variants at ±10% in 1% increments:

`python train_from_drumbreaks.py --folder "C:\Users\dunnm\Downloads\Drum Breaks" --bars "4,8" --tempo_augment --tempo_min_pct -10 --tempo_max_pct 10 --tempo_step_pct 1`

This simulates tempo fader movement and helps loop ranking stay robust when tempo changes.

## Fine-tune for real music (recommended)

Mix your drum-break positives with hard negatives mined from full songs:

`python train_from_drumbreaks.py --folder "C:\Users\dunnm\Downloads\Drum Breaks" --full_music_folder "C:\path\to\Full Songs" --neg_per_track 24`

What this does:

- Uses your drum-break folder as positives.
- Mines low drum-break-likelihood segments from full songs as hard negatives.
- Trains one mixed XGBoost model with sample weighting.

## Notes

- This trainer uses weak labels derived from rhythmic/percussive features.
- For stronger results on full songs, include hard negatives (non-break sections) and some complete mixes.
- The inference pipeline in `sample_pack_pipeline_ml.py` expects model/features artifacts under `training/models/` by default in `batch_export_loops.py`.

## Compare breaks-only vs mixed fine-tuned model

Use `evaluate_loop_models.py` on a validation folder:

`python evaluate_loop_models.py --folder "C:\path\to\Validation Music" --bars "4,8" --model_a "training/models/loop_ranker_breaks_only.json" --features_a "training/models/features_breaks_only.json" --model_b "training/models/loop_ranker_mixed.json" --features_b "training/models/features_mixed.json" --top_k 5`

Outputs:

- `training/models/eval_rows.csv`
- `training/models/eval_metrics.json`

Main metrics include:

- `mean_track_topk_target`
- `uplift_vs_random_pct`
- `mean_track_ndcg_at_k`
- `delta_topk_target_b_minus_a` (comparison)

## One-command orchestrator (train + compare)

Use `orchestrate_training_and_eval.py` to run everything in one go:

`python orchestrate_training_and_eval.py --drumbreak_folder "C:\Users\dunnm\Downloads\Drum Breaks" --validation_folder "C:\path\to\Validation Music" --full_music_folder "C:\path\to\Full Songs" --bars "4,8" --tempo_augment --tempo_min_pct -10 --tempo_max_pct 10 --tempo_step_pct 1 --neg_per_track 24 --top_k 5`

By default, artifacts are written to `training/runs/<timestamp>/` including:

- `models/loop_ranker_breaks_only.json`
- `models/loop_ranker_mixed.json` (if full-music folder is provided)
- `rows/training_rows_breaks_only.csv`
- `rows/training_rows_mixed.csv` (if mixed training runs)
- `eval/eval_rows.csv`
- `eval/eval_metrics.json`
- `run_summary.json`
