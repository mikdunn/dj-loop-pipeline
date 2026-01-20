# DJ Loop Extraction Pipeline 

**End-to-End ML-Powered Loop Finder and Exporter**

Automatically generates DJ/DAW-ready loops from audio tracks using a combination of hand-crafted features, PyTorch/torchaudio embeddings, and ML scoring (XGBoost or PyTorch).

---

## Features

- **Bar-aligned candidate generation**
  - Automatically slices tracks into candidate loops based on beats and bars.
- **Full feature extraction**
  - Energy, RMS, onset strength, and rhythm analysis.
  - **PyTorch/torchaudio log-mel embeddings** (no network downloads) for rich representations.
  - (Optional/legacy) OpenL3 embeddings if you install a compatible OpenL3 stack.
- **ML Scoring**
  - Uses an XGBoost model to predict loop quality.
- **Top-K Candidate Ranking**
  - Selects the best loops based on ML predictions.
- **Loop Export**
  - Normalized WAV slices
  - Optional MP3 previews
  - Fade-in/out applied for seamless playback
  - Metadata CSV per track
- **Batch Processing**
  - Process all WAV files in a directory automatically
- **Ranking Metrics**
  - Precision@K, Recall@K for evaluation

---

## System Architecture

```mermaid
flowchart TD
    A[Raw Audio WAVs] --> B[Candidate Generation]
    B --> C[Feature Extraction]
    C --> D[ML Scoring (XGBoost)]
    D --> E[Candidate Ranking]
    E --> F[Loop Export (WAV + MP3 + Metadata)]
    F --> G[Aggregated Metadata CSV]

```

---

## PyTorch model support

The scoring step can use either:

- **XGBoost** (default / legacy): model file `training/models/loop_ranker.json`
- **PyTorch MLP** (new): model file `training/models/loop_ranker.pt`

The pipeline auto-detects which backend to use based on the model filename extension.

### Train a PyTorch model

This trains a simple MLP regressor to predict `weight` from your extracted features (including `bars` + OpenL3 embedding dims) and writes:

- `training/models/loop_ranker.pt` (checkpoint; includes scaler + feature list)
- `training/models/features.json` (kept for compatibility)

Run:

- `python training/train_loop_ranker_torch.py --dataset loop_training_dataset.csv`

### Use a PyTorch model for inference

Point the pipeline at the `.pt` file:

- `LoopPipelineML("training/models/loop_ranker.pt", "training/models/features.json")`

If you point at a `.json` file, the pipeline will use XGBoost.

### Cutoff scores (skip low-quality loops)

`LoopPipelineML.process_track(...)` returns loops sorted best → worst, and also adds:

- `score`: the raw model score
- `score_prob`: a per-track softmax probability over candidates (useful for a **scale-invariant cutoff**)
- `score_rank`: 1 = best candidate

You can apply cutoffs directly during scoring:

- `min_score`: absolute threshold on `score`
- `min_score_prob`: threshold on `score_prob` (recommended; works well even if the model's raw score scale changes)
- `min_rel_score`: keep only candidates with `score >= best_score + min_rel_score`

Example:

- `pipeline.process_track(path, min_score_prob=0.05)`

---

## Finding "clean" loop cut points (quiet/no drums)

Yes—this is possible, and this repo now exposes useful proxies in feature extraction.

When DJs say "cut where the drums drop out" (or where drums are quiet), a practical signal is:

- **Percussive energy** (via HPSS percussive component)
- **Percussive onset strength** (transient density)

For each candidate loop, `training.feature_extraction.extract_full_features(...)` now computes:

- `perc_to_total_rms`: lower means fewer/softer drums in the loop overall
- `boundary_perc_onset_start`, `boundary_perc_onset_end`: lower means fewer transients right at the cut edges
- `boundary_quiet_score`: a single scalar combining boundary percussive RMS + onset strength

### Recommended heuristic

To pick 2/4/8/16-bar loops with cleaner edges:

1. Generate bar-aligned candidates (already done)
2. Rank candidates by your ML score **and** prefer small `boundary_quiet_score`
3. Optionally require `boundary_quiet_score` below a threshold to avoid chopping drum hits

This works especially well when beat tracking is correct and your loops align to bar boundaries.

---

## Drum breaks ➜ export loops + one-shots

Yes—once a drum break (or drum-heavy loop) is identified, you can:

1) export the **loop** (the break)
2) export an **isolated percussive stem** of that break (HPSS percussive component)
3) detect onsets in the percussive stem and slice **individual hits** (“one-shots”)

This repo now includes:

- `pipelines/drum_breaks.py` — utilities to isolate percussive audio and slice hits
- `pipelines/batch_export_drum_breaks.py` — batch runner

### Output structure

Exports to `data/exported_breaks/<track_id>/break_XX/`:

- `drum_break.wav` (the loop)
- `drum_break_perc.wav` (percussive-only approximation)
- `hits/hit_0000.wav`, ... (sliced one-shots from the full mix)
- `hits/hit_0000_perc.wav`, ... (percussive-only one-shots for cleaner drum hits)
- `hits/hits_metadata.csv` (labels + features per hit)

Also writes a CSV per input track:

- `data/exported_breaks/<track>_drum_breaks.csv`

### Notes

- Hit slicing uses onset detection on the percussive signal. It’s very effective for kicks/snares/hats,
  but it’s not perfect for every genre (e.g., washed-out cymbals).
- Hit exports include heuristic labels to help build a **drum kit** or **synth sound bank**:
  - `bank_type`: `drum_kit` or `synth`
  - `class_label`: e.g. `kick`, `snare`, `hat`, `bass`, `lead`, etc.
- If you want true drum-stem isolation, the next step is integrating a separator model (e.g., Demucs),
  which is heavier but more accurate.

---

## Quiet/drumless loops ➜ slice on note hits

If you want to find **drumless / quiet-drum** regions to cut 2/4/8/16-bar loops, and then slice those loops
into musical chunks based on **note hits**:

1) The pipeline already computes drum-activity features per candidate:
  - `perc_to_total_rms` (lower = more drumless)
  - `boundary_quiet_score` (lower = cleaner cut edges)

2) This repo includes a batch script that:
  - selects top candidates that satisfy “quiet drums” thresholds
  - exports the loop
  - slices the loop into **8 and 16** slices; and for **16-bar loops**, also **32** slices
  - slice boundaries are snapped to the nearest **harmonic onset** (“note hit”) when possible

Run:

- `python pipelines/batch_export_quiet_loops_and_slices.py`

Outputs:

- `data/exported_quiet_loops/<track_id>/loop_XX.wav`
- `data/exported_quiet_loops/<track_id>/loop_XX/slices_08/*.wav` (+ CSV)
- `data/exported_quiet_loops/<track_id>/loop_XX/slices_16/*.wav` (+ CSV)
- `data/exported_quiet_loops/<track_id>/loop_XX/slices_32/*.wav` (+ CSV) (16-bar loops only)

---

## Improving training data & model quality

The current training setup uses a weak label (`weight`) derived from how close a candidate’s center is to a set of “good loop” timestamps.
You can significantly improve performance by improving **labels**, **negatives**, and **evaluation**.

### High-impact upgrades (recommended)

- **Collect better labels**
  - Instead of only timestamps, store explicit “good loop” boundaries: `(start_time, end_time)`.
  - Add a few **hard negatives** per track (loops that *sound* bad: off-beat, awkward boundary cuts, bad energy, etc.).

- **Train a ranking objective**
  - Your real task is “pick the best loop among candidates”. A pairwise/listwise rank loss (e.g., hinge or softmax ranking) often beats pure regression on `weight`.
  - You can keep the current model as-is, but add a second-stage reranker that optimizes ranking metrics (Precision@K).

#### Listwise per-track ranking trainer (implemented)

This repo includes a listwise trainer that groups candidates by `track_id` and learns to rank them within each track:

- `python training/train_loop_ranker_torch_rank.py --dataset loop_training_dataset.csv`

It writes:

- `training/models/loop_ranker_rank.pt`

This model is especially compatible with using `min_score_prob` cutoffs at inference time.

##### MLflow logging (optional)

If you want experiment tracking (params, metrics per epoch, and the saved model artifact), install MLflow:

- `pip install -r requirements_optional_mlflow.txt`

Then run training with logging enabled:

- `python training/train_loop_ranker_torch_rank.py --dataset loop_training_dataset.csv --mlflow`

You can also configure:

- `--mlflow-uri` (tracking server URI)
- `--mlflow-experiment` (experiment name)
- `--mlflow-run-name` (run name)

Logged ranking metrics include per-track **MRR** and **Precision@K** where “correct” means the model ranked the max-`weight` loop in the top K for that track.

##### Quick start (generate a dataset CSV)

The ranking trainer expects a dataset CSV like `loop_training_dataset.csv`. You can build one from local WAVs + a timestamps JSON:

- `python training/build_training_dataset_cli.py --audio-dir data/raw_audio --timestamps-json timestamps_example.json --out loop_training_dataset.csv`

Then train:

- `python training/train_loop_ranker_torch_rank.py --dataset loop_training_dataset.csv`

- **Calibrate for cut quality**
  - Use the boundary features (`boundary_quiet_score`, etc.) either as:
    - hard filters (avoid cuts on transients)
    - or in the final score: `final = ml_score - α * boundary_quiet_score`

- **Evaluate properly**
  - Split by **track_id** (not random rows) to avoid leakage.
  - Track metrics like Precision@K / Recall@K per track.

### Speed improvements (implemented)

- **Per-track feature caching (huge win)**
  - Candidate extraction used to recompute HPSS + log-mel embedding for every candidate slice.
  - The repo now computes those transforms **once per track** and reuses them for all candidates.
  - This speeds up both dataset building and inference substantially.

- **Parallel dataset building (joblib)**
  - `training/build_training_dataset_cli.py` now supports `--n-jobs` to parallelize across tracks.

### Accuracy improvements (recommended next steps)

- **Ranking loss instead of regression** (PyTorch)
  - The task is top-K selection; pairwise/listwise ranking usually improves precision@K.

- **Use boundary features in the objective**
  - Penalize cuts that land on transients: `final = ml_score - α * boundary_quiet_score`.

### Optional ecosystem tools

- **MLflow**: experiment tracking (metrics + artifacts) for model comparisons
- **Dask**: scale feature extraction/training prep if you have very large corpora

### Training from your Dropbox music

Yes. There are two common approaches:

1) **If Dropbox is synced locally** (fastest)

- Just point dataset building at the local folder.
- Use the CLI:
  - `python training/build_training_dataset_cli.py --audio-dir "C:\\Users\\<you>\\Dropbox\\Music\\Training" --timestamps-json timestamps.json --out loop_training_dataset.csv`

2) **If you want to pull from Dropbox via API** (optional)

Dropbox does **not** support username/password authentication for third-party apps.
You connect via **OAuth** (browser login), which yields tokens.

- Add credentials to `.env` (either access token, or refresh-token flow)

#### Dropbox OAuth login helper (recommended)

1) Create a Dropbox app in the Dropbox developer console.
2) Run the interactive helper and follow the browser prompts:

- `python tools/dropbox_oauth_login.py --app-key YOUR_APP_KEY --offline`

3) Paste the printed env vars into `.env`.

Supported `.env` setups:

- **Option A (simple):** `DROPBOX_ACCESS_TOKEN=...`
- **Option B (recommended):** `DROPBOX_REFRESH_TOKEN=...` + `DROPBOX_APP_KEY=...` + `DROPBOX_APP_SECRET=...`

#### Option B1: Download then train (saves audio locally)

- Use:
  - `python tools/dropbox_download_audio.py --dropbox-folder "/Music/Training" --outdir data/raw_audio`

#### Option B2: Stream from Dropbox and **do not save audio locally** (recommended)

If you want training that reads directly from Dropbox **without writing audio files to disk**, use:

- Install the optional dependency:
  - `pip install -r requirements_optional_dropbox.txt`

- Train from one or more Dropbox folders (this is how you can "use all datasets" you collected in Dropbox):
  - `python training/train_loop_ranker_torch_dropbox_stream.py --dropbox-folders "/FMA,/Jamendo,/MUSDB18" --timestamps-json timestamps.json --model-dropbox-path "/Models/loop_ranker.pt"`

Notes:

- This script **streams audio bytes**, extracts features, trains a model, then **uploads the trained model to Dropbox**.
- It does **not** write audio files or a local model checkpoint.
- For pure in-memory decode, it currently supports WAV/FLAC/AIFF/AIF (MP3/M4A typically require ffmpeg/temp files).

### Training from online sources (safe + legal)

Use datasets that are explicitly licensed for research/training. Good options:

- **Free Music Archive (FMA)** (open-licensed music)
- **MTG-Jamendo** (Creative Commons music; requires following their terms)
- **GiantSteps Key/BPM** datasets (useful for tempo/key metadata; check licenses)
- **MUSDB18** (for stems; great if you later want explicit “drum stem” energy features)

Practical workflow:

1. Download an open dataset subset to a folder of audio files.
2. Build weak labels (timestamps) or better boundary labels.
3. Run `training/build_training_dataset_cli.py`.
4. Train `training/train_loop_ranker_torch.py`.
