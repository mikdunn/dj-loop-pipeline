# DJ Loop Extraction Pipeline

## End-to-End ML-Powered Loop Finder and Exporter

Automatically generates DJ/DAW-ready loops from audio tracks using a combination of hand-crafted features, pretrained music embeddings, and XGBoost machine learning scoring.

---

## Features

- **Bar-aligned candidate generation**
  - Automatically slices tracks into candidate loops based on beats and bars.
- **Full feature extraction**
  - Energy, RMS, onset strength, and rhythm analysis.
  - Pretrained **OpenL3 embeddings** for rich, learned music representations.
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

## Multi-label training label sources

`train_multilabel_loop_classifier.py` supports configurable weak-label sources:

- `--label_source filename`: legacy filename regex tags from `TAG_PATTERNS`
- `--label_source audio`: universal, acoustically grounded tags from audio analysis (tempo bin, beat stability, syncopation proxy, percussive ratio, onset density, phrase regularity)
- `--label_source hybrid`: union of filename + audio tags

Example:

`python train_multilabel_loop_classifier.py --folder "<loops>" --label_source audio`

Outputs remain compatible with downstream scripts and keep the same prediction schema (`predicted_tags`, `prob_*`).

## Contact-map overlap labels (DTW + Laplacian)

For SVD/Laplacian-style similarity analysis and pseudo-labeling from diagonal blocks, use:

- `build_contact_map_labels.py`

This computes pairwise overlap similarity between loops using DTW-aligned rhythmic feature sequences (onset/transient, low-band rhythm energy, tempogram summary), then:

- builds a contact/similarity matrix
- reorders it by Fiedler vector (Laplacian eigenvector) to reveal block-diagonal structure
- exports pseudo-labels from spectral bins/eigenvector signs

Example:

`python build_contact_map_labels.py --folder "<loops>" --out_dir training/models/contact_map --max_files 200`

With explicit ranking controls:

`python build_contact_map_labels.py --folder "<loops>" --out_dir training/models/contact_map --max_files 200 --top_k_neighbors 10 --top_pairs_limit 500`

With Laplacian/spectral-style sharpening:

- Self-tuned local scaling + kNN graph (default):

`python build_contact_map_labels.py --folder "<loops>" --out_dir training/models/contact_map --sharpen_mode self_tuned_knn --self_tune_k 7`

- Doubly-stochastic Sinkhorn normalization:

`python build_contact_map_labels.py --folder "<loops>" --out_dir training/models/contact_map --sharpen_mode sinkhorn --sinkhorn_iters 25`

Main outputs:

- `contact_similarity_matrix.csv`
- `contact_distance_matrix.csv`
- `contact_stretch_ratio_matrix.csv`
- `contact_map_labels.csv` (`contact_bin`, `contact_tags`)
- `contact_map_order.csv`
- `contact_top_pairs.csv` (global highest-similarity pairs)
- `contact_nearest_neighbors.csv` (top neighbors for each file)
- `contact_similarity_matrix_raw.csv` (unsharpened baseline)
- `contact_similarity_heatmap.png` (if matplotlib is available)
- `contact_similarity_heatmap_raw.png` (when sharpening is enabled)
- `contact_map_summary.json`

### Train + use contact-map classifier head

1) Build contact-map labels:

`python build_contact_map_labels.py --folder "<loops>" --out_dir training/models/contact_map --max_files 200`

1) Train contact-map multilabel classifier:

`python train_contact_map_classifier.py --labels_csv training/models/contact_map/contact_map_labels.csv --model_out training/models/contact_map/contact_multilabel_classifier.joblib --report_out training/models/contact_map/contact_multilabel_report.json`

1) Run standard + contact heads together in one CSV:

`python classify_multilabel_loops.py --model training/models/loop_multilabel_classifier.joblib --contact_model training/models/contact_map/contact_multilabel_classifier.joblib --folder "<loops>" --out training/models/loop_multilabel_predictions_with_contact.csv`

Combined output keeps existing columns and adds:

- `predicted_contact_tags`
- `prob_contact_*`

### Query nearest neighbors + stretch hints for one file

After generating `contact_nearest_neighbors.csv`, you can query one loop and get ranked neighbors with BPM/stretch suggestions:

`python query_contact_neighbors.py --nn_csv training/models/contact_map/contact_nearest_neighbors.csv --query "<filename-or-path-or-unique-substring>" --top_k 10 --out_csv training/models/contact_map/query_neighbors.csv`

Useful output columns include:

- `similarity` / `distance`
- `stretch_ratio_neighbor_over_file`
- `neighbor_pitch_fader_pct_from_stretch`
- `neighbor_pitch_fader_pct_to_match_query_bpm`
