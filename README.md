# DJ Loop Extraction Pipeline 

**End-to-End ML-Powered Loop Finder and Exporter**

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
