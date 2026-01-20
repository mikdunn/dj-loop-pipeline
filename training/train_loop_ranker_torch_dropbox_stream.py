from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Allow running as a script: `python training/train_loop_ranker_torch_dropbox_stream.py ...`
if __package__ is None or __package__ == "":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tools.dropbox_audio_stream import (
    decode_audio_bytes,
    download_bytes,
    get_dropbox_client,
    iter_dropbox_files,
)
from training.dataset_builder import compute_candidate_weight, generate_bar_aligned_candidates
from training.feature_extraction import extract_full_features
from training.torch_ranker import build_mlp


def _parse_list(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def _as_float_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    x = df[cols].to_numpy(dtype=np.float32, copy=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _build_examples_for_track(
    *,
    track_id: str,
    y: np.ndarray,
    sr: int,
    timestamps: Sequence[float],
    bars_options: Sequence[int],
    max_candidates: int,
    seed: int,
) -> pd.DataFrame:
    import librosa

    bpm, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    cands = generate_bar_aligned_candidates(track_id, beats, bpm, list(bars_options))

    # Optionally subsample candidates to keep training bounded.
    if max_candidates and len(cands) > max_candidates:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(cands), size=max_candidates, replace=False)
        cands = [cands[i] for i in idx]

    rows = []
    for cand in cands:
        feats = extract_full_features(y, sr, cand.start_time, cand.end_time)
        if not feats:
            continue
        row = feats.copy()
        row.update(
            {
                "track_id": track_id,
                "start_time": cand.start_time,
                "end_time": cand.end_time,
                "bars": cand.bars,
                "weight": compute_candidate_weight(cand.center_time, list(timestamps), bpm),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _save_torch_checkpoint_to_dropbox(
    *,
    dbx,
    model,
    features: List[str],
    dropbox_dest_path: str,
    meta: Dict,
) -> None:
    """Upload a torch checkpoint to Dropbox without saving locally."""

    import torch

    ckpt = {
        "state_dict": model.state_dict(),
        "features": list(features),
        "meta": meta,
        # no scaler: we train without a global scaler for streaming
    }

    bio = BytesIO()
    torch.save(ckpt, bio)
    bio.seek(0)

    # Use overwrite mode
    from tools.dropbox_audio_stream import _require_dropbox

    dropbox = _require_dropbox()
    dbx.files_upload(
        bio.read(),
        dropbox_dest_path,
        mode=dropbox.files.WriteMode.overwrite,
        mute=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Stream audio from Dropbox and train a PyTorch loop ranker without saving audio locally.\n\n"
            "Expected inputs:\n"
            "- One or more Dropbox folders containing WAV/FLAC/AIFF files\n"
            "- timestamps.json mapping track_id -> list[seconds] (weak labels)\n\n"
            "Output:\n"
            "- Uploads loop_ranker.pt to Dropbox (no local model save if you provide --model-dropbox-path)."
        )
    )

    p.add_argument(
        "--dropbox-folders",
        required=True,
        help=(
            "Comma-separated Dropbox folders to scan, e.g. /FMA,/Jamendo,/MUSDB18/train"
        ),
    )
    p.add_argument(
        "--timestamps-json",
        required=True,
        help="Local JSON mapping track_id -> list of seconds (labels)",
    )
    p.add_argument(
        "--ext",
        default="wav,flac,aiff,aif",
        help="Extensions to include (no mp3/m4a for pure in-memory decode)",
    )

    p.add_argument("--bars", default="2,4,8,16", help="Bars options for candidate generation")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", default="256,128")
    p.add_argument("--dropout", type=float, default=0.10)

    p.add_argument(
        "--max-candidates-per-track",
        type=int,
        default=500,
        help="Randomly subsample candidates per track to bound compute",
    )
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument(
        "--model-dropbox-path",
        required=True,
        help="Dropbox destination path for model checkpoint, e.g. /Models/loop_ranker.pt",
    )

    args = p.parse_args()

    dropbox_folders = _parse_list(args.dropbox_folders)
    exts = _parse_list(args.ext)
    bars_options = _parse_int_list(args.bars)

    with open(args.timestamps_json, "r", encoding="utf-8") as f:
        timestamps: Dict[str, Sequence[float]] = json.load(f)

    dbx = get_dropbox_client()

    # Build a manifest of files across all folders
    files: List[Tuple[str, str]] = []
    for folder in dropbox_folders:
        files.extend(list(iter_dropbox_files(dbx, folder, exts=exts)))

    if not files:
        raise SystemExit("No audio files found in Dropbox folders for the requested extensions")

    print(f"Found {len(files)} audio files across {len(dropbox_folders)} folder(s).")

    # Stream -> build a training dataframe (features) in memory.
    # NOTE: This stores features locally in RAM, not audio files.
    frames: List[pd.DataFrame] = []

    for path_lower, name in files:
        track_id = Path(name).stem
        label_times = timestamps.get(track_id, [])

        content = download_bytes(dbx, path_lower)
        y, sr = decode_audio_bytes(content, target_sr=44100, mono=True)

        df_track = _build_examples_for_track(
            track_id=track_id,
            y=y,
            sr=sr,
            timestamps=label_times,
            bars_options=bars_options,
            max_candidates=args.max_candidates_per_track,
            seed=args.seed,
        )
        if len(df_track) == 0:
            continue
        frames.append(df_track)
        print(f"{track_id}: {len(df_track)} examples")

    if not frames:
        raise SystemExit("No training examples were produced (check beat tracking and labels)")

    df = pd.concat(frames, axis=0, ignore_index=True)

    drop_cols = {"weight", "track_id", "start_time", "end_time"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    x_all = _as_float_matrix(df, feature_cols)
    y_all = df["weight"].to_numpy(dtype=np.float32).reshape(-1, 1)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_sizes = tuple(int(s.strip()) for s in args.hidden.split(",") if s.strip())
    model = build_mlp(len(feature_cols), hidden_sizes=hidden_sizes, dropout=args.dropout).to(device)

    ds = TensorDataset(torch.from_numpy(x_all), torch.from_numpy(y_all))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        print(f"epoch {epoch:03d} | train_mse={float(np.mean(losses)):.6f} | device={device}")

    # Upload checkpoint to Dropbox (no local model file written).
    model_cpu = model.to("cpu").eval()
    meta = {
        "arch": "mlp",
        "hidden_sizes": list(hidden_sizes),
        "dropout": float(args.dropout),
        "trained_epochs": int(args.epochs),
        "bars_options": list(bars_options),
        "note": "Trained from Dropbox-streamed audio without local audio saves.",
    }
    _save_torch_checkpoint_to_dropbox(
        dbx=dbx,
        model=model_cpu,
        features=feature_cols,
        dropbox_dest_path=args.model_dropbox_path,
        meta=meta,
    )

    print(f"Uploaded model -> {args.model_dropbox_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
