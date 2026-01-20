from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

# Allow running as a script: `python training/build_training_dataset_cli.py ...`
if __package__ is None or __package__ == "":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from training.dataset_builder import build_training_dataset


def _parse_bars(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    bars = [int(p) for p in parts]
    if not bars:
        raise ValueError("--bars must contain at least one integer")
    return bars


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Build loop_training_dataset.csv from a folder of audio (WAV) + timestamps labels.\n\n"
            "Tip: if Dropbox is installed, you can point --audio-dir at your local Dropbox sync folder."
        )
    )
    p.add_argument(
        "--audio-dir",
        required=True,
        help="Directory containing .wav files (can be a Dropbox-synced local folder)",
    )
    p.add_argument(
        "--timestamps-json",
        required=True,
        help=(
            "JSON mapping track_id -> list of labeled times in seconds, e.g. {\"track1\":[12.3, 58.0]}"
        ),
    )
    p.add_argument("--bars", default="1,2,4,8,16", help="Comma-separated bars options")
    p.add_argument("--out", default="loop_training_dataset.csv", help="Output CSV path")
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel processes across tracks")
    args = p.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"audio-dir does not exist: {audio_dir}")

    with open(args.timestamps_json, "r", encoding="utf-8") as f:
        timestamps: Dict[str, Sequence[float]] = json.load(f)

    bars = _parse_bars(args.bars)

    # build_training_dataset now uses a per-track cache; parallelize with --n-jobs when you have many WAVs
    from training.dataset_builder import build_training_dataset_parallel

    df: pd.DataFrame = build_training_dataset_parallel(
        audio_dir=audio_dir,
        timestamps=timestamps,
        bars_options=bars,
        n_jobs=args.n_jobs,
    )

    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
