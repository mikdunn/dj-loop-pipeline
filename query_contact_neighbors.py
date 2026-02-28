import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd


def estimate_bpm(file_path: Path, sr: int = 22050, max_seconds: float = 45.0) -> Optional[float]:
    try:
        y, fs = librosa.load(str(file_path), sr=sr, mono=True)
    except Exception:
        return None

    if y is None or len(y) < 4096:
        return None

    y = y[: int(max_seconds * fs)]
    onset = librosa.onset.onset_strength(y=y, sr=fs)
    if onset.size < 8:
        return None

    try:
        bpm = float(librosa.feature.tempo(onset_envelope=onset, sr=fs, aggregate=np.median)[0])
    except Exception:
        return None

    if not np.isfinite(bpm) or bpm <= 0:
        return None
    return bpm


def resolve_query_file(query: str, available_files: List[str]) -> str:
    q = query.strip().lower()

    # exact path match first
    exact = [f for f in available_files if f.lower() == q]
    if exact:
        return exact[0]

    # basename exact match
    qname = Path(query).name.lower()
    by_name = [f for f in available_files if Path(f).name.lower() == qname]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise ValueError(
            f"Query basename matched multiple files ({len(by_name)}). Provide full path or a more specific substring."
        )

    # substring fallback
    subs = [f for f in available_files if q in f.lower()]
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        raise ValueError(
            f"Query substring matched multiple files ({len(subs)}). Provide full path or more specific query."
        )

    raise ValueError("No file matched query in nearest-neighbors CSV")


def infer_bpm_shift(query_bpm: Optional[float], neighbor_bpm: Optional[float], stretch_ratio: float) -> Dict[str, Optional[float]]:
    # stretch_ratio in CSV is neighbor_over_query from DTW path geometry.
    # If we stretch neighbor by this ratio, neighbor should align to query's timing.
    out: Dict[str, Optional[float]] = {
        "query_bpm": query_bpm,
        "neighbor_bpm": neighbor_bpm,
        "target_neighbor_bpm_from_stretch": None,
        "neighbor_pitch_fader_pct": None,
        "neighbor_pct_to_match_query_bpm": None,
    }

    if neighbor_bpm is not None and np.isfinite(stretch_ratio) and stretch_ratio > 0:
        target_bpm = float(neighbor_bpm * stretch_ratio)
        out["target_neighbor_bpm_from_stretch"] = target_bpm
        out["neighbor_pitch_fader_pct"] = float((stretch_ratio - 1.0) * 100.0)

    if query_bpm is not None and neighbor_bpm is not None and neighbor_bpm > 0:
        pct = (query_bpm / neighbor_bpm - 1.0) * 100.0
        out["neighbor_pct_to_match_query_bpm"] = float(pct)

    return out


def run_query(
    nn_csv: Path,
    query: str,
    top_k: int,
    out_csv: Optional[Path],
    estimate_bpms: bool,
) -> pd.DataFrame:
    if not nn_csv.exists():
        raise FileNotFoundError(f"Nearest-neighbors CSV not found: {nn_csv}")

    df = pd.read_csv(nn_csv)
    required = {
        "file",
        "neighbor_rank",
        "neighbor_file",
        "similarity",
        "distance",
        "stretch_ratio_neighbor_over_file",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    files = sorted(df["file"].astype(str).unique().tolist())
    selected = resolve_query_file(query, files)

    out = (
        df[df["file"].astype(str) == selected]
        .sort_values("neighbor_rank", ascending=True)
        .head(max(1, int(top_k)))
        .copy()
    )

    q_bpm = estimate_bpm(Path(selected)) if estimate_bpms else None

    query_bpms: List[Optional[float]] = []
    neighbor_bpms: List[Optional[float]] = []
    target_bpms: List[Optional[float]] = []
    pitch_pcts: List[Optional[float]] = []
    match_query_pcts: List[Optional[float]] = []

    if estimate_bpms:
        for _, r in out.iterrows():
            n_file = Path(str(r["neighbor_file"]))
            n_bpm = estimate_bpm(n_file)
            stretch = float(r["stretch_ratio_neighbor_over_file"])
            shift = infer_bpm_shift(q_bpm, n_bpm, stretch)
            query_bpms.append(shift["query_bpm"])
            neighbor_bpms.append(shift["neighbor_bpm"])
            target_bpms.append(shift["target_neighbor_bpm_from_stretch"])
            pitch_pcts.append(shift["neighbor_pitch_fader_pct"])
            match_query_pcts.append(shift["neighbor_pct_to_match_query_bpm"])

        out["query_bpm_est"] = query_bpms
        out["neighbor_bpm_est"] = neighbor_bpms
        out["neighbor_target_bpm_from_stretch"] = target_bpms
        out["neighbor_pitch_fader_pct_from_stretch"] = pitch_pcts
        out["neighbor_pitch_fader_pct_to_match_query_bpm"] = match_query_pcts

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Query nearest contact-map neighbors for one file with BPM/stretch hints")
    parser.add_argument(
        "--nn_csv",
        default="training/models/contact_map/contact_nearest_neighbors.csv",
        help="Path to contact_nearest_neighbors.csv",
    )
    parser.add_argument("--query", required=True, help="Query file path, filename, or unique substring")
    parser.add_argument("--top_k", type=int, default=10, help="Number of neighbors to return")
    parser.add_argument("--out_csv", default=None, help="Optional output CSV path for query results")
    parser.add_argument(
        "--disable_bpm_estimation",
        action="store_true",
        help="Skip BPM estimation and pitch-fader suggestions",
    )
    args = parser.parse_args()

    out = run_query(
        nn_csv=Path(args.nn_csv),
        query=args.query,
        top_k=max(1, int(args.top_k)),
        out_csv=Path(args.out_csv) if args.out_csv else None,
        estimate_bpms=not args.disable_bpm_estimation,
    )

    print("\nTop neighbors:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
