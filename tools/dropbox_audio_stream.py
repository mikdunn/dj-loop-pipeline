from __future__ import annotations

"""Dropbox audio streaming helpers (no local file writes).

This module downloads file bytes from Dropbox and decodes into (y, sr) in memory.

Supported decode formats without writing to disk:
- WAV/FLAC/AIFF/AIF (via soundfile)

MP3/M4A typically require ffmpeg-backed decoders; intentionally not supported here
because it usually forces external binaries and/or temp files.

Auth:
- Set DROPBOX_ACCESS_TOKEN in environment or .env
"""

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

# Allow running as a script: `python tools/dropbox_audio_stream.py ...`
if __package__ is None or __package__ == "":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def _load_dotenv_if_present() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _require_dropbox():
    try:
        import dropbox  # type: ignore
    except Exception as e:
        raise ImportError(
            "Missing dependency: dropbox. Install it with 'pip install dropbox'."
        ) from e
    return dropbox


def get_dropbox_client():
    _load_dotenv_if_present()
    dropbox = _require_dropbox()

    # Preferred: refresh-token based auth (recommended for long-term use)
    refresh = os.environ.get("DROPBOX_REFRESH_TOKEN")
    app_key = os.environ.get("DROPBOX_APP_KEY")
    app_secret = os.environ.get("DROPBOX_APP_SECRET")
    if refresh and app_key and app_secret:
        return dropbox.Dropbox(
            oauth2_refresh_token=refresh,
            app_key=app_key,
            app_secret=app_secret,
        )

    # Fallback: access token
    token = os.environ.get("DROPBOX_ACCESS_TOKEN")
    if token:
        return dropbox.Dropbox(token)

    raise RuntimeError(
        "Dropbox credentials not configured.\n\n"
        "Set ONE of:\n"
        "  - DROPBOX_ACCESS_TOKEN\n"
        "OR\n"
        "  - DROPBOX_REFRESH_TOKEN + DROPBOX_APP_KEY + DROPBOX_APP_SECRET\n\n"
        "Tip: run tools/dropbox_oauth_login.py to generate these." 
    )


def iter_dropbox_files(dbx, folder: str, *, exts: Sequence[str]) -> Iterator[Tuple[str, str]]:
    """Yield (path_lower, name) for files under folder matching extensions."""

    dropbox = _require_dropbox()
    want = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

    res = dbx.files_list_folder(folder, recursive=True)
    entries = list(res.entries)
    while res.has_more:
        res = dbx.files_list_folder_continue(res.cursor)
        entries.extend(res.entries)

    for entry in entries:
        if not isinstance(entry, dropbox.files.FileMetadata):
            continue
        suffix = Path(entry.name).suffix.lower()
        if suffix in want:
            yield entry.path_lower, entry.name


def download_bytes(dbx, dropbox_path_lower: str) -> bytes:
    _md, resp = dbx.files_download(dropbox_path_lower)
    return resp.content


def decode_audio_bytes(
    content: bytes,
    *,
    target_sr: int = 44100,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Decode WAV/FLAC/AIFF bytes to float32 mono (optionally resampled)."""

    import soundfile as sf

    bio = BytesIO(content)
    y, sr = sf.read(bio, dtype="float32", always_2d=False)

    if y.ndim == 2:
        # shape [samples, channels]
        if mono:
            y = np.mean(y, axis=1).astype(np.float32, copy=False)
        else:
            # librosa expects [samples] or [channels, samples]; we keep [samples, channels]
            y = y.astype(np.float32, copy=False)

    if sr != target_sr:
        import librosa

        y = librosa.resample(y.astype(np.float32, copy=False), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y.astype(np.float32, copy=False), int(sr)
