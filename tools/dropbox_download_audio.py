from __future__ import annotations

"""Optional Dropbox downloader.

This is useful if your Dropbox is NOT locally synced, or you want to pull a curated training subset.

Requires:
  pip install dropbox

Auth:
  Set DROPBOX_ACCESS_TOKEN in .env (or environment).

Notes:
- This script intentionally downloads *files you have rights to use*.
- It filters by extension; conversion to WAV is not performed here.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List

# Allow running as a script: `python tools/dropbox_download_audio.py ...`
if __package__ is None or __package__ == "":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _require_dropbox():
    try:
        import dropbox  # type: ignore
    except Exception as e:
        raise ImportError(
            "Missing dependency: dropbox. Install it with 'pip install dropbox'."
        ) from e
    return dropbox


def _load_dotenv_if_present():
    # Avoid adding python-dotenv dependency; keep it simple.
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


def _iter_entries(dbx, folder: str):
    res = dbx.files_list_folder(folder, recursive=True)
    yield from res.entries
    while res.has_more:
        res = dbx.files_list_folder_continue(res.cursor)
        yield from res.entries


def main() -> int:
    p = argparse.ArgumentParser(description="Download audio files from a Dropbox folder.")
    p.add_argument("--dropbox-folder", required=True, help="Dropbox folder path, e.g. /Music/Training")
    p.add_argument("--outdir", default="data/raw_audio", help="Local directory to write downloads")
    p.add_argument(
        "--ext",
        default="wav,mp3,flac,aiff,aif,m4a",
        help="Comma-separated extensions to download",
    )
    args = p.parse_args()

    # Reuse shared auth logic (access token OR refresh token flow)
    from tools.dropbox_audio_stream import get_dropbox_client

    dbx = get_dropbox_client()
    dropbox = _require_dropbox()

    exts = {f".{e.strip().lower().lstrip('.')}" for e in args.ext.split(",") if e.strip()}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0

    for entry in _iter_entries(dbx, args.dropbox_folder):
        if not isinstance(entry, dropbox.files.FileMetadata):
            continue
        suffix = Path(entry.name).suffix.lower()
        if suffix not in exts:
            skipped += 1
            continue

        local_path = outdir / entry.name
        if local_path.exists() and local_path.stat().st_size == entry.size:
            skipped += 1
            continue

        md, resp = dbx.files_download(entry.path_lower)
        local_path.write_bytes(resp.content)
        downloaded += 1
        print(f"downloaded: {entry.path_display} -> {local_path}")

    print(f"Done. downloaded={downloaded} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
