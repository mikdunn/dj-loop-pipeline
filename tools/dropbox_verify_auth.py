from __future__ import annotations

"""Verify Dropbox credentials from .env / environment.

This script does NOT print any tokens.
It just checks that required env vars are present (non-empty) and then calls
users_get_current_account() to verify API access.

Usage:
  python tools/dropbox_verify_auth.py
"""

import os
import sys
from pathlib import Path

# Allow running as a script: `python tools/dropbox_verify_auth.py`
if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from tools.dropbox_audio_stream import _load_dotenv_if_present, get_dropbox_client

    _load_dotenv_if_present()

    keys = ["DROPBOX_ACCESS_TOKEN", "DROPBOX_REFRESH_TOKEN", "DROPBOX_APP_KEY", "DROPBOX_APP_SECRET"]
    for k in keys:
        v = os.environ.get(k, "")
        print(f"{k}: {'set' if v else 'MISSING'}")

    dbx = get_dropbox_client()
    acct = dbx.users_get_current_account()
    # This is account metadata, not a secret.
    print("\nConnected OK")
    print(f"Name:  {acct.name.display_name}")
    try:
        print(f"Email: {acct.email}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
