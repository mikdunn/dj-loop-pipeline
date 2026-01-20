from __future__ import annotations

"""Interactive Dropbox OAuth login helper.

Dropbox does NOT support username+password authentication for third-party apps.
Instead, you authenticate via OAuth in a browser and obtain tokens.

This script starts an OAuth flow, opens the authorization URL, and then asks you
to paste the authorization code.

Outputs env var lines you can paste into .env:
- DROPBOX_ACCESS_TOKEN
- (optional) DROPBOX_REFRESH_TOKEN
- (optional) DROPBOX_APP_KEY / DROPBOX_APP_SECRET if refresh is used

Requirements:
  pip install dropbox

Usage examples:
  python tools/dropbox_oauth_login.py --app-key YOUR_APP_KEY
  python tools/dropbox_oauth_login.py --app-key YOUR_APP_KEY --app-secret YOUR_APP_SECRET --offline

Notes:
- If you use --offline and your Dropbox app supports refresh tokens, you'll get a refresh token.
- Keep tokens secret. Do not commit .env.
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path
from typing import Optional


# Allow running as a script: `python tools/dropbox_oauth_login.py ...`
if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_dotenv_if_present() -> None:
    """Lightweight .env loader (no dependency on python-dotenv)."""

    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _parse_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _upsert_env_lines(original: str, updates: dict[str, str]) -> str:
    """Upsert (key=value) lines while preserving comments and non-updated lines."""

    lines = original.splitlines()
    out_lines: list[str] = []
    seen: set[str] = set()

    for line in lines:
        if "=" in line:
            k = line.split("=", 1)[0].strip()
            if k in updates:
                out_lines.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        out_lines.append(line)

    for k, v in updates.items():
        if k not in seen and k not in _parse_env(original):
            out_lines.append(f"{k}={v}")

    return "\n".join(out_lines) + ("\n" if not original.endswith("\n") else "")


def _require_dropbox():
    try:
        import dropbox  # type: ignore
    except Exception as e:
        raise ImportError(
            "Missing dependency: dropbox. Install it with 'pip install dropbox'."
        ) from e
    return dropbox


def main() -> int:
    _load_dotenv_if_present()

    p = argparse.ArgumentParser(description="Dropbox OAuth login helper (no username/password).")
    p.add_argument("--app-key", default=os.environ.get("DROPBOX_APP_KEY"), help="Dropbox app key")
    p.add_argument(
        "--app-secret",
        default=os.environ.get("DROPBOX_APP_SECRET"),
        help="Dropbox app secret (optional; needed for some refresh-token flows)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Request a refresh token (token_access_type='offline') when supported.",
    )
    p.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open the browser (just print the URL).",
    )
    p.add_argument(
        "--write-env",
        action="store_true",
        help="Write resulting tokens into .env (recommended; avoids copy/paste errors).",
    )
    p.add_argument(
        "--print-secrets",
        action="store_true",
        help="Print token lines to stdout (not recommended).",
    )
    args = p.parse_args()

    if not args.app_key:
        raise SystemExit("Missing --app-key (or set DROPBOX_APP_KEY in env).")

    dropbox = _require_dropbox()

    # Use PKCE when possible to avoid requiring app secret for the authorization step.
    # We still accept app_secret for compatibility.
    token_access_type: Optional[str] = "offline" if args.offline else "online"

    try:
        flow = dropbox.DropboxOAuth2FlowNoRedirect(
            args.app_key,
            args.app_secret,
            token_access_type=token_access_type,
            use_pkce=True,
        )
    except TypeError:
        # Older SDKs may not support some kwargs.
        flow = dropbox.DropboxOAuth2FlowNoRedirect(args.app_key, args.app_secret)

    authorize_url = flow.start()
    print("\n1) Open this URL and approve access:\n")
    print(authorize_url)
    print()

    if not args.no_open:
        try:
            webbrowser.open(authorize_url)
        except Exception:
            pass

    auth_code = input("2) Paste the authorization code here: ").strip()
    if not auth_code:
        raise SystemExit("No authorization code provided.")

    oauth_result = flow.finish(auth_code)

    # oauth_result fields vary by SDK version.
    access_token = getattr(oauth_result, "access_token", None) or getattr(oauth_result, "oauth2_access_token", None)
    refresh_token = getattr(oauth_result, "refresh_token", None)
    expires_at = getattr(oauth_result, "expires_at", None)
    account_id = getattr(oauth_result, "account_id", None)

    print("\nSuccess.")
    if account_id:
        print(f"account_id: {account_id}")
    if expires_at:
        print(f"expires_at: {expires_at}")

    updates: dict[str, str] = {}
    if args.app_key:
        updates["DROPBOX_APP_KEY"] = args.app_key
    if args.app_secret:
        updates["DROPBOX_APP_SECRET"] = args.app_secret
    if refresh_token:
        updates["DROPBOX_REFRESH_TOKEN"] = refresh_token
    if access_token:
        updates["DROPBOX_ACCESS_TOKEN"] = access_token

    if args.write_env:
        env_path = Path(".env")
        existing = env_path.read_text(encoding="utf-8", errors="replace") if env_path.exists() else ""
        env_path.write_text(_upsert_env_lines(existing, updates), encoding="utf-8")
        # Never print secrets, even if writing.
        print("\nWrote updated .env (values not printed).")
        for k, v in updates.items():
            print(f"- {k}: len={len(v)}")
    else:
        if args.print_secrets:
            print("\nPaste these into your .env (keep them secret):\n")
            for k in ["DROPBOX_APP_KEY", "DROPBOX_APP_SECRET", "DROPBOX_REFRESH_TOKEN", "DROPBOX_ACCESS_TOKEN"]:
                if k in updates:
                    print(f"{k}={updates[k]}")
        else:
            print("\nTokens ready (values not printed).")
            for k, v in updates.items():
                print(f"- {k}: len={len(v)}")
            print("\nRe-run with --write-env to automatically update .env.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
