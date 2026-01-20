"""Convenience wrapper for Dropbox OAuth helper.

This lets you run:
  python dropbox_oauth_login.py --offline --write-env

instead of:
  python tools/dropbox_oauth_login.py --offline --write-env

All logic lives in tools/dropbox_oauth_login.py.
"""

from __future__ import annotations


def main() -> int:
    from tools.dropbox_oauth_login import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
