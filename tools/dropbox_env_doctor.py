from __future__ import annotations

"""Dropbox .env doctor.

This script helps detect the common mistake of pasting Dropbox tokens into code
or into the wrong env var name.

It will:
- Parse .env
- Report which DROPBOX_* vars are present and whether they are empty (length only)
- If it finds an access token (starts with 'sl.') under the WRONG key, it will
  move it to DROPBOX_ACCESS_TOKEN automatically.

It never prints token values.

Usage:
  python tools/dropbox_env_doctor.py --apply
  python tools/dropbox_env_doctor.py
"""

import argparse
from pathlib import Path


def _parse_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _rewrite_env(original_text: str, new_map: dict[str, str]) -> str:
    # Rewrite only DROPBOX_* assignments; preserve other lines/comments as-is.
    lines = original_text.splitlines()
    out_lines: list[str] = []
    seen: set[str] = set()

    for line in lines:
        s = line.strip()
        if s.startswith("DROPBOX_") and "=" in line:
            k = line.split("=", 1)[0].strip()
            if k in new_map:
                out_lines.append(f"{k}={new_map[k]}")
                seen.add(k)
            else:
                out_lines.append(line)
            continue
        out_lines.append(line)

    # Ensure required keys exist
    for k in ["DROPBOX_ACCESS_TOKEN", "DROPBOX_REFRESH_TOKEN", "DROPBOX_APP_KEY", "DROPBOX_APP_SECRET"]:
        if k not in seen and k not in "\n".join(lines):
            out_lines.append(f"{k}=")

    return "\n".join(out_lines) + ("\n" if not original_text.endswith("\n") else "")


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose and optionally fix Dropbox .env entries")
    p.add_argument("--apply", action="store_true", help="Apply safe fixes to .env")
    args = p.parse_args()

    env_path = Path(".env")
    if not env_path.exists():
        print(".env not found in repo root.")
        return 2

    txt = env_path.read_text(encoding="utf-8", errors="replace")
    m = _parse_env(txt)

    keys = ["DROPBOX_ACCESS_TOKEN", "DROPBOX_REFRESH_TOKEN", "DROPBOX_APP_KEY", "DROPBOX_APP_SECRET"]
    print("Dropbox env (.env) status (length only):")
    for k in keys:
        v = m.get(k, "")
        print(f"- {k}: len={len(v)}")

    # Safe auto-fix: if access token is pasted under the wrong key.
    access_like_keys = [k for k, v in m.items() if k.startswith("DROPBOX_") and v.startswith("sl.")]
    if "DROPBOX_ACCESS_TOKEN" not in m or not m.get("DROPBOX_ACCESS_TOKEN"):
        for k in access_like_keys:
            if k != "DROPBOX_ACCESS_TOKEN":
                print(f"Found access-token-looking value under {k} (len={len(m[k])}).")
                if args.apply:
                    m["DROPBOX_ACCESS_TOKEN"] = m[k]
                    m[k] = ""
                    print("Moved it to DROPBOX_ACCESS_TOKEN and cleared the original key.")
                break

    if args.apply:
        new_txt = _rewrite_env(txt, m)
        env_path.write_text(new_txt, encoding="utf-8")
        print("Wrote updated .env")

    print("\nNext: run 'python tools/dropbox_verify_auth.py' to test the API connection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
