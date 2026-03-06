"""Compatibility launcher/wrapper for relocated script.

New location: processes/06_analysis/analyze_loop_structure.py
"""

from pathlib import Path
import runpy as _runpy

_TARGET = Path(__file__).resolve().parent / r"processes/06_analysis/analyze_loop_structure.py"
_NS = _runpy.run_path(str(_TARGET))

for _k, _v in _NS.items():
    if not _k.startswith("__"):
        globals().setdefault(_k, _v)

if __name__ == "__main__":
    _main = _NS.get("main")
    if callable(_main):
        _main()
