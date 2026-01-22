# Thin wrapper so batch_export_loops.py can import from pipelines
from importlib import import_module

_base = import_module('export_top_loops_pro')
export_top_loops = _base.export_top_loops
