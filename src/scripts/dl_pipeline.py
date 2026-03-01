#!/usr/bin/env python3
"""
Compatibility shim.

Tests and older entrypoints expect: src/scripts/dl_pipeline.py
Canonical implementation lives at: pipelines/dl_pipeline.py
"""
from __future__ import annotations
import runpy

if __name__ == "__main__":
    runpy.run_path("pipelines/dl_pipeline.py", run_name="__main__")
