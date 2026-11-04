"""Train micro-gesture recognition in macro activity recordings using rgb.

Use --experiment to select the training context, --protocol for participant or
macro-repetition evaluation, --audit-only to inspect splits, and --prepare-only
to prepare signals/frames. All outputs are saved inside macro/runs/rgb.
"""
import os
import sys
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from macro_tools.cli import main

if __name__ == "__main__":
    raise SystemExit(main("rgb"))
