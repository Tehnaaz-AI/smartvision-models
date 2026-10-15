"""Shared project paths and environment-based settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = Path(os.environ.get("SMARTVISION_OCR_DATA_DIR") or DEFAULT_DATA_DIR).expanduser()
BASE_MODEL_ID = "microsoft/trocr-base-printed"
MODEL_NAME_OR_PATH = os.environ.get("SMARTVISION_OCR_MODEL", BASE_MODEL_ID)
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / "final"
