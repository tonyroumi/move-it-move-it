from pathlib import Path
import os

_REPO_ROOT = Path(__file__).resolve().parents[2]
MOVEITMOVEIT_DATA_DIR = Path(os.environ.get(
    "MOVEITMOVEIT_DATA_DIR", _REPO_ROOT / "data"
))

CONFIGS_DIR = _REPO_ROOT / "configs"

MODEL_DESCRIPTIONS_DIR = MOVEITMOVEIT_DATA_DIR / "model_descriptions"
MOTIONS_DIR = MOVEITMOVEIT_DATA_DIR / "motions"
