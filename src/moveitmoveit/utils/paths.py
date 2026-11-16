from pathlib import Path
import os


_REPO_ROOT = Path(__file__).resolve().parents[3]
MOVEITMOVEIT_DATA_DIR = Path(os.environ.get(
    "MOVEITMOVEIT_DATA_DIR", _REPO_ROOT / "data"
))

CONFIGS_DIR = _REPO_ROOT / "configs"
