from pathlib import Path
import contextlib
import os


_REPO_ROOT = Path(__file__).resolve().parents[2]
MOVEITMOVEIT_DATA_DIR = Path(os.environ.get(
    "MOVEITMOVEIT_DATA_DIR", _REPO_ROOT / "data"
))

CONFIGS_DIR = _REPO_ROOT / "configs"

MODEL_DESCRIPTIONS_DIR = MOVEITMOVEIT_DATA_DIR / "model_descriptions"
MOTIONS_DIR = MOVEITMOVEIT_DATA_DIR / "motions"


def _dir_mtime(path: str) -> float:
    """Best-effort 'last written to' time for a directory.

    Prefers the mtime of the directory itself (which updates when entries are
    added/removed directly inside it), falling back to the newest mtime among
    its immediate files if that's more informative.
    """
    try:
        latest = os.path.getmtime(path)
    except OSError:
        return -1.0
    try:
        for entry in os.scandir(path):
            with contextlib.suppress(OSError):
                latest = max(latest, entry.stat().st_mtime)
    except OSError:
        pass
    return latest

def _find_most_recent_run_dir(log_root_path: str) -> str:
    """Return the run directory under `log_root_path` most recently written to.

    Layout is logs/<experiment>/<algorithm>/<run>/checkpoints/*.pt, so this
    looks one level down from `log_root_path` for run directories and ranks
    them by the mtime of their `checkpoints/` subdir (falling back to the run
    directory itself).
    """
    if not os.path.isdir(log_root_path):
        raise FileNotFoundError(f"Log directory does not exist: {log_root_path}")

    run_dirs = [
        entry.path for entry in os.scandir(log_root_path) if entry.is_dir()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {log_root_path}")

    def run_key(run_dir: str) -> float:
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        if os.path.isdir(checkpoints_dir):
            return _dir_mtime(checkpoints_dir)
        return _dir_mtime(run_dir)

    return max(run_dirs, key=run_key)

def resolve_checkpoint(log_root_path: str, checkpoint: str = None, step: int = None) -> str:
    """Resolve the checkpoint path to play.

    - If `checkpoint` is given, it is used as-is.
    - Otherwise, the most recently written-to run directory under
      `log_root_path` is used, and:
        - if `step` is given, checkpoints/{step}.pt is used (error if missing)
        - otherwise, checkpoints/best_agent.pt is used
    """
    if checkpoint:
        return os.path.abspath(checkpoint)

    most_recent_run_dir = _find_most_recent_run_dir(log_root_path)

    if step is not None:
        checkpoint_path = os.path.join(most_recent_run_dir, "checkpoints", f"{step}.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"No checkpoint for step {step} found in the most recent run: {checkpoint_path}"
            )
        return os.path.abspath(checkpoint_path)

    checkpoint_path = os.path.join(most_recent_run_dir, "checkpoints", "best_agent.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No best_agent.pt found in the most recent run: {checkpoint_path}"
        )
    return os.path.abspath(checkpoint_path)
