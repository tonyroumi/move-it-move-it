"""Train once for every run directory under a sweep root, configured by the directory's name.

    python scripts/sweep.py [ROOT] [--force] [TRAIN_ARGS...]

ROOT (default `sweep`) holds one directory per run. A run directory's name is that run's whole
config, as `key=value` segments joined by `__`:

    algo=AMP__manifest=goal_manifest__num_envs=2048__seed=7

Each run invokes scripts/train.py in its own process with the interpreter running this script,
so launch the sweep the same way as training (`./isaaclab.sh -p scripts/sweep.py`). Its logs,
dumped params and checkpoints are written inside the run directory. TRAIN_ARGS, any other
arguments such as `--headless`, are passed to every run.

This module imports nothing from Isaac Lab or moveitmoveit, so names can be parsed and sweep roots
listed without launching Isaac.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
EXPERIMENT_FILE = REPO_ROOT / "configs" / "experiment.yaml"
MANIFESTS_DIR = REPO_ROOT / "configs" / "manifests"

SEGMENT_SEPARATOR = "__"

ALGOS = ("PPO", "AMP")
LOGGER_BACKENDS = ("tensorboard", "wandb", "both", "none")

# Keys that are passed to scripts/train.py as flags.
REQUIRED_KEYS = ("algo", "manifest")
TRAIN_FLAG_KEYS = ("num_envs", "checkpoint")
# Keys that override configs/experiment.yaml for this run only.
EXPERIMENT_KEYS = (
    "seed",
    "experiment.directory",
    "experiment.experiment_name",
    "logger.backend",
    "logger.write_interval",
    "logger.checkpoint_interval",
    "description",
)
# Keys forwarded verbatim to the preset/Hydra CLI of scripts/train.py.
HYDRA_PREFIXES = ("env.", "agent.")
HYDRA_PRESET_KEYS = ("presets", "physics", "renderer")

_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)*")
_INT_KEYS = {
    "num_envs": 1,
    "seed": 0,
    "logger.write_interval": 1,
    "logger.checkpoint_interval": 1,
}


class RunNameError(ValueError):
    """A run directory's name does not describe a valid run."""


@dataclass
class RunConfig:
    """What a run directory's name asks for."""

    algo: str
    manifest: str
    num_envs: int | None = None
    checkpoint: str | None = None
    # Dotted experiment.yaml key -> value, applied on top of configs/experiment.yaml.
    experiment: dict[str, object] = field(default_factory=dict)
    # `key=value` tokens for the preset/Hydra CLI, in the order the name gives them.
    overrides: list[str] = field(default_factory=list)


def _is_hydra_key(key: str) -> bool:
    return key in HYDRA_PRESET_KEYS or key.startswith(HYDRA_PREFIXES)


def _to_int(key: str, value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise RunNameError(f"{key} must be an integer, got {value!r}") from None
    minimum = _INT_KEYS[key]
    if number < minimum:
        raise RunNameError(f"{key} must be at least {minimum}, got {number}")
    return number


def _resolve_manifest(value: str) -> str:
    """A bare name like `goal_manifest` means configs/manifests/goal_manifest.yaml; anything with
    a `/` (percent-encoded in the name) is a path, used as given."""
    if "/" in value:
        return value
    filename = value if value.endswith((".yaml", ".yml")) else f"{value}.yaml"
    return str(MANIFESTS_DIR / filename)


def parse_run_name(name: str) -> RunConfig:
    """Parse a run directory's name into a RunConfig, or raise RunNameError saying why not."""
    if not name:
        raise RunNameError("empty name")

    values: dict[str, str] = {}
    order: list[str] = []
    for segment in name.split(SEGMENT_SEPARATOR):
        if not segment:
            raise RunNameError("empty segment; segments are separated by a single '__'")
        if "=" not in segment:
            raise RunNameError(f"segment {segment!r} is not key=value")
        key, raw_value = segment.split("=", 1)
        if "=" in raw_value:
            raise RunNameError(f"value of {key!r} contains '='; percent-encode it as %3D")
        if not _KEY_PATTERN.fullmatch(key):
            raise RunNameError(f"{key!r} is not a valid key")
        if key in values:
            raise RunNameError(f"{key!r} is given more than once")
        if not (key in REQUIRED_KEYS or key in TRAIN_FLAG_KEYS or key in EXPERIMENT_KEYS or _is_hydra_key(key)):
            raise RunNameError(
                f"unknown key {key!r}; expected one of "
                f"{', '.join(REQUIRED_KEYS + TRAIN_FLAG_KEYS + EXPERIMENT_KEYS)}, "
                f"{', '.join(HYDRA_PRESET_KEYS)}, or an env.* / agent.* override"
            )
        try:
            values[key] = unquote(raw_value, errors="strict")
        except UnicodeDecodeError:
            raise RunNameError(f"value of {key!r} is not valid percent-encoded UTF-8") from None
        order.append(key)

    missing = [key for key in REQUIRED_KEYS if key not in values]
    if missing:
        raise RunNameError(f"missing required key{'s' if len(missing) > 1 else ''}: {', '.join(missing)}")

    algo = values["algo"]
    if algo not in ALGOS:
        raise RunNameError(f"algo must be one of {', '.join(ALGOS)}, got {algo!r}")
    if not values["manifest"]:
        raise RunNameError("manifest is empty")

    config = RunConfig(algo=algo, manifest=_resolve_manifest(values["manifest"]))
    if "num_envs" in values:
        config.num_envs = _to_int("num_envs", values["num_envs"])
    if "checkpoint" in values:
        if not values["checkpoint"]:
            raise RunNameError("checkpoint is empty")
        config.checkpoint = values["checkpoint"]

    for key in order:
        value = values[key]
        if key in EXPERIMENT_KEYS:
            if key in _INT_KEYS:
                config.experiment[key] = _to_int(key, value)
            elif key == "logger.backend":
                if value not in LOGGER_BACKENDS:
                    raise RunNameError(f"logger.backend must be one of {', '.join(LOGGER_BACKENDS)}, got {value!r}")
                config.experiment[key] = value
            elif key != "description" and not value:
                raise RunNameError(f"{key} is empty")
            else:
                config.experiment[key] = value
        elif _is_hydra_key(key):
            config.overrides.append(f"{key}={value}")

    return config


def apply_experiment_overrides(base: dict, overrides: dict[str, object]) -> dict:
    """A copy of the experiment config `base` with dotted-key `overrides` applied."""
    merged = copy.deepcopy(base)
    for dotted_key, value in overrides.items():
        *parents, leaf = dotted_key.split(".")
        node = merged
        for part in parents:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[leaf] = value
    return merged


def discover_run_dirs(root: Path) -> list[Path]:
    """The immediate child directories of `root`, sorted by name. Hidden ones (a leading `.`)
    are not runs."""
    if not root.is_dir():
        raise FileNotFoundError(f"Sweep root is not a directory: {root}")
    return sorted(
        (entry for entry in root.iterdir() if entry.is_dir() and not entry.name.startswith(".")),
        key=lambda path: path.name,
    )


def finished_checkpoints(run_dir: Path) -> list[Path]:
    """The final checkpoints of the finished training runs inside `run_dir`.

    scripts/train.py dumps params/agent.yaml when a run starts and writes
    checkpoints/<runner.timesteps>.pt when it finishes, so a log directory holding both is a
    finished run. Periodic and best_agent.pt checkpoints don't count.
    """
    found = []
    for agent_params in sorted(run_dir.rglob("params/agent.yaml")):
        log_dir = agent_params.parent.parent
        try:
            with open(agent_params, "r") as f:
                timesteps = yaml.safe_load(f)["runner"]["timesteps"]
        except (OSError, yaml.YAMLError, KeyError, TypeError):
            continue
        checkpoint = log_dir / "checkpoints" / f"{timesteps}.pt"
        if checkpoint.is_file():
            found.append(checkpoint)
    return found


def build_train_command(
    config: RunConfig,
    run_dir: Path,
    experiment_file: Path,
    extra_args: list[str] | None = None,
) -> list[str]:
    """The scripts/train.py command line for one run."""
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--algo",
        config.algo,
        "--manifest",
        config.manifest,
        "--experiment",
        str(experiment_file),
        "--log_root",
        str(run_dir.resolve()),
    ]
    if config.num_envs is not None:
        command += ["--num_envs", str(config.num_envs)]
    if config.checkpoint is not None:
        command += ["--checkpoint", config.checkpoint]
    command += list(extra_args or [])
    command += config.overrides
    return command


def _train(config: RunConfig, run_dir: Path, base_experiment: dict, extra_args: list[str]) -> int:
    """Train one run in its own process and return its exit code. The run's experiment file is a
    temporary file outside the run directory, so the shared configs/experiment.yaml is untouched."""
    experiment = apply_experiment_overrides(base_experiment, config.experiment)
    fd, experiment_path = tempfile.mkstemp(prefix="sweep-experiment-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(experiment, f, sort_keys=False)
        command = build_train_command(config, run_dir, Path(experiment_path), extra_args)
        print(f"[sweep] {' '.join(command)}", flush=True)
        return subprocess.run(command).returncode
    finally:
        os.unlink(experiment_path)


def run_sweep(root: Path, force: bool = False, extra_args: list[str] | None = None) -> int:
    """Train every run under `root` in turn and return the number of runs that failed."""
    extra_args = list(extra_args or [])
    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        print(f"[sweep] No run directories under {root}")
        return 0

    with open(EXPERIMENT_FILE, "r") as f:
        base_experiment = yaml.safe_load(f)

    trained, skipped, failed = [], [], []
    for index, run_dir in enumerate(run_dirs, start=1):
        name = run_dir.name
        print(f"\n[sweep] ({index}/{len(run_dirs)}) {name}", flush=True)
        try:
            config = parse_run_name(name)
        except RunNameError as error:
            print(f"[sweep] ERROR {name}: {error}", flush=True)
            failed.append(name)
            continue
        if not Path(config.manifest).is_file():
            print(f"[sweep] ERROR {name}: manifest not found: {config.manifest}", flush=True)
            failed.append(name)
            continue

        finished = finished_checkpoints(run_dir)
        if finished and not force:
            print(f"[sweep] SKIP {name}: already finished ({finished[-1]}); pass --force to rerun", flush=True)
            skipped.append(name)
            continue

        returncode = _train(config, run_dir, base_experiment, extra_args)
        if returncode != 0:
            print(f"[sweep] ERROR {name}: training exited with code {returncode}", flush=True)
            failed.append(name)
        elif not set(finished_checkpoints(run_dir)) - set(finished):
            print(f"[sweep] ERROR {name}: training ended without a final checkpoint", flush=True)
            failed.append(name)
        else:
            trained.append(name)

    print(f"\n[sweep] {len(trained)} trained, {len(skipped)} skipped, {len(failed)} failed")
    for name in failed:
        print(f"[sweep]   failed: {name}")
    return len(failed)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # The root, if given, comes first, so a train.py flag's value (`--device cuda:1`) is never
    # taken for it.
    root = argv.pop(0) if argv and not argv[0].startswith("-") else "sweep"

    parser = argparse.ArgumentParser(
        usage="%(prog)s [ROOT] [--force] [TRAIN_ARGS ...]",
        description="Train once for every run directory under ROOT (default: sweep); each directory's name is its config.",
        epilog="Any other argument, such as --headless, is passed to every scripts/train.py run.",
        allow_abbrev=False,
    )
    parser.add_argument("--force", action="store_true", help="Rerun runs that already have a finished checkpoint.")
    args, extra_args = parser.parse_known_args(argv)
    # `--` separates TRAIN_ARGS when running with plain python; it isn't one of them.
    extra_args = [arg for arg in extra_args if arg != "--"]

    try:
        failures = run_sweep(Path(root), force=args.force, extra_args=extra_args)
    except FileNotFoundError as error:
        print(f"[sweep] {error}", file=sys.stderr)
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
