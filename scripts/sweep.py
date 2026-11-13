"""Train once for every run directory under a sweep root.

    python scripts/sweep.py [ROOT] [--force] [--dry_run] [TRAIN_ARGS...]

ROOT (default `sweep`) holds one directory per run. Each run directory carries its own parameter
files, and its name says which of them to use and how to tell the run apart:

    sweep/algo=PPO__manifest=box_manifest__num_envs=4096__seed=1__experiment.experiment_name=boxing/
    ├── experiment.yaml            # seed, experiment directory and name, logger, description
    ├── agents/ppo.yaml            # agents/<algo>.yaml: the agent's hyperparameters
    └── manifests/box_manifest.yaml

The name is `key=value` segments joined by `__`. `algo` and `manifest` pick agents/<algo>.yaml and
manifests/<manifest>.yaml in the run directory; `seed`, `experiment.experiment_name` and `num_envs`
override the run's experiment.yaml (or, for num_envs, the env default) for that run only.

Each run invokes scripts/train.py in its own process with the interpreter running this script, so
launch the sweep the same way as training (`./isaaclab.sh -p scripts/sweep.py`). Its logs, dumped
params and checkpoints are written inside the run directory. TRAIN_ARGS, any other arguments such
as `--headless`, are passed to every run. `--dry_run` prints each run's command without training.

This module imports nothing from Isaac Lab or moveitmoveit, so run directories can be checked
without launching Isaac.
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

SEGMENT_SEPARATOR = "__"

ALGOS = ("PPO", "AMP")

# The run directory's own parameter files.
EXPERIMENT_FILENAME = "experiment.yaml"
AGENTS_DIRNAME = "agents"
MANIFESTS_DIRNAME = "manifests"

REQUIRED_KEYS = ("algo", "manifest")
# Keys that override the run directory's experiment.yaml.
EXPERIMENT_KEYS = ("seed", "experiment.experiment_name")
NAME_KEYS = REQUIRED_KEYS + ("num_envs",) + EXPERIMENT_KEYS

# What scripts/train.py reads from an experiment file.
REQUIRED_EXPERIMENT_FIELDS = ("seed", "experiment.directory", "experiment.experiment_name", "logger", "description")

_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)*")
_INT_KEYS = {"num_envs": 1, "seed": 0}


class RunError(ValueError):
    """A run directory does not describe a valid run: a bad name or a missing or bad file."""


@dataclass
class RunConfig:
    """What a run directory's name asks for."""

    algo: str
    manifest: str
    num_envs: int | None = None
    # Dotted experiment.yaml key -> value, applied on top of the run's experiment.yaml.
    experiment: dict[str, object] = field(default_factory=dict)


@dataclass
class RunFiles:
    """The parameter files a run trains with, all inside its run directory."""

    experiment: Path
    agent_cfg: Path
    manifest: Path


def _to_int(key: str, value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise RunError(f"{key} must be an integer, got {value!r}") from None
    minimum = _INT_KEYS[key]
    if number < minimum:
        raise RunError(f"{key} must be at least {minimum}, got {number}")
    return number


def parse_run_name(name: str) -> RunConfig:
    """Parse a run directory's name into a RunConfig, or raise RunError saying why not."""
    if not name:
        raise RunError("empty name")

    values: dict[str, str] = {}
    order: list[str] = []
    for segment in name.split(SEGMENT_SEPARATOR):
        if not segment:
            raise RunError("empty segment; segments are separated by a single '__'")
        if "=" not in segment:
            raise RunError(f"segment {segment!r} is not key=value")
        key, raw_value = segment.split("=", 1)
        if "=" in raw_value:
            raise RunError(f"value of {key!r} contains '='; percent-encode it as %3D")
        if not _KEY_PATTERN.fullmatch(key):
            raise RunError(f"{key!r} is not a valid key")
        if key in values:
            raise RunError(f"{key!r} is given more than once")
        if key not in NAME_KEYS:
            raise RunError(f"unknown key {key!r}; expected one of {', '.join(NAME_KEYS)}")
        try:
            values[key] = unquote(raw_value, errors="strict")
        except UnicodeDecodeError:
            raise RunError(f"value of {key!r} is not valid percent-encoded UTF-8") from None
        order.append(key)

    missing = [key for key in REQUIRED_KEYS if key not in values]
    if missing:
        raise RunError(f"missing required key{'s' if len(missing) > 1 else ''}: {', '.join(missing)}")

    algo = values["algo"]
    if algo not in ALGOS:
        raise RunError(f"algo must be one of {', '.join(ALGOS)}, got {algo!r}")
    manifest = values["manifest"]
    if not manifest:
        raise RunError("manifest is empty")
    if "/" in manifest or manifest in (".", ".."):
        raise RunError(f"manifest must be a file name in {MANIFESTS_DIRNAME}/, got {manifest!r}")

    config = RunConfig(algo=algo, manifest=manifest)
    if "num_envs" in values:
        config.num_envs = _to_int("num_envs", values["num_envs"])
    for key in order:
        if key == "seed":
            config.experiment[key] = _to_int(key, values[key])
        elif key == "experiment.experiment_name":
            if not values[key]:
                raise RunError(f"{key} is empty")
            config.experiment[key] = values[key]
    return config


def run_files(run_dir: Path, config: RunConfig) -> RunFiles:
    """The run directory's experiment.yaml, agents/<algo>.yaml and manifests/<manifest>.yaml, or
    a RunError naming the first one that is missing."""
    manifest_name = config.manifest if config.manifest.endswith((".yaml", ".yml")) else f"{config.manifest}.yaml"
    files = RunFiles(
        experiment=run_dir / EXPERIMENT_FILENAME,
        agent_cfg=run_dir / AGENTS_DIRNAME / f"{config.algo.lower()}.yaml",
        manifest=run_dir / MANIFESTS_DIRNAME / manifest_name,
    )
    for path in (files.experiment, files.agent_cfg, files.manifest):
        if not path.is_file():
            raise RunError(f"missing {path.relative_to(run_dir)} in the run directory")
    return files


def _load_yaml(path: Path, run_dir: Path) -> dict:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as error:
        raise RunError(f"can't read {path.relative_to(run_dir)}: {error}") from None
    if not isinstance(data, dict):
        raise RunError(f"{path.relative_to(run_dir)} is not a YAML mapping")
    return data


def check_agent_cfg(files: RunFiles, config: RunConfig, run_dir: Path) -> None:
    """Raise RunError if the agent config names a different trainer than `algo`."""
    agent = _load_yaml(files.agent_cfg, run_dir).get("agent")
    class_type = agent.get("class_type") if isinstance(agent, dict) else None
    if class_type is not None and class_type != config.algo:
        raise RunError(
            f"{files.agent_cfg.relative_to(run_dir)} has agent.class_type {class_type!r}, but algo is {config.algo!r}"
        )


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


def run_experiment(files: RunFiles, config: RunConfig, run_dir: Path) -> dict:
    """The run's experiment.yaml with the name's overrides applied, checked for the fields
    scripts/train.py reads."""
    experiment = apply_experiment_overrides(_load_yaml(files.experiment, run_dir), config.experiment)
    for dotted_key in REQUIRED_EXPERIMENT_FIELDS:
        node = experiment
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise RunError(f"{EXPERIMENT_FILENAME} has no {dotted_key}, and the name doesn't set it")
            node = node[part]
    if not experiment["experiment"]["experiment_name"]:
        raise RunError("the experiment name is empty")
    return experiment


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
    files: RunFiles,
    run_dir: Path,
    experiment_file: Path,
    experiment_name: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    """The scripts/train.py command line for one run."""
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--algo",
        config.algo,
        "--manifest",
        str(files.manifest.resolve()),
        "--name",
        experiment_name,
    ]
    if config.num_envs is not None:
        command += ["--num_envs", str(config.num_envs)]
    command += list(extra_args or [])
    return command


def _train(
    config: RunConfig, files: RunFiles, run_dir: Path, experiment: dict, extra_args: list[str], dry_run: bool
) -> int:
    """Train one run in its own process and return its exit code. The run's experiment, with the
    name's overrides, goes to a temporary file outside the run directory; the run directory's own
    experiment.yaml is not rewritten."""
    experiment_name = experiment["experiment"]["experiment_name"]
    if dry_run:
        command = build_train_command(
            config, files, run_dir, Path("<experiment.yaml below>"), experiment_name, extra_args
        )
        print(f"[sweep] would run: {' '.join(command)}", flush=True)
        print("[sweep] with experiment.yaml:", flush=True)
        print(yaml.safe_dump(experiment, sort_keys=False).rstrip(), flush=True)
        return 0

    fd, experiment_path = tempfile.mkstemp(prefix="sweep-experiment-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(experiment, f, sort_keys=False)
        command = build_train_command(config, files, run_dir, Path(experiment_path), experiment_name, extra_args)
        print(f"[sweep] {' '.join(command)}", flush=True)
        return subprocess.run(command).returncode
    finally:
        os.unlink(experiment_path)


def run_sweep(root: Path, force: bool = False, extra_args: list[str] | None = None, dry_run: bool = False) -> int:
    """Train every run under `root` in turn and return the number of runs that failed."""
    extra_args = list(extra_args or [])
    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        print(f"[sweep] No run directories under {root}")
        return 0

    trained, skipped, failed = [], [], []
    for index, run_dir in enumerate(run_dirs, start=1):
        name = run_dir.name
        print(f"\n[sweep] ({index}/{len(run_dirs)}) {name}", flush=True)
        try:
            config = parse_run_name(name)
            files = run_files(run_dir, config)
            check_agent_cfg(files, config, run_dir)
            experiment = run_experiment(files, config, run_dir)
        except RunError as error:
            print(f"[sweep] ERROR {name}: {error}", flush=True)
            failed.append(name)
            continue

        finished = finished_checkpoints(run_dir)
        if finished and not force:
            print(f"[sweep] SKIP {name}: already finished ({finished[-1]}); pass --force to rerun", flush=True)
            skipped.append(name)
            continue

        returncode = _train(config, files, run_dir, experiment, extra_args, dry_run)
        if dry_run:
            trained.append(name)
        elif returncode != 0:
            print(f"[sweep] ERROR {name}: training exited with code {returncode}", flush=True)
            failed.append(name)
        elif not set(finished_checkpoints(run_dir)) - set(finished):
            print(f"[sweep] ERROR {name}: training ended without a final checkpoint", flush=True)
            failed.append(name)
        else:
            trained.append(name)

    verb = "would train" if dry_run else "trained"
    print(f"\n[sweep] {len(trained)} {verb}, {len(skipped)} skipped, {len(failed)} failed")
    for name in failed:
        print(f"[sweep]   failed: {name}")
    return len(failed)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # The root, if given, comes first, so a train.py flag's value (`--device cuda:1`) is never
    # taken for it.
    root = argv.pop(0) if argv and not argv[0].startswith("-") else "sweep"

    parser = argparse.ArgumentParser(
        usage="%(prog)s [ROOT] [--force] [--dry_run] [TRAIN_ARGS ...]",
        description="Train once for every run directory under ROOT (default: sweep), with the parameter "
        "files inside it.",
        epilog="Any other argument, such as --headless, is passed to every scripts/train.py run.",
        allow_abbrev=False,
    )
    parser.add_argument("--force", action="store_true", help="Rerun runs that already have a finished checkpoint.")
    parser.add_argument("--dry_run", action="store_true", help="Check each run and print its command; train nothing.")
    args, extra_args = parser.parse_known_args(argv)
    # `--` separates TRAIN_ARGS when running with plain python; it isn't one of them.
    extra_args = [arg for arg in extra_args if arg != "--"]

    try:
        failures = run_sweep(Path(root), force=args.force, extra_args=extra_args, dry_run=args.dry_run)
    except FileNotFoundError as error:
        print(f"[sweep] {error}", file=sys.stderr)
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
