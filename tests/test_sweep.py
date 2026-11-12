"""Tests for scripts/sweep.py: run-name parsing, sweep-root discovery and the sweep loop, with
training replaced by a fake so Isaac is never launched."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_SWEEP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sweep.py"
_spec = importlib.util.spec_from_file_location("sweep", _SWEEP_PATH)
sweep = importlib.util.module_from_spec(_spec)
sys.modules["sweep"] = sweep
_spec.loader.exec_module(sweep)


# --- name parsing ---------------------------------------------------------------------------


def test_minimal_name():
    config = sweep.parse_run_name("algo=AMP__manifest=goal_manifest")
    assert config.algo == "AMP"
    assert config.manifest == str(sweep.MANIFESTS_DIR / "goal_manifest.yaml")
    assert config.num_envs is None
    assert config.checkpoint is None
    assert config.experiment == {}
    assert config.overrides == []


def test_all_key_kinds():
    config = sweep.parse_run_name(
        "algo=PPO__manifest=boxing_manifest.yaml__num_envs=512__seed=7"
        "__experiment.experiment_name=walk_lr__logger.backend=none__logger.write_interval=5"
        "__logger.checkpoint_interval=50__agent.agent.learning_rate=1e-4__env.episode_length_s=5.0"
        "__presets=newton"
    )
    assert config.algo == "PPO"
    assert config.manifest == str(sweep.MANIFESTS_DIR / "boxing_manifest.yaml")
    assert config.num_envs == 512
    assert config.experiment == {
        "seed": 7,
        "experiment.experiment_name": "walk_lr",
        "logger.backend": "none",
        "logger.write_interval": 5,
        "logger.checkpoint_interval": 50,
    }
    assert config.overrides == [
        "agent.agent.learning_rate=1e-4",
        "env.episode_length_s=5.0",
        "presets=newton",
    ]


def test_segment_order_does_not_matter():
    a = sweep.parse_run_name("algo=AMP__manifest=goal_manifest__seed=3")
    b = sweep.parse_run_name("seed=3__manifest=goal_manifest__algo=AMP")
    assert a == b


def test_percent_encoded_values():
    config = sweep.parse_run_name(
        "algo=AMP__manifest=configs%2Fmanifests%2Fdance_manifest.yaml"
        "__checkpoint=logs%2Frun%2Fcheckpoints%2Fbest_agent.pt"
        "__description=lr%3D1e-4%5F%5Fwide%20net__experiment.experiment_name=a%5F%5Fb"
    )
    # A value with a `/` is a path, used as given.
    assert config.manifest == "configs/manifests/dance_manifest.yaml"
    assert config.checkpoint == "logs/run/checkpoints/best_agent.pt"
    assert config.experiment["description"] == "lr=1e-4__wide net"
    assert config.experiment["experiment.experiment_name"] == "a__b"


def test_percent_encoded_hydra_value():
    config = sweep.parse_run_name("algo=AMP__manifest=goal_manifest__env.some.path=a%2Fb%3Dc")
    assert config.overrides == ["env.some.path=a/b=c"]


def test_value_ending_in_underscore():
    with pytest.raises(sweep.RunNameError, match="'_seed' is not a valid key"):
        sweep.parse_run_name("algo=AMP__manifest=goal_manifest__experiment.experiment_name=run___seed=1")
    config = sweep.parse_run_name("algo=AMP__manifest=goal_manifest__experiment.experiment_name=run%5F__seed=1")
    assert config.experiment == {"experiment.experiment_name": "run_", "seed": 1}


def test_empty_description_is_allowed():
    config = sweep.parse_run_name("algo=AMP__manifest=goal_manifest__description=")
    assert config.experiment == {"description": ""}


@pytest.mark.parametrize(
    "name, reason",
    [
        ("manifest=goal_manifest", "missing required key: algo"),
        ("algo=AMP", "missing required key: manifest"),
        ("seed=1", "missing required keys: algo, manifest"),
        ("algo=AMP__manifest=goal_manifest__seed=x", "seed must be an integer"),
        ("algo=AMP__manifest=goal_manifest__num_envs=0", "num_envs must be at least 1"),
        ("algo=SAC__manifest=goal_manifest", "algo must be one of PPO, AMP"),
        ("algo=AMP__manifest=goal_manifest__logger.backend=csv", "logger.backend must be one of"),
        ("algo=AMP__manifest=goal_manifest__lr=1", "unknown key 'lr'"),
        ("algo=AMP__manifest=goal_manifest__logger.flush=1", "unknown key 'logger.flush'"),
        ("algo=AMP__manifest=goal_manifest__seed", "segment 'seed' is not key=value"),
        ("algo=AMP__manifest=goal_manifest____seed=1", "empty segment"),
        ("algo=AMP__algo=PPO__manifest=goal_manifest", "'algo' is given more than once"),
        ("algo=AMP__manifest=goal_manifest__description=a=b", "contains '='; percent-encode it as %3D"),
        ("algo=AMP__manifest=", "manifest is empty"),
        ("algo=AMP__manifest=goal_manifest__experiment.experiment_name=", "experiment.experiment_name is empty"),
        ("algo=AMP__manifest=goal_manifest__description=%FF", "not valid percent-encoded UTF-8"),
        ("my_first_run", "is not key=value"),
        ("", "empty name"),
    ],
)
def test_bad_names_raise_with_reason(name, reason):
    with pytest.raises(sweep.RunNameError, match=reason.replace("(", r"\(").replace(".", r"\.")):
        sweep.parse_run_name(name)


# --- experiment overrides and the train command ---------------------------------------------


def test_experiment_overrides_merge_onto_base_without_mutating_it():
    base = {
        "seed": 42,
        "experiment": {"directory": "locomotion", "experiment_name": "walk"},
        "logger": {"backend": "tensorboard", "write_interval": 1, "checkpoint_interval": 200},
        "description": "",
    }
    merged = sweep.apply_experiment_overrides(
        base, {"seed": 7, "experiment.experiment_name": "sweep_a", "logger.write_interval": 10}
    )
    assert merged == {
        "seed": 7,
        "experiment": {"directory": "locomotion", "experiment_name": "sweep_a"},
        "logger": {"backend": "tensorboard", "write_interval": 10, "checkpoint_interval": 200},
        "description": "",
    }
    assert base["seed"] == 42 and base["experiment"]["experiment_name"] == "walk"


def test_train_command(tmp_path):
    config = sweep.parse_run_name(
        "algo=AMP__manifest=goal_manifest__num_envs=64__checkpoint=a%2Fb.pt__agent.runner.timesteps=1000"
    )
    run_dir = tmp_path / "run"
    command = sweep.build_train_command(config, run_dir, tmp_path / "exp.yaml", ["--headless"])
    assert command == [
        sys.executable,
        str(sweep.TRAIN_SCRIPT),
        "--algo", "AMP",
        "--manifest", str(sweep.MANIFESTS_DIR / "goal_manifest.yaml"),
        "--experiment", str(tmp_path / "exp.yaml"),
        "--log_root", str(run_dir.resolve()),
        "--num_envs", "64",
        "--checkpoint", "a/b.pt",
        "--headless",
        "agent.runner.timesteps=1000",
    ]  # fmt: skip


# --- sweep-directory discovery --------------------------------------------------------------


def test_discover_run_dirs_lists_immediate_child_dirs_sorted(tmp_path):
    for name in ["b=2", "a=1", ".hidden", "c=3"]:
        (tmp_path / name).mkdir()
    (tmp_path / "a=1" / "nested=dir").mkdir()
    (tmp_path / "notes.txt").write_text("not a run")
    assert [p.name for p in sweep.discover_run_dirs(tmp_path)] == ["a=1", "b=2", "c=3"]


def test_discover_run_dirs_empty_root(tmp_path):
    assert sweep.discover_run_dirs(tmp_path) == []


def test_discover_run_dirs_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        sweep.discover_run_dirs(tmp_path / "nope")


def _write_log_dir(run_dir: Path, timesteps: int, checkpoints: list[str]) -> Path:
    log_dir = run_dir / "locomotion" / "walk" / "AMP" / f"2026-01-01_00-00-{len(list(run_dir.rglob('params')))}"
    (log_dir / "params").mkdir(parents=True)
    (log_dir / "params" / "agent.yaml").write_text(yaml.safe_dump({"runner": {"timesteps": timesteps}}))
    (log_dir / "checkpoints").mkdir()
    for name in checkpoints:
        (log_dir / "checkpoints" / name).write_bytes(b"")
    return log_dir


def test_finished_checkpoints(tmp_path):
    assert sweep.finished_checkpoints(tmp_path) == []
    # Periodic and best checkpoints are not a finished run.
    _write_log_dir(tmp_path, 1000, ["200.pt", "best_agent.pt"])
    assert sweep.finished_checkpoints(tmp_path) == []
    log_dir = _write_log_dir(tmp_path, 1000, ["200.pt", "1000.pt"])
    assert sweep.finished_checkpoints(tmp_path) == [log_dir / "checkpoints" / "1000.pt"]


# --- the sweep loop, with training faked ----------------------------------------------------


@pytest.fixture
def fake_training(monkeypatch):
    """Replace the train.py subprocess with one that records the call and writes what a finished
    run would. `outcome` sets what the next fake run does."""
    calls = []
    outcome = {"returncode": 0, "finish": True}

    def fake_run(command):
        args = dict(zip(command[2::2], command[3::2]))
        with open(args["--experiment"]) as f:
            experiment = yaml.safe_load(f)
        calls.append({"command": command, "experiment": experiment})
        if outcome["finish"]:
            _write_log_dir(Path(args["--log_root"]), 1000, ["1000.pt"])

        class Completed:
            returncode = outcome["returncode"]

        return Completed()

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    return calls, outcome


def test_sweep_trains_each_run_and_continues_past_bad_names(tmp_path, fake_training, capsys):
    calls, _ = fake_training
    shared_experiment = sweep.EXPERIMENT_FILE.read_text()
    for name in [
        "algo=AMP__manifest=goal_manifest__seed=1",
        "algo=PPO__manifest=goal_manifest__seed=2__experiment.experiment_name=ppo_run",
        "manifest=goal_manifest",
        "algo=AMP__manifest=no_such_manifest",
    ]:
        (tmp_path / name).mkdir()

    failures = sweep.run_sweep(tmp_path, extra_args=["--headless"])

    out = capsys.readouterr().out
    assert failures == 2
    assert "ERROR manifest=goal_manifest: missing required key: algo" in out
    assert "ERROR algo=AMP__manifest=no_such_manifest: manifest not found" in out
    assert "2 trained, 0 skipped, 2 failed" in out
    assert [call["experiment"]["seed"] for call in calls] == [1, 2]
    assert calls[1]["experiment"]["experiment"]["experiment_name"] == "ppo_run"
    assert all("--headless" in call["command"] for call in calls)
    # Each run's results land inside its own directory.
    for call in calls:
        log_root = Path(call["command"][call["command"].index("--log_root") + 1])
        assert log_root.parent == tmp_path.resolve()
        assert sweep.finished_checkpoints(log_root)
    # The shared experiment file is untouched.
    assert sweep.EXPERIMENT_FILE.read_text() == shared_experiment


def test_sweep_skips_finished_runs_unless_forced(tmp_path, fake_training, capsys):
    calls, _ = fake_training
    (tmp_path / "algo=AMP__manifest=goal_manifest").mkdir()

    assert sweep.run_sweep(tmp_path) == 0
    assert len(calls) == 1

    assert sweep.run_sweep(tmp_path) == 0
    assert len(calls) == 1
    assert "SKIP algo=AMP__manifest=goal_manifest: already finished" in capsys.readouterr().out

    assert sweep.run_sweep(tmp_path, force=True) == 0
    assert len(calls) == 2


def test_sweep_reports_failed_and_unfinished_training(tmp_path, fake_training, capsys):
    calls, outcome = fake_training
    (tmp_path / "algo=AMP__manifest=goal_manifest").mkdir()

    outcome.update(returncode=1, finish=False)
    assert sweep.run_sweep(tmp_path) == 1
    assert "training exited with code 1" in capsys.readouterr().out

    # Exit code 0 without a final checkpoint (e.g. train.py interrupted) is not a finished run.
    outcome.update(returncode=0, finish=False)
    assert sweep.run_sweep(tmp_path) == 1
    assert "training ended without a final checkpoint" in capsys.readouterr().out
    assert len(calls) == 2


def test_main_missing_root(tmp_path, capsys):
    assert sweep.main([str(tmp_path / "nope")]) == 2
    assert "Sweep root is not a directory" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, root, force, extra_args",
    [
        ([], "sweep", False, []),
        (["my_sweep"], "my_sweep", False, []),
        (["my_sweep", "--force", "--headless"], "my_sweep", True, ["--headless"]),
        (["--headless", "--device", "cuda:1"], "sweep", False, ["--headless", "--device", "cuda:1"]),
        (["my_sweep", "--device", "cuda:1", "--force"], "my_sweep", True, ["--device", "cuda:1"]),
        (["my_sweep", "--", "--headless"], "my_sweep", False, ["--headless"]),
    ],
)
def test_main_arguments(monkeypatch, argv, root, force, extra_args):
    seen = {}

    def fake_run_sweep(root, force=False, extra_args=None):
        seen.update(root=root, force=force, extra_args=extra_args)
        return 0

    monkeypatch.setattr(sweep, "run_sweep", fake_run_sweep)
    assert sweep.main(argv) == 0
    assert seen == {"root": Path(root), "force": force, "extra_args": extra_args}
