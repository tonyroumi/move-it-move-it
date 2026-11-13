"""scripts/sweep.py: run-name parsing, each run directory's parameter files, sweep-root discovery
and the sweep loop, with training replaced by a fake. No simulator."""

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

EXPERIMENT = {
    "seed": 42,
    "experiment": {"directory": "locomotion", "experiment_name": "command_conditioned_walking"},
    "logger": {"backend": "tensorboard", "write_interval": 1, "checkpoint_interval": 200},
    "description": "",
}
NAME = "algo=PPO__manifest=box_manifest__num_envs=4096__seed=1__experiment.experiment_name=boxing"


def make_run_dir(root: Path, name: str, algo: str = "ppo", manifest: str = "box_manifest", experiment=None,
                 class_type: str | None = "PPO") -> Path:
    """A run directory laid out like the captain's: experiment.yaml, agents/<algo>.yaml and
    manifests/<manifest>.yaml."""
    run_dir = root / name
    (run_dir / "agents").mkdir(parents=True)
    (run_dir / "manifests").mkdir()
    (run_dir / "experiment.yaml").write_text(yaml.safe_dump(EXPERIMENT if experiment is None else experiment))
    agent = {"agent": {"class_type": class_type}} if class_type else {"models": {}}
    (run_dir / "agents" / f"{algo}.yaml").write_text(yaml.safe_dump(agent))
    (run_dir / "manifests" / f"{manifest}.yaml").write_text(yaml.safe_dump({"motions": {}}))
    return run_dir


# --- name parsing ---------------------------------------------------------------------------


def test_full_name():
    config = sweep.parse_run_name(NAME)
    assert config.algo == "PPO"
    assert config.manifest == "box_manifest"
    assert config.num_envs == 4096
    assert config.experiment == {"seed": 1, "experiment.experiment_name": "boxing"}


def test_minimal_name():
    config = sweep.parse_run_name("algo=AMP__manifest=goal")
    assert (config.algo, config.manifest, config.num_envs, config.experiment) == ("AMP", "goal", None, {})


def test_segment_order_does_not_matter():
    assert sweep.parse_run_name("seed=3__manifest=m__algo=AMP") == sweep.parse_run_name("algo=AMP__manifest=m__seed=3")


def test_percent_encoded_values():
    config = sweep.parse_run_name(
        "algo=AMP__manifest=jump%5F%5Fv2__experiment.experiment_name=lr%3D1e-4%2Fwide"
    )
    assert config.manifest == "jump__v2"
    assert config.experiment == {"experiment.experiment_name": "lr=1e-4/wide"}


def test_value_ending_in_underscore():
    with pytest.raises(sweep.RunError, match="'_seed' is not a valid key"):
        sweep.parse_run_name("algo=AMP__manifest=m__experiment.experiment_name=run___seed=1")
    config = sweep.parse_run_name("algo=AMP__manifest=m__experiment.experiment_name=run%5F__seed=1")
    assert config.experiment == {"experiment.experiment_name": "run_", "seed": 1}


@pytest.mark.parametrize(
    "name, reason",
    [
        ("manifest=m", "missing required key: algo"),
        ("algo=AMP", "missing required key: manifest"),
        ("seed=1", "missing required keys: algo, manifest"),
        ("algo=AMP__manifest=m__seed=x", "seed must be an integer"),
        ("algo=AMP__manifest=m__num_envs=0", "num_envs must be at least 1"),
        ("algo=SAC__manifest=m", "algo must be one of PPO, AMP"),
        ("algo=AMP__manifest=m__lr=1", "unknown key 'lr'"),
        ("algo=AMP__manifest=m__logger.backend=none", "unknown key 'logger.backend'"),
        ("algo=AMP__manifest=m__seed", "segment 'seed' is not key=value"),
        ("algo=AMP__manifest=m____seed=1", "empty segment"),
        ("algo=AMP__algo=PPO__manifest=m", "'algo' is given more than once"),
        ("algo=AMP__manifest=m__experiment.experiment_name=a=b", "contains '='"),
        ("algo=AMP__manifest=", "manifest is empty"),
        ("algo=AMP__manifest=configs%2Fmanifests%2Fm.yaml", "manifest must be a file name in manifests/"),
        ("algo=AMP__manifest=m__experiment.experiment_name=", "experiment.experiment_name is empty"),
        ("algo=AMP__manifest=m__experiment.experiment_name=%FF", "not valid percent-encoded UTF-8"),
        ("my_first_run", "is not key=value"),
        ("", "empty name"),
    ],
)
def test_bad_names_raise_with_reason(name, reason):
    with pytest.raises(sweep.RunError, match=reason.replace("(", r"\(").replace(".", r"\.")):
        sweep.parse_run_name(name)


# --- the run directory's files --------------------------------------------------------------


def test_run_files_come_from_the_run_directory(tmp_path):
    run_dir = make_run_dir(tmp_path, NAME)
    files = sweep.run_files(run_dir, sweep.parse_run_name(NAME))
    assert files.experiment == run_dir / "experiment.yaml"
    assert files.agent_cfg == run_dir / "agents" / "ppo.yaml"
    assert files.manifest == run_dir / "manifests" / "box_manifest.yaml"


def test_manifest_with_yaml_suffix(tmp_path):
    name = "algo=PPO__manifest=box_manifest.yaml"
    run_dir = make_run_dir(tmp_path, name)
    assert sweep.run_files(run_dir, sweep.parse_run_name(name)).manifest == run_dir / "manifests" / "box_manifest.yaml"


@pytest.mark.parametrize(
    "missing, reason",
    [
        ("experiment.yaml", "missing experiment.yaml"),
        ("agents/ppo.yaml", "missing agents/ppo.yaml"),
        ("manifests/box_manifest.yaml", "missing manifests/box_manifest.yaml"),
    ],
)
def test_missing_run_file(tmp_path, missing, reason):
    run_dir = make_run_dir(tmp_path, NAME)
    (run_dir / missing).unlink()
    with pytest.raises(sweep.RunError, match=reason):
        sweep.run_files(run_dir, sweep.parse_run_name(NAME))


def test_algo_picks_the_agent_file(tmp_path):
    # algo=AMP needs agents/amp.yaml; the run directory only has ppo.yaml.
    name = "algo=AMP__manifest=box_manifest"
    run_dir = make_run_dir(tmp_path, name)
    with pytest.raises(sweep.RunError, match="missing agents/amp.yaml"):
        sweep.run_files(run_dir, sweep.parse_run_name(name))


def test_agent_class_type_must_match_algo(tmp_path):
    run_dir = make_run_dir(tmp_path, NAME, class_type="AMP")
    config = sweep.parse_run_name(NAME)
    with pytest.raises(sweep.RunError, match="agent.class_type 'AMP', but algo is 'PPO'"):
        sweep.check_agent_cfg(sweep.run_files(run_dir, config), config, run_dir)


def test_agent_without_class_type_is_accepted(tmp_path):
    run_dir = make_run_dir(tmp_path, NAME, class_type=None)
    config = sweep.parse_run_name(NAME)
    sweep.check_agent_cfg(sweep.run_files(run_dir, config), config, run_dir)


def test_name_overrides_the_run_experiment(tmp_path):
    run_dir = make_run_dir(tmp_path, NAME)
    config = sweep.parse_run_name(NAME)
    experiment = sweep.run_experiment(sweep.run_files(run_dir, config), config, run_dir)
    assert experiment == {
        **EXPERIMENT,
        "seed": 1,
        "experiment": {"directory": "locomotion", "experiment_name": "boxing"},
    }
    # The run directory's own experiment.yaml is left as it was.
    assert yaml.safe_load((run_dir / "experiment.yaml").read_text()) == EXPERIMENT


def test_run_experiment_without_overrides(tmp_path):
    name = "algo=PPO__manifest=box_manifest"
    run_dir = make_run_dir(tmp_path, name)
    config = sweep.parse_run_name(name)
    assert sweep.run_experiment(sweep.run_files(run_dir, config), config, run_dir) == EXPERIMENT


def test_run_experiment_missing_field(tmp_path):
    experiment = {key: value for key, value in EXPERIMENT.items() if key != "description"}
    run_dir = make_run_dir(tmp_path, NAME, experiment=experiment)
    config = sweep.parse_run_name(NAME)
    with pytest.raises(sweep.RunError, match="experiment.yaml has no description"):
        sweep.run_experiment(sweep.run_files(run_dir, config), config, run_dir)


def test_run_experiment_not_a_mapping(tmp_path):
    run_dir = make_run_dir(tmp_path, NAME)
    (run_dir / "experiment.yaml").write_text("- a list\n")
    config = sweep.parse_run_name(NAME)
    with pytest.raises(sweep.RunError, match="experiment.yaml is not a YAML mapping"):
        sweep.run_experiment(sweep.run_files(run_dir, config), config, run_dir)


def test_apply_experiment_overrides_does_not_mutate_base():
    merged = sweep.apply_experiment_overrides(EXPERIMENT, {"seed": 7, "experiment.experiment_name": "a"})
    assert merged["seed"] == 7 and merged["experiment"]["experiment_name"] == "a"
    assert EXPERIMENT["seed"] == 42 and EXPERIMENT["experiment"]["experiment_name"] == "command_conditioned_walking"


def test_train_command(tmp_path):
    run_dir = make_run_dir(tmp_path, NAME)
    config = sweep.parse_run_name(NAME)
    files = sweep.run_files(run_dir, config)
    command = sweep.build_train_command(config, files, run_dir, tmp_path / "exp.yaml", "boxing", ["--headless"])
    assert command == [
        sys.executable,
        str(sweep.TRAIN_SCRIPT),
        "--algo", "PPO",
        "--manifest", str((run_dir / "manifests" / "box_manifest.yaml").resolve()),
        "--agent_cfg", str((run_dir / "agents" / "ppo.yaml").resolve()),
        "--experiment", str(tmp_path / "exp.yaml"),
        "--name", "boxing",
        "--log_root", str(run_dir.resolve()),
        "--num_envs", "4096",
        "--headless",
    ]  # fmt: skip


# --- sweep-directory discovery --------------------------------------------------------------


def test_discover_run_dirs_lists_immediate_child_dirs_sorted(tmp_path):
    for name in ["b=2", "a=1", ".hidden", "c=3"]:
        (tmp_path / name).mkdir()
    (tmp_path / "a=1" / "agents").mkdir()
    (tmp_path / "README.txt").write_text("not a run")
    assert [p.name for p in sweep.discover_run_dirs(tmp_path)] == ["a=1", "b=2", "c=3"]


def test_discover_run_dirs_empty_root(tmp_path):
    assert sweep.discover_run_dirs(tmp_path) == []


def test_discover_run_dirs_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        sweep.discover_run_dirs(tmp_path / "nope")


def _write_log_dir(run_dir: Path, timesteps: int, checkpoints: list[str]) -> Path:
    log_dir = run_dir / "locomotion" / "boxing" / "PPO" / f"2026-01-01_00-00-{len(list(run_dir.rglob('params')))}"
    (log_dir / "params").mkdir(parents=True)
    (log_dir / "params" / "agent.yaml").write_text(yaml.safe_dump({"runner": {"timesteps": timesteps}}))
    (log_dir / "checkpoints").mkdir()
    for name in checkpoints:
        (log_dir / "checkpoints" / name).write_bytes(b"")
    return log_dir


def test_finished_checkpoints(tmp_path):
    run_dir = make_run_dir(tmp_path, NAME)
    assert sweep.finished_checkpoints(run_dir) == []
    # Periodic and best checkpoints are not a finished run.
    _write_log_dir(run_dir, 1000, ["200.pt", "best_agent.pt"])
    assert sweep.finished_checkpoints(run_dir) == []
    log_dir = _write_log_dir(run_dir, 1000, ["200.pt", "1000.pt"])
    assert sweep.finished_checkpoints(run_dir) == [log_dir / "checkpoints" / "1000.pt"]


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
        calls.append({"args": args, "command": command, "experiment": experiment})
        if outcome["finish"]:
            _write_log_dir(Path(args["--log_root"]), 1000, ["1000.pt"])

        class Completed:
            returncode = outcome["returncode"]

        return Completed()

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    return calls, outcome


def test_sweep_trains_each_run_from_its_own_files(tmp_path, fake_training, capsys):
    calls, _ = fake_training
    good = [
        make_run_dir(tmp_path, NAME),
        make_run_dir(tmp_path, "algo=AMP__manifest=jump_manifest__seed=2", algo="amp", manifest="jump_manifest",
                     class_type="AMP"),
    ]
    (tmp_path / "manifest=box_manifest").mkdir()
    make_run_dir(tmp_path, "algo=PPO__manifest=salsa_manifest")  # holds box_manifest.yaml, not salsa

    failures = sweep.run_sweep(tmp_path, extra_args=["--headless"])

    out = capsys.readouterr().out
    assert failures == 2
    assert "ERROR manifest=box_manifest: missing required key: algo" in out
    assert "ERROR algo=PPO__manifest=salsa_manifest: missing manifests/salsa_manifest.yaml" in out
    assert "2 trained, 0 skipped, 2 failed" in out

    amp, ppo = calls  # name order
    assert ppo["args"]["--agent_cfg"] == str((good[0] / "agents" / "ppo.yaml").resolve())
    assert ppo["args"]["--manifest"] == str((good[0] / "manifests" / "box_manifest.yaml").resolve())
    assert ppo["args"]["--name"] == "boxing"
    assert ppo["experiment"]["seed"] == 1
    assert amp["args"]["--agent_cfg"] == str((good[1] / "agents" / "amp.yaml").resolve())
    # No experiment_name in the name: the run's experiment.yaml names it.
    assert amp["args"]["--name"] == "command_conditioned_walking"
    assert amp["experiment"]["seed"] == 2
    for call, run_dir in ((ppo, good[0]), (amp, good[1])):
        assert call["args"]["--log_root"] == str(run_dir.resolve())
        assert "--headless" in call["command"]
        assert sweep.finished_checkpoints(run_dir)
        # The temporary experiment file is outside the run directory and gone afterwards.
        assert not Path(call["args"]["--experiment"]).exists()
        assert run_dir.resolve() not in Path(call["args"]["--experiment"]).parents
        assert yaml.safe_load((run_dir / "experiment.yaml").read_text()) == EXPERIMENT


def test_sweep_skips_finished_runs_unless_forced(tmp_path, fake_training, capsys):
    calls, _ = fake_training
    make_run_dir(tmp_path, NAME)

    assert sweep.run_sweep(tmp_path) == 0
    assert len(calls) == 1

    assert sweep.run_sweep(tmp_path) == 0
    assert len(calls) == 1
    assert f"SKIP {NAME}: already finished" in capsys.readouterr().out

    assert sweep.run_sweep(tmp_path, force=True) == 0
    assert len(calls) == 2


def test_sweep_reports_failed_and_unfinished_training(tmp_path, fake_training, capsys):
    calls, outcome = fake_training
    make_run_dir(tmp_path, NAME)

    outcome.update(returncode=1, finish=False)
    assert sweep.run_sweep(tmp_path) == 1
    assert "training exited with code 1" in capsys.readouterr().out

    # Exit code 0 without a final checkpoint (e.g. train.py interrupted) is not a finished run.
    outcome.update(returncode=0, finish=False)
    assert sweep.run_sweep(tmp_path) == 1
    assert "training ended without a final checkpoint" in capsys.readouterr().out
    assert len(calls) == 2


def test_dry_run_trains_nothing(tmp_path, monkeypatch, capsys):
    def no_run(command):
        raise AssertionError("dry run launched training")

    monkeypatch.setattr(sweep.subprocess, "run", no_run)
    make_run_dir(tmp_path, NAME)
    assert sweep.run_sweep(tmp_path, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "would run:" in out and "--name boxing" in out and "--num_envs 4096" in out
    assert "seed: 1" in out
    assert "1 would train, 0 skipped, 0 failed" in out


def test_main_missing_root(tmp_path, capsys):
    assert sweep.main([str(tmp_path / "nope")]) == 2
    assert "Sweep root is not a directory" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, root, force, dry_run, extra_args",
    [
        ([], "sweep", False, False, []),
        (["my_sweep"], "my_sweep", False, False, []),
        (["my_sweep", "--force", "--headless"], "my_sweep", True, False, ["--headless"]),
        (["--dry_run"], "sweep", False, True, []),
        (["--headless", "--device", "cuda:1"], "sweep", False, False, ["--headless", "--device", "cuda:1"]),
        (["my_sweep", "--device", "cuda:1", "--force"], "my_sweep", True, False, ["--device", "cuda:1"]),
        (["my_sweep", "--", "--headless"], "my_sweep", False, False, ["--headless"]),
    ],
)
def test_main_arguments(monkeypatch, argv, root, force, dry_run, extra_args):
    seen = {}

    def fake_run_sweep(root, force=False, extra_args=None, dry_run=False):
        seen.update(root=root, force=force, dry_run=dry_run, extra_args=extra_args)
        return 0

    monkeypatch.setattr(sweep, "run_sweep", fake_run_sweep)
    assert sweep.main(argv) == 0
    assert seen == {"root": Path(root), "force": force, "dry_run": dry_run, "extra_args": extra_args}
