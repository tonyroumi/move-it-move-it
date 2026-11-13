"""`train.py --name` sets the experiment name used in the log path (no simulator)."""

import os

from moveitmoveit.utils.paths import train_log_root


def test_name_replaces_yaml_experiment_name_in_log_root():
    experiment = {"directory": "locomotion", "experiment_name": "command_conditioned_walking"}

    log_root = train_log_root(experiment, "my_run", "AMP")

    assert experiment["experiment_name"] == "my_run"
    assert log_root == os.path.abspath(os.path.join("logs", "locomotion", "my_run", "AMP"))
    assert "command_conditioned_walking" not in log_root
