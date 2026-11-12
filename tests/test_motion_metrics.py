"""The training log's motion sample histogram and behavior completion counts. No simulator."""

from pathlib import Path

import numpy as np
import pytest
import torch

from moveitmoveit.behavior import LOCOMOTION, BehaviorManager, settled
from moveitmoveit.motion_metrics import MotionMetrics
from moveitmoveit.motion_set import MotionSet
from moveitmoveit.utils.logger import Logger
from test_loop import NUM_DOFS, write_clip, write_manifest

HISTOGRAM = "Motion Sample Histogram"


class RecordingLogger(Logger):
    """A logger with no backend that keeps what it was asked to log."""

    def __init__(self):
        super().__init__(backend="none")
        self.histograms = []
        self.scalars = []

    def log_histogram(self, tag, values, step, bins=None):
        self.histograms.append((tag, np.asarray(values).tolist(), step, np.asarray(bins).tolist()))

    def log_scalars(self, metrics, step):
        self.scalars.append((dict(metrics), step))


def make_motions(tmp_path: Path) -> MotionSet:
    """walk, run, an idle motion with sample_prob 0 that is never drawn, and two 1 s behaviors: a jab
    that completes at phase 1 and a kick that also settles once it is half done and still."""
    walk = write_clip(tmp_path / "walk.npz", 21)
    move = write_clip(tmp_path / "move.npz", 11)
    manifest = write_manifest(
        tmp_path,
        {
            "walk": {"file": walk, "sample_prob": 0.4},
            "run": {"file": walk, "sample_prob": 0.2},
            "idle": {"file": walk, "sample_prob": 0.0},
            "jab": {"file": move, "sample_prob": 0.2, "trigger": "binary_key_0"},
            "kick": {
                "file": move,
                "sample_prob": 0.2,
                "trigger": "binary_key_1",
                "settle": {"min_phase": 0.5, "max_dof_speed": 0.1},
            },
        },
    )
    return MotionSet.from_manifest(manifest, "cpu")


@pytest.fixture
def motions(tmp_path) -> MotionSet:
    return make_motions(tmp_path)


def logged(motions: MotionSet, behaviors: BehaviorManager | None = None) -> RecordingLogger:
    logger = RecordingLogger()
    behaviors = behaviors or BehaviorManager(1, motions.num_behaviors, "cpu")
    logger.add_writer(MotionMetrics(motions, behaviors).write)
    return logger


def script_draws(monkeypatch, motions: MotionSet, draws: list[tuple[str, list[str]]]):
    """Make the samplers return `draws` in order: ("all" or "locomotion", motion names)."""
    remaining = list(draws)

    def multinomial(probs, num_samples, replacement):
        pool, names = remaining.pop(0)
        expected = motions.sample_prob if pool == "all" else motions.locomotion_sample_prob
        assert torch.equal(probs, expected) and replacement and num_samples == len(names)
        return torch.tensor([motions.name_to_id[name] for name in names])

    monkeypatch.setattr(torch, "multinomial", multinomial)
    return remaining


def test_histogram_counts_resets_return_motions_and_discriminator_batches(motions, monkeypatch):
    """Two resets into the jab draw a walk and a run to return to; a discriminator batch draws four
    more. Every draw counts, and idle, never drawn, still has a bucket and a 0."""
    remaining = script_draws(
        monkeypatch,
        motions,
        [
            ("all", ["walk", "jab", "jab", "run", "walk"]),
            ("locomotion", ["run", "walk"]),
            ("all", ["kick", "walk", "walk", "walk"]),
        ],
    )
    logger = logged(motions)

    motions.sample_reset(5)
    motions.sample_ids(4)  # what ReferenceMotions.style_batch draws for the discriminator
    assert not remaining
    logger.write_data()

    assert motions.sample_counts.tolist() == [6, 2, 0, 2, 1]
    ((tag, values, step, bins),) = logger.histograms
    assert tag == HISTOGRAM and step == logger.timestep
    assert bins == [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    assert np.histogram(values, bins)[0].tolist() == [6, 2, 0, 2, 1]
    ((scalars, _),) = logger.scalars
    assert scalars == {
        "Motion Samples/walk": 6,
        "Motion Samples/run": 2,
        "Motion Samples/idle": 0,
        "Motion Samples/jab": 2,
        "Motion Samples/kick": 1,
        "Behavior Completions/jab": 0,
        "Behavior Completions/kick": 0,
    }


def test_histogram_holds_only_draws_since_the_previous_write(motions, monkeypatch):
    script_draws(monkeypatch, motions, [("all", ["walk", "walk"]), ("all", ["run"])])
    logger = logged(motions)

    motions.sample_ids(2)
    logger.write_data()
    logger.write_data()
    motions.sample_ids(1)
    logger.write_data()

    assert [values for _, values, _, _ in logger.histograms] == [[0, 0], [], [1]]
    assert [scalars["Motion Samples/walk"] for scalars, _ in logger.scalars] == [2, 2, 2]
    assert [scalars["Motion Samples/run"] for scalars, _ in logger.scalars] == [0, 0, 1]


def test_sampled_counts_match_real_draws(motions):
    torch.manual_seed(0)
    reset = motions.sample_reset(500)
    batch = motions.sample_ids(300)

    in_behavior = motions.motion_behavior[reset.motion_ids] >= 0
    drawn = torch.cat([reset.motion_ids, reset.base_motion_ids[in_behavior], batch])
    assert torch.equal(motions.sample_counts, torch.bincount(drawn, minlength=motions.num_motions))
    assert motions.sample_counts[motions.name_to_id["idle"]] == 0


def test_writes_on_the_logger_write_interval(motions):
    logger = logged(motions)
    logger.write_interval = 2
    for _ in range(4):
        logger.log()
    assert len(logger.histograms) == len(logger.scalars) == 2


@pytest.mark.parametrize("backend", ["none", "tensorboard"])
def test_real_logger_backends_write(tmp_path, backend):
    """A backend of none logs nothing without crashing; tensorboard keeps idle's empty bucket."""
    if backend == "tensorboard":
        pytest.importorskip("tensorboard")
    motions = make_motions(tmp_path)
    log_dir = tmp_path / "logs"
    with Logger(backend=backend, log_dir=str(log_dir), write_interval=1) as logger:
        logger.add_writer(MotionMetrics(motions, BehaviorManager(1, motions.num_behaviors, "cpu")).write)
        logger.log()  # an interval with no draws
        motions.sample_reset(64)
        logger.step(64)
        logger.log()

    if backend == "none":
        assert not log_dir.exists()
        return
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(log_dir), size_guidance={"histograms": 0}).Reload()
    empty, histogram = (event.histogram_value for event in events.Histograms(HISTOGRAM))
    assert empty.num == 0 and list(empty.bucket) == [0] * 5
    assert list(histogram.bucket_limit) == [0.5, 1.5, 2.5, 3.5, 4.5]
    assert list(histogram.bucket) == motions.sample_counts.tolist()
    assert histogram.bucket[motions.name_to_id["idle"]] == 0
    assert events.Scalars("Motion Samples/idle")[-1].value == 0


def step_behaviors(motions, behaviors, step, presses, dof_speed, base_motion_ids):
    """One `MotionLearningEnv._update_behaviors` at 10 steps per second. `presses` maps env to the
    behavior key it presses on this step."""
    time = torch.full((behaviors.mode.shape[0],), step * 0.1)
    keys = torch.zeros(behaviors.mode.shape[0], motions.num_behaviors)
    for env, behavior in presses.items():
        keys[env, behavior] = 1.0
    mode = behaviors.mode
    motion_ids = torch.where(mode != LOCOMOTION, motions.behavior_motion_ids[mode.clamp_min(0)], base_motion_ids)
    phase = behaviors.phase(time, motions.library.duration[motion_ids])
    stable = settled(
        phase, dof_speed, motions.settle_min_phase[motion_ids], motions.settle_max_dof_speed[motion_ids]
    )
    behaviors.update(keys, (phase >= 1.0) | stable, time)


def test_completions_count_phase_one_and_settle_but_not_death_or_reset(motions):
    """env 0's jab runs to phase 1 and env 1's kick settles: each returns to locomotion and counts.
    env 2's jab dies halfway and env 3, reset into a jab, times out: a reset replaces both, and
    neither counts."""
    jab, kick = 0, 1
    walk = motions.name_to_id["walk"]
    behaviors = BehaviorManager(4, motions.num_behaviors, "cpu")
    logger = logged(motions, behaviors)
    base_motion_ids = torch.full((4,), walk)
    dof_speed = torch.tensor([1.0, 0.0, 1.0, 1.0]).unsqueeze(-1).expand(4, NUM_DOFS)
    no_keys = torch.zeros(1, motions.num_behaviors)
    behaviors.reset(torch.tensor([3]), no_keys, torch.tensor([jab]), torch.tensor([-0.3]))

    for step in range(30):
        presses = {0: jab, 1: kick, 2: jab} if step == 2 else {}
        step_behaviors(motions, behaviors, step, presses, dof_speed, base_motion_ids)
        if step == 4:
            assert behaviors.mode.tolist() == [jab, kick, jab, jab]
            # env 3's episode times out and env 2 dies: both reset into locomotion
            behaviors.reset(torch.tensor([2, 3]), torch.zeros(2, motions.num_behaviors), torch.tensor([LOCOMOTION] * 2), torch.zeros(2))
        if step == 9:
            assert behaviors.mode.tolist() == [jab, LOCOMOTION, LOCOMOTION, LOCOMOTION]

    assert behaviors.mode.tolist() == [LOCOMOTION] * 4
    assert behaviors.completions.tolist() == [1, 1]
    logger.write_data()
    ((scalars, _),) = logger.scalars
    assert scalars["Behavior Completions/jab"] == 1
    assert scalars["Behavior Completions/kick"] == 1


def test_no_behaviors_logs_no_completions(tmp_path):
    manifest = write_manifest(tmp_path, {"walk": {"file": write_clip(tmp_path / "walk.npz", 11)}})
    motions = MotionSet.from_manifest(manifest, "cpu")
    behaviors = BehaviorManager(2, motions.num_behaviors, "cpu")
    logger = logged(motions, behaviors)

    behaviors.update(torch.zeros(2, 0), torch.ones(2, dtype=torch.bool), torch.zeros(2))
    motions.sample_reset(3)
    logger.write_data()

    ((scalars, _),) = logger.scalars
    assert scalars == {"Motion Samples/walk": 3}
