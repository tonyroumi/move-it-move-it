"""ReferenceMotions semantics: draw order, yaw handling and what each consumer receives."""

import pytest
import torch

pytest.importorskip("isaaclab.utils.math", reason="needs IsaacLab's quaternion math")

from conftest import FPS  # noqa: E402
from moveitmoveit.motion.library import MotionLibrary  # noqa: E402
from moveitmoveit.motion.reference import ReferenceMotions, SkeletonMap  # noqa: E402
from moveitmoveit.mdp import OBSERVATIONS, BodyState, ComposedMDP  # noqa: E402
from moveitmoveit.utils import transforms  # noqa: E402

DOF_ORDER = [2, 0, 1]  # robot order differs from clip order
ROOT = 0
KEY_BODIES = [2, 1]
NUM_STEPS = 2
PROBS = torch.tensor([0.5, 0.5, 0.0])
OBS_DIM = 3 + 3 + 1 + 6 + 3 + 3 + len(KEY_BODIES) * 3


@pytest.fixture
def library(clip_files) -> MotionLibrary:
    return MotionLibrary(clip_files, "cpu")


def picker(num_samples: int) -> torch.Tensor:
    return torch.multinomial(PROBS, num_samples, replacement=True)


def make_reference(library: MotionLibrary, random_yaw: bool = True, pick=picker) -> ReferenceMotions:
    style_obs = ComposedMDP([], OBSERVATIONS, OBSERVATIONS, "cpu").style_obs
    return ReferenceMotions(pick, library, SkeletonMap(DOF_ORDER, ROOT, KEY_BODIES), NUM_STEPS, random_yaw, style_obs)


def expected_amp_obs(library, ids, times, yaw, local_frame: bool) -> torch.Tensor:
    """The observation built by hand from the library, in the order training expects."""
    frames = library.history(ids, times, NUM_STEPS)
    positions, rotations, linear, angular = frames[2:]
    if yaw is not None:
        positions, rotations, linear, angular = transforms.apply_random_yaw(
            yaw.repeat_interleave(NUM_STEPS, dim=0), ROOT, positions, rotations, linear, angular
        )
    state = BodyState(
        frames.dof_positions[:, DOF_ORDER],
        frames.dof_velocities[:, DOF_ORDER],
        positions[:, ROOT],
        rotations[:, ROOT],
        linear[:, ROOT],
        angular[:, ROOT],
        positions[:, KEY_BODIES],
    )
    return torch.cat([term.fn(state, local_frame) for term in OBSERVATIONS.values()], dim=-1)


def test_style_batch_draws_ids_then_times_then_yaw(library):
    reference = make_reference(library)

    torch.manual_seed(5)
    batch = reference.style_batch(32)
    state_after = torch.get_rng_state()

    torch.manual_seed(5)
    ids = picker(32)
    times = library.sample_times(ids)
    yaw = transforms.random_yaw_orientation(32, "cpu")
    expected = expected_amp_obs(library, ids, times, yaw, local_frame=True).view(32, -1)

    assert batch.shape == (32, NUM_STEPS * OBS_DIM)
    assert torch.equal(batch, expected)
    assert torch.equal(state_after, torch.get_rng_state())


def test_style_batch_without_yaw_draws_no_yaw_and_stays_in_world_frame(library):
    reference = make_reference(library, random_yaw=False)

    torch.manual_seed(5)
    batch = reference.style_batch(32)
    state_after = torch.get_rng_state()

    torch.manual_seed(5)
    ids = picker(32)
    times = library.sample_times(ids)
    expected = expected_amp_obs(library, ids, times, None, local_frame=False).view(32, -1)

    assert torch.equal(batch, expected)
    assert torch.equal(state_after, torch.get_rng_state())


def test_style_batch_selects_clips_only_through_the_shared_picker(library):
    calls = []

    def spy(num_samples: int) -> torch.Tensor:
        calls.append(num_samples)
        return torch.zeros(num_samples, dtype=torch.long)

    batch = make_reference(library, random_yaw=False, pick=spy).style_batch(64)

    assert calls == [64]
    # only clip 0, the shortest and slowest: other clips' ramps would run well past its final frame
    assert batch[:, 0].max() <= library.dof_positions[0, :, DOF_ORDER[0]].max() + 1e-3


def test_amp_pairs_are_spaced_by_clip_dt(library):
    reference = make_reference(library, random_yaw=False)
    ids = torch.tensor([1])
    times = torch.tensor([10.0 / FPS], dtype=torch.float32)

    newest, older = reference.amp_seed(ids, times, torch.zeros(1, 4))[0]

    # clip 1's dof ramp is 2 * frame * (dof index + 1)
    assert newest[0].item() == pytest.approx(2.0 * 10.0 * (DOF_ORDER[0] + 1), abs=1e-2)
    assert older[0].item() == pytest.approx(2.0 * 9.0 * (DOF_ORDER[0] + 1), abs=1e-2)


def test_amp_seed_uses_the_callers_yaw_and_draws_nothing(library):
    reference = make_reference(library)
    ids = torch.tensor([0, 1, 1])
    times = torch.tensor([0.05, 0.1, 0.2])
    yaw = transforms.random_yaw_orientation(3, "cpu")

    state_before = torch.get_rng_state()
    seed = reference.amp_seed(ids, times, yaw)

    assert torch.equal(state_before, torch.get_rng_state())
    assert seed.shape == (3, NUM_STEPS, OBS_DIM)
    assert torch.equal(seed.view(6, -1), expected_amp_obs(library, ids, times, yaw, local_frame=True))


def test_amp_seed_ignores_yaw_when_random_yaw_is_off(library):
    reference = make_reference(library, random_yaw=False)
    ids = torch.tensor([0, 1])
    times = torch.tensor([0.05, 0.1])

    seed = reference.amp_seed(ids, times, transforms.random_yaw_orientation(2, "cpu"))

    assert torch.equal(seed.view(4, -1), expected_amp_obs(library, ids, times, None, local_frame=False))


def test_frames_for_reset_maps_to_the_robot_and_draws_nothing(library):
    reference = make_reference(library)
    ids = torch.tensor([0, 1])
    times = torch.tensor([0.05, 0.1])
    frames = library.frames(ids, times)

    state_before = torch.get_rng_state()
    reset = reference.frames_for_reset(ids, times)

    assert torch.equal(state_before, torch.get_rng_state())
    assert torch.equal(reset.dof_positions, frames.dof_positions[:, DOF_ORDER])
    assert torch.equal(reset.dof_velocities, frames.dof_velocities[:, DOF_ORDER])
    assert torch.equal(reset.root_positions, frames.body_positions[:, ROOT])
    assert torch.equal(reset.root_rotations, frames.body_rotations[:, ROOT])
    assert torch.equal(reset.root_linear_velocities, frames.body_linear_velocities[:, ROOT])
    assert torch.equal(reset.root_angular_velocities, frames.body_angular_velocities[:, ROOT])


def test_tracking_targets_rotate_key_bodies_about_the_root(library):
    reference = make_reference(library)
    ids = torch.tensor([0, 1])
    times = torch.tensor([0.05, 0.1])
    yaw = transforms.random_yaw_orientation(2, "cpu")
    frames = library.frames(ids, times)

    targets = reference.tracking_targets(ids, times, yaw)

    assert torch.equal(targets.dof_positions, frames.dof_positions[:, DOF_ORDER])
    # the root is the pivot, so it stays put while the offsets rotate
    assert torch.equal(targets.root_positions, frames.body_positions[:, ROOT])
    offsets = frames.body_positions[:, KEY_BODIES] - frames.body_positions[:, ROOT].unsqueeze(1)
    rotated = transforms.quat_apply(yaw.unsqueeze(1).expand(-1, len(KEY_BODIES), 4), offsets)
    assert torch.allclose(targets.key_body_positions - targets.root_positions.unsqueeze(1), rotated, atol=1e-5)


def test_tracking_targets_ignore_yaw_when_random_yaw_is_off(library):
    reference = make_reference(library, random_yaw=False)
    ids = torch.tensor([0, 1])
    times = torch.tensor([0.05, 0.1])
    frames = library.frames(ids, times)

    targets = reference.tracking_targets(ids, times, transforms.random_yaw_orientation(2, "cpu"))

    assert torch.equal(targets.key_body_positions, frames.body_positions[:, KEY_BODIES])


def test_tracking_targets_hold_the_final_frame_past_the_clip_end(library):
    reference = make_reference(library, random_yaw=False)

    targets = reference.tracking_targets(torch.tensor([0]), torch.tensor([50.0]), torch.zeros(1, 4))

    assert targets.dof_positions[0].tolist() == pytest.approx([10.0 * (k + 1) for k in DOF_ORDER], abs=1e-2)
