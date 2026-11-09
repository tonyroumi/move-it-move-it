# move-it-move-it

An RL framework built on IsaacLab, implementing PPO and AMP
(Adversarial Motion Priors) for training stylistic humanoid locomotion policies.

## Setup

```bash
pip install -e . --no-deps
```

Requires IsaacLab/Isaac Sim installed separately. IsaacLab submodules
(`isaaclab`, `isaaclab_tasks`, `isaaclab_physx`) are not on PyPI — install
them via `./isaaclab.sh --install` from the IsaacLab repo.

## Running

Train:

```bash
./isaaclab.sh -p scripts/train.py --task <task_name>
```

Play a checkpoint:

```bash
./isaaclab.sh -p scripts/play.py --task <task_name> --checkpoint <path>
```

## Goal tracking

`configs/goal_manifest.yaml` trains walking toward a random world-space goal, sampled per
environment as an offset from the robot (`goal_x`/`goal_y`) and held fixed in the world until the
next resample. The policy observes the goal as an offset in its heading frame; `progress` rewards
closing the distance and is 0 on the first step after a reset or a new goal.

```bash
./isaaclab.sh -p scripts/train.py --algo AMP --manifest configs/goal_manifest.yaml
```

Drive the goal from the keyboard while playing a checkpoint trained this way. The goal (a red
sphere) starts 1 m ahead of the robot's start position and the arrow keys move it relative to the
robot's current heading; `L` puts it back there. `--overlay` shows the held keys:

```bash
./isaaclab.sh -p scripts/play.py --checkpoint <path> --joystick-mode point_goal
```

A policy only walks to goals like those it trained on, so goals far outside the sampled range,
or behind obstacles, may not be reached.

## Standing mode

A motion that declares a `standing` block trains a second, per-environment mode: when the
command asks for no locomotion, the robot is asked to stand as still as it can instead of
imitating the motion's clip.

```yaml
standing:
  lin_vel_threshold: 0.1   # m/s, on the lin_x/lin_y norm (default 0.1)
  yaw_vel_threshold: 0.1   # rad/s, on |yaw| (default 0.1)
  goal_radius: 0.3         # m, on the goal's distance from the root (default 0.3)
  rewards:                 # replace the motion's `rewards` while standing
    root_lin_vel_l2: 2.0
    upright: 1.0
```

- **When an env stands.** Its command, as the policy observes it, has a lin_x/lin_y norm and
  |yaw| within the thresholds and a goal within `goal_radius` of the root. Motions without a goal
  observe a zero goal offset, so only the velocity thresholds apply. A goal farther away means
  "walk there", even though its velocity commands are 0; arriving within the radius means
  "stand". The mode is a function of the observed command, so the observation layout is
  unchanged and the policy can tell the modes apart.
- **What changes while standing.** The task reward comes from `standing.rewards` and AMP's style
  reward is 0 for that env. The discriminator's training is unchanged: standing envs' AMP
  observations still reach its buffer. A nonzero command restores the motion's `rewards` and the
  style reward from the next step on.
- **Style reward off.** AMP doesn't know about standing: it zeroes the style reward for the envs in
  the env's per-step `extras["style_reward_off"]` (N,) bool mask, and every env earns style when
  the mask is missing. `MotionLearningEnv._style_reward_off()` builds that mask, today from
  standing mode alone; OR another per-env condition into it to turn style off for those envs too,
  without putting them in standing mode. `extras["standing"]` is still exported, but AMP no longer
  reads it.
- **Transitions.** Commands resample every `command_resample_steps` and on reset, with a zero
  command at the motion's `zero_command_prob`, so episodes switch between modes. Resets start
  from clip frames, so standing envs also learn to stop from a walking pose. Goal envs also switch
  when they arrive at a goal or get a new one.
- **Terms.** `root_lin_vel_l2`, `root_ang_vel_l2`, `upright` and `support_centered` (root over the
  midpoint of the feet) are bounded in [0, 1], like `joint_vel` and `action_rate_l2`. They reward
  standing still; they can't guarantee the robot is perfectly motionless.
- A motion without a `standing` block behaves as before.

### Standing pose

Those terms leave the posture free, and a policy can stand still in a wide, arms-out stance.
`standing_pose` asks for a posture too: a `standing.pose` target per joint, scored only in standing
mode, since it may only be declared in `standing.rewards`.

```yaml
standing:
  pose:                    # rad per joint, by the robot's joint names; unlisted joints target 0
    right_hip_x: -0.12
    left_hip_x: 0.12
  rewards:
    standing_pose:
      weight: 1.5
      scale: 2.0           # how sharply the reward falls off past the tolerance (default 2.0)
      tolerance: 0.05      # rad each joint may deviate for free (default 0.05)
```

- **Semantics.** Per joint, the deviation from the target beyond `tolerance` is squared and summed
  over joints, and the reward is `exp(-scale * sum)`, in [0, 1]. The tolerance is the room the
  policy has to balance at no cost. Targets are in the robot's joint coordinates: at 0 the robot
  stands upright with legs and arms straight down and its feet 0.17 m apart. The `_x` joints
  roll about the forward axis: a positive angle swings a hanging limb towards the robot's left.
  Elbows bend at negative angles and knees at positive ones.
- **The shipped target.** Both manifests ask for an upright stance with the feet about shoulder
  width apart and the arms down. Hip rolls of ±0.12 rad put the ankles 0.37 m apart, the
  robot's distance between its shoulder joints (0.17 m at 0), and the opposite ankle rolls keep
  the soles flat. Shoulder rolls of ±0.18 rad hold the hands clear of the thighs, which they
  nearly touch at 0, and the elbows bend 0.2 rad. Every other joint targets 0.
- **Tuning.** Raise `scale` or lower `tolerance` to hold the posture more tightly; raise
  `tolerance` to leave more room for balance. With the shipped values, every joint at its
  tolerance scores 1.0, every joint 0.1 rad off scores 0.87, the zero pose 0.82, the feet 0.75 m
  apart 0.77, the arms 45° out 0.29, and a wide T pose 0.003. All joints count equally.
- **It's a reward, not a constraint.** The policy trades posture off against staying still and
  upright, and against what physics allows, so it won't hold the target exactly. Contacts,
  gravity and the PD drives' tracking error still move the joints.
- **Compatibility.** `standing.pose` and `standing_pose` are optional. A manifest without them
  scores standing as before, and the observation and action layouts don't change. A policy trained
  without the term won't stand differently until it is retrained with it.

## Key-triggered boxing behaviors

`configs/manifests/boxing_manifest.yaml` trains orthodox-stance footwork toward a world-space goal, as
above, plus three one-shot behaviors: a jab, a right hook and a combo. A manifest motion with
`trigger: binary_key_N` is a behavior. Pressing that key starts it once, and the env goes back to
locomotion when the behavior completes. Releasing, holding or pressing keys while it runs changes
nothing, and a key held past its end does not start it again.

- **What the policy observes**: its state, the goal as an offset in its heading frame, the
  running behavior z(t) as a one-hot in the trigger slots (all 0 during locomotion), and its
  progress phi(t) = min(1, (t - t_start) / duration) in `BEHAVIOR_PHASE`. It never sees the raw
  key.
- **When a behavior completes**: at phi >= 1, or earlier at a stable posture if the motion
  declares `settle: {min_phase, max_dof_speed}`. That means phi >= min_phase with an RMS joint
  speed below max_dof_speed. The robot stands just as still in its guard before the punch, so
  min_phase is set past the end of each clip's punch.
- **What is rewarded**: while a behavior runs, the env follows that behavior's clip from t_start.
  Its rewards (`heading_motion_tracking`, which compares poses in each root's heading frame) and
  its terminations apply. The goal rewards resume when it completes.
- **How training triggers behaviors**: training presses the trigger keys at random, with random
  hold lengths, into the same command slots the keyboard writes. Training and play therefore run
  the same latching code (`moveitmoveit/behavior.py`). The press rate is derived from
  `sample_prob`, so the agent spends about as much of its time in each behavior as the
  discriminator's reference data does. An episode that resets into a behavior clip starts that
  behavior part-way through.
- **Standing mode**: a behavior can't declare `standing`, so an env running one is never in
  standing mode and earns AMP's style reward. When the behavior completes, the locomotion motion's
  own standing mode, if it declares one, applies again. The boxing manifest declares none, so a
  zero command there still imitates the bounce clip's guard, with style reward on.

```bash
./isaaclab.sh -p scripts/train.py --algo AMP --manifest configs/manifests/boxing_manifest.yaml
./isaaclab.sh -p scripts/play.py --checkpoint <path> --joystick --joystick-mode point_goal
```

In play, the arrow keys move the goal, Left Shift throws the jab, Space the right hook and C the
combo. Run both commands from the repo root, because the manifest's clip paths are relative.

### Boxing clips and attribution

`data/` isn't tracked, so put the clips at the paths in the table below before training. They
are adaptations of "Open Motion Project" by ACCAD/The Ohio State University, performer set
"Male 2", licensed under
[Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/).
Source: <https://accad.osu.edu/research/motion-lab/mocap-system-and-data>, archive
`Male2_bvh.zip` (terms read on 2026-11-06). The source's terms apply to the clips, and no source
endorses this project or these adaptations.

Changes made: each take was retargeted from its joint positions to the IsaacLab Humanoid28 with the
`humanoid28_pyroki` solver at default weights, resampled from 30 to 60 fps, rotated to face +X at
the start, and centred on the origin in XY. Clips marked "Mirrored" were reflected left-to-right,
which turns the source's southpaw stance into orthodox (left foot forward). The three `_crop` clips
were then cut from those retargeted clips to the window below and translated in XY so the torso
starts at the origin. These are new kinematic trajectories on a different body, not the original
capture data. Hand orientation and fingers are not represented.

| Clip | Seconds | Source take | Source crop | Mirrored |
|---|---:|---|---|---|
| `data/boxing/orthodox_advance.npz` | 5.65 | `Male2_E3_Advance.bvh` | 0–5.7 s | yes |
| `data/boxing/orthodox_quick_advance.npz` | 3.72 | `Male2_E4_QuickAdvance.bvh` | full | yes |
| `data/boxing/orthodox_retreat.npz` | 7.68 | `Male2_E5_Retreat.bvh` | full | yes |
| `data/boxing/orthodox_quick_retreat.npz` | 4.15 | `Male2_E6_QuickRetreat.bvh` | 1.2 s–end | yes |
| `data/boxing/orthodox_bounce.npz` | 6.95 | `Male2_E8_Bounce.bvh` | 0–7.0 s | yes |
| `data/boxing/orthodox_side_step_right.npz` | 2.25 | `Male2_E9_SideStepLeft.bvh` | full | yes |
| `data/boxing/orthodox_side_step_left.npz` | 1.92 | `Male2_E10_SideStepRight.bvh` | full | yes |
| `data/boxing/orthodox_jab_left_crop.npz` | 0.80 | `Male2_E1_JabLeft.bvh` | 0.85–1.65 s of the full retargeted take | no |
| `data/boxing/orthodox_hook_right_crop.npz` | 1.50 | `Male2_E5_HookLeft.bvh` | 0.50–2.00 s of the full retargeted take | yes |
| `data/boxing/orthodox_combo_123_crop.npz` | 1.30 | `Male2_Extended_1.bvh` | 1.55–2.85 s of the retargeted 0–13.5 s crop | yes |

**Clip requirements.** The manifest loads ten clips, and all must be in place before training or
play.

- The seven footwork clips are used whole.
- The three `_crop` clips are not full takes. Cut each from its full retargeted take in
  `data/boxing/`:
  - `orthodox_jab_left.npz` → `orthodox_jab_left_crop.npz`, 0.85–1.65 s
  - `orthodox_hook_right.npz` → `orthodox_hook_right_crop.npz`, 0.50–2.00 s
  - `orthodox_combo_guard.npz` → `orthodox_combo_123_crop.npz`, 1.55–2.85 s

  Keep frames `round(start * fps)` to `round(end * fps)` inclusive of every per-frame array. Then
  subtract the first frame's torso XY from every body's XY. Leave all other arrays unchanged:

```python
import numpy as np

def crop(src, dst, start, end):
    clip = dict(np.load(src))
    fps, n = float(clip["fps"]), clip["dof_positions"].shape[0]
    i0, i1 = round(start * fps), round(end * fps)
    out = {k: v[i0 : i1 + 1].copy() if v.ndim and v.shape[0] == n else v for k, v in clip.items()}
    torso = clip["body_names"].tolist().index("torso")
    out["body_positions"][..., :2] -= out["body_positions"][0, torso, :2]
    np.savez(dst, **out)
```

- A training run doesn't load the other clips in `data/boxing/`, such as southpaw, uppercut or
  CMU rounds.

The right hook is mirrored from a take whose source stance is weak (about 0.13 m of foot lead).
The bounce clips exceeded the retarget's hand-speed check (1.9 against 1.1 m/s peak) and were
kept as is.

Nothing here has been trained or run in Isaac Lab, so whether a policy learns footwork, punches
or the combo is untested. The combo clip is a candidate jab, cross and lead hook, picked out by
hand-speed peaks and not checked by eye. The punch data is one repetition each, and nothing in it
punches while moving.

The command block grew from 7 to 9 slots (`BINARY_KEY_2`, `BEHAVIOR_PHASE`), so checkpoints
trained before this change do not load. Manifests without triggers train as before; their
`BEHAVIOR_PHASE` slot is always 0.
