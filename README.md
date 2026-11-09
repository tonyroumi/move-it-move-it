# move-it-move-it

PPO and AMP (Adversarial Motion Priors) for training a 28-DOF humanoid to imitate motion clips
and follow commands, built on IsaacLab. Everything runs as the single task `MoveIt-Humanoid-v0`.

## Setup

1. Install IsaacLab / Isaac Sim. Its packages (`isaaclab`, `isaaclab_tasks`, `isaaclab_assets`, ...)
   are not on PyPI; install them with `./isaaclab.sh --install` from the IsaacLab repo.
2. Install this package into the same environment:

   ```bash
   pip install -e . --no-deps
   ```

Run commands from this repo's root through IsaacLab's launcher (`<isaaclab>/isaaclab.sh -p ...`;
shortened to `./isaaclab.sh` below).

## Data

Motion clips (`.npz`) are distributed separately:

> **Download:** _TBD (link coming soon)._

Put them in `data/` at the repo root (git-ignored), or anywhere and set `MOVEITMOVEIT_DATA_DIR` to
that directory. A relative manifest `file:` (e.g. `walk.npz`, `jump/running_jump.npz`) is resolved
against that data directory; an absolute one is used as-is.

Preview a clip kinematically:

```bash
./isaaclab.sh -p scripts/play_motion.py --motion data/walk.npz
```

## Training

```bash
./isaaclab.sh -p scripts/train.py --algo AMP --manifest configs/manifest.yaml --headless
```

| Option | Meaning |
| --- | --- |
| `--manifest` | Required. Motions, commands and rewards (see [Manifest](#manifest)). |
| `--algo` | Required, `AMP` or `PPO`. Loads `configs/agents/amp.yaml` or `configs/agents/ppo.yaml`. |
| `--num_envs` | Defaults to 4096. |
| `--checkpoint` | Resume from a `.pt` file. |
| `env.<field>=...` / `agent.<path>=...` | Hydra overrides of `MotionLearningEnvCfg` (`src/moveitmoveit/env_cfg.py`) or the agent YAML. |

Seed, experiment name, logger backend (`tensorboard`, `wandb`, `both`, `none`) and checkpoint
interval come from `configs/experiment.yaml`. Runs are written to
`logs/<directory>/<experiment_name>/<algo>/<timestamp>/`, with `params/` (env, agent and manifest
configs) and `checkpoints/` (`<step>.pt` every `checkpoint_interval` iterations, `best_agent.pt`
for the best mean episode reward).

## Playing

```bash
./isaaclab.sh -p scripts/play.py --checkpoint logs/.../<timestamp>/checkpoints/best_agent.pt
```

The env and agent configs are reloaded from the run's `params/`, so play uses the manifest and
observations the policy was trained with.

| Option | Meaning |
| --- | --- |
| `--num_envs` | Defaults to 1. |
| `--joystick` | Drive commands from the keyboard (no `--headless`). Key bindings are printed at start-up. |
| `--joystick-mode velocity\|point_goal` | `velocity` (default): lin_x/lin_y/yaw, with Left Shift / Space as `binary_key_0` / `binary_key_1`. `point_goal`: arrows move a red goal sphere relative to the robot's heading, `L` resets it; needs a manifest with `goal_x`/`goal_y`. |
| `--joystick-overlay` | Show held keys on screen (with `--joystick`). |
| `--camera_type facing\|third-person\|none` | Scripted camera. |
| `--real-time` | Step at wall-clock speed. |
| `--record [--record-length N]` | Save a video to `<run>/videos/play/` (default 200 steps). |

## Manifest

A manifest is a YAML file with one top-level `motions:` key. Each motion is one clip plus how to
train on it; unknown keys are rejected.

```yaml
motions:
  walk:
    file: walk.npz             # required; relative to the data directory
    sample_prob: 1.0           # relative weight for resets and discriminator batches (default 1.0)
    commands:                  # [low, high] range or a fixed value; omitted commands are 0
      lin_x: [0.0, 1.5]        # m/s, body frame
      lin_y: [-0.5, 0.5]       # m/s, body frame
      yaw: [-1.5, 1.5]         # rad/s
      binary_key_0: 0.0        # also binary_key_1
      # goal_x / goal_y: [low, high] m, offset from the root when sampled, then fixed in the world
    zero_command_prob: 0.2     # chance a sampled command is all zeros (default 0.0)
    rewards:                   # name: weight, or name: {weight, <kwarg>: value}
      lin_vel_tracking: {weight: 3.0, scale: 2.0}
      yaw_vel_tracking: 3.0
    termination: unhealthy     # name or list (default unhealthy)
    standing:                  # optional, see below
      lin_vel_threshold: 0.1   # m/s (default 0.1)
      yaw_vel_threshold: 0.1   # rad/s (default 0.1)
      goal_radius: 0.3         # m (default 0.3)
      pose: {right_hip_x: -0.12, left_hip_x: 0.12}  # rad per joint, unlisted joints target 0
      rewards:
        root_lin_vel_l2: 2.0
        standing_pose: {weight: 1.5, scale: 2.0, tolerance: 0.05}
```

Commands resample every `env.command_resample_steps` (150) steps and on reset.

**Standing mode.** With a `standing` block, an env whose observed command is below both velocity
thresholds and whose goal (if any) is within `goal_radius` is "standing": `standing.rewards`
replace `rewards` and the AMP style reward is 0 for that env.

Shipped manifests: `manifest.yaml` (velocity walking + standing), `smooth_walk_manifest.yaml`
(forward walking + standing), `goal_manifest.yaml` (walk to a point goal + standing),
`manifest_jump.yaml` (jump-clip catalog for viewing; no commands or rewards).

## Rewards

Defined in `src/moveitmoveit/mdp.py`; the kwargs below can be set per motion (defaults shown). The
task reward is the weighted sum; AMP adds its style reward (`style_reward_lambda`,
`goal_reward_lambda` in the agent YAML).

| Reward | Kwargs | Rewards |
| --- | --- | --- |
| `motion_tracking` | `dof_scale=2.0`, `key_body_scale=5.0` | DOF positions and key-body offsets matching the clip |
| `lin_vel_tracking` | `scale=2.0` | body-frame lin_x/lin_y matching the command |
| `yaw_vel_tracking` | `scale=2.0` | yaw rate matching the command |
| `line_following` | `scale=2.0` | moving in the commanded world direction |
| `target_hit` | `scale=0.5` | being close to the goal |
| `progress` | `scale=1.0` | distance closed to the goal this step (unbounded) |
| `action_rate_l2` | `scale=1.0` | small action changes |
| `joint_acc` | `scale=1e-4` | small joint accelerations |
| `joint_vel` | `scale=1e-2` | small joint velocities |
| `root_lin_vel_l2` | `scale=4.0` | low root linear speed |
| `root_ang_vel_l2` | `scale=2.0` | low root angular speed |
| `upright` | `scale=10.0` | an upright root |
| `support_centered` | `scale=20.0` | root over the midpoint of the feet |
| `standing_pose` | `scale=2.0`, `tolerance=0.05` | joints at `standing.pose`; only allowed in `standing.rewards` |

All but `progress` are `exp(-scale * error)` in [0, 1]. Terminations: `unhealthy` (root below
0.65 m), `deviation_from_motion` (summed squared DOF error from the clip above 1.5).

## Observations

The policy and the AMP discriminator each observe a list of named terms, set by `policy_obs` and
`style_obs` in `src/moveitmoveit/env_cfg.py` (both default to all seven terms):

| Term | Dim |
| --- | --- |
| `dof_pos` | 28 |
| `dof_vel` | 28 |
| `root_height` | 1 |
| `root_orientation` | 6 |
| `root_lin_vel` | 3 |
| `root_ang_vel` | 3 |
| `key_body_offsets` | 12 (hands and feet relative to the root) |

The policy observation always ends with the 7 commands (`lin_x`, `lin_y`, `yaw`, `binary_key_0`,
`binary_key_1`, `goal_x`, `goal_y`; the goal as an offset in the robot's heading frame). Change
the terms by editing `env_cfg.py` or overriding at train time:

```bash
./isaaclab.sh -p scripts/train.py --algo AMP --manifest configs/manifest.yaml \
  'env.policy_obs=[dof_pos,dof_vel,root_height,root_orientation,root_lin_vel,root_ang_vel]'
```

To add a term, register a function in `mdp.py` with `@observation(dim=...)`.

## Tests

```bash
pytest
```

The tests cover manifest loading and agent-config selection; they need `torch`, `gymnasium` and
`pyyaml`, not IsaacLab.
