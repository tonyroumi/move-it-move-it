# move-it-move-it 


<table align="center" width="100%">
  <tr>
    <td align="center" width="33%">
      <a href="docs/videos/humanoid_motion_hero.mp4">
        <img src="docs/videos/humanoid_motion_hero.gif" width="100%">
      </a>
      <sub><b>Humanoid Motion Imitation</b></sub>
    </td>
    <td align="center" width="17%">
      <a href="docs/videos/standing_broad_jump.mp4">
        <img src="docs/videos/standing_broad_jump.gif" width="100%">
      </a>
      <sub><b>Standing Broad Jump</b></sub>
      <br>
      <a href="docs/videos/walk.mp4">
        <img src="docs/videos/walk.gif" width="100%">
      </a>
      <sub><b>Walking</b></sub>
    </td>
    <td align="center" width="33%">
      <a href="docs/videos/humanoid_motion_hero.mp4">
        <img src="docs/videos/humanoid_motion_hero.gif" width="100%">
      </a>
      <sub><b>Humanoid Motion Imitation</b></sub>
    </td>
    <td align="center" width="17%">
      <a href="docs/videos/standing_broad_jump.mp4">
        <img src="docs/videos/standing_broad_jump.gif" width="100%">
      </a>
      <sub><b>Standing Broad Jump</b></sub>
      <br>
      <a href="docs/videos/walk.mp4">
        <img src="docs/videos/walk.gif" width="100%">
      </a>
      <sub><b>Walking</b></sub>
    </td>
  </tr>
</table>

# Overview

The following code base provides a flexible RL environment for learning stylistic humanoid control.

The environment is driven by a single manifest file that supports learning diverse control behaviors through motion-specific
rewards, termination conditions, task commands, behavior triggers, and optional AMP-style rewards. 


The following methods have been implemented:
- [Adversarial Motion Priors (AMP)](https://arxiv.org/abs/2104.02180)
- [DeepMimic](https://arxiv.org/abs/1804.02717)

## Installation

1.  This framework is built in [IsaacLab](https://github.com/isaac-sim/IsaacLab) and tested with `2e44ddb2e19536579140496023b5ccb060bc4152`.
2. Install using Isaac Lab's Python environment:

```bash
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e . --no-deps
```


## Data
Download the motion data from [Google Drive](https://drive.google.com/file/d/1xOpZKUo--7TSZqnA4kxun0gezIoPZx9m/view?usp=sharing) and unzip the archive into the repository's `data/` folder:

```bash
unzip <downloaded-archive.zip> -d data/
```

## Running

Train:

```bash
./isaaclab.sh -p scripts/train.py --algo <PPO|AMP> --manifest <manifest.yaml> --name <experiment_name>
```

- `--algo`: `PPO` uses the manifest rewards; `AMP` also adds a discriminator style reward.
- `--manifest`: motion manifest.
- `--name`: experiment name. Runs are saved to `logs/<experiment.directory>/<name>/<algo>/<timestamp>/`.
- `--num_envs`: number of parallel environments (default 4096).
- `--checkpoint`: resume from a checkpoint.

Play a checkpoint:

```bash
./isaaclab.sh -p scripts/play.py --checkpoint <path>
```

- `--checkpoint`: checkpoint in a run's `checkpoints/` folder. Environment and agent settings are loaded from that run's `params/` folder. Play uses one environment.
- `--joystick none|velocity|point_goal` (default `none`): use random commands or keyboard control. In `velocity` mode, arrow keys move and `Z`/`X` turn. In `point_goal` mode, arrow keys move the goal marker. `L` resets the keyboard.
- `--camera facing|third-person|none`: choose the follow camera.
- `--real-time`: run at real-time speed when possible.
- `--record`, `--record-length <steps>` (default 200): save a video to `<run>/videos/play/`.

Play a motion:

```bash
./isaaclab.sh -p scripts/play_motion.py --motion <path>
```

- `--motion`: `.npz` motion clip to play on the humanoid,
- `--camera facing|third-person|none`: choose the follow camera.
- `--record`, `--record-length <steps>` (default 200): save a video to `logs/videos/play_motion/`.

<br>

# Framework

## Observations

The environment supports the following observation terms:

| Name | Size | Description |
| --- | --- | --- |
| `dof_pos` | 28 | Joint positions |
| `dof_vel` | 28 | Joint velocities |
| `root_height` | 1 | Root height |
| `root_orientation` | 6 | Root orientation represented by tangent and normal vectors |
| `root_lin_vel` | 3 | Root linear velocity |
| `root_ang_vel` | 3 | Root angular velocity |
| `key_body_offsets` | 12 | Hand and foot positions relative to the root |
| `task_commands` | Variable | Task-specific commands defined by the active motion |
| `behavior` | Number of behaviors | One-hot encoding of the active behavior |
| `phase` | 1 | Normalized motion phase in the range [0, 1] |


Proprioceptive observation terms are defined in `mdp/observations.py`. When `random_reset` is enabled, velocities and key body offsets are expressed in the robot's local frame.

Proprioceptive observations are configured in `env_cfg.py`.

### Adding an Observation
Add a function in `observations.py` with the following arguments, then add its name to `policy_obs` and/or `style_obs`.

```python
@observation(dim=1)
def root_speed(state: BodyState, local_frame: bool) -> torch.Tensor:
    return state.root_linear_velocities.norm(dim=-1, keepdim=True)
```

## Rewards

The environment supports the following reward terms:

| Name | Description |
| --- | --- |
| `motion_tracking` | Match reference joint positions and hand/foot offsets |
| `heading_motion_tracking` | Motion tracking with offsets in the heading frame |
| `lin_vel_tracking` | Match commanded forward and sideways velocity |
| `yaw_vel_tracking` | Match the commanded turn rate |
| `progress` | Move closer to the goal each step |
| `target_hit` | Be close to the goal |
| `action_rate_l2` | Favor small changes between consecutive actions |
| `joint_acc` | Favor low joint accelerations |
| `joint_vel` | Favor low joint velocities |
| `root_lin_vel_l2` | Favor low root linear velocity |
| `root_ang_vel_l2` | Favor low root angular velocity |
| `upright` | Keep the torso upright |
| `support_centered` | Keep the root above the middle of the feet |

`progress` and `target_hit` are zero for motions without a goal command.

Configure rewards per motion in the manifest:

```yaml
    rewards:
      heading_motion_tracking:
        weight: 25.0
        dof_scale: 0.5
      action_rate_l2: 0.5   # shorthand for weight: 0.5
```

### Adding a Reward
Add a function in `rewards.py`. It can then be selected by name in a manifest.

```python
@reward(scale=1.0)
def root_height_tracking(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    error = c.state.root_positions[:, 2] - c.reference_state.root_positions[:, 2]
    return torch.exp(-scale * error**2)
```

## Tasks

Tasks are expressed as commands, listed in `mdp/commands.py`:

| Command | Description |
| --- | --- |
| `lin_x`, `lin_y` | Target forward and sideways velocity |
| `yaw` | Target turn rate |
| `goal_x`, `goal_y` | Point goal |
| `binary_key_0`, `binary_key_1`, `binary_key_2` | Behavior triggers (see [Behavior Triggers](#behavior-triggers)) |

Set commands per motion in the manifest.

>Note: Sampled goals are offsets from its current position, and the policy sees the goal relative to its root.

```yaml
    commands:
      goal_x: [-5.0, 5.0]
      goal_y: [-5.0, 5.0]
    zero_command_prob: 0.25
```

Commands are resampled every `command_resample_steps` environment steps (in `env_cfg.py`). 

Pair commands with matching rewards: velocity commands with `lin_vel_tracking`/`yaw_vel_tracking`, or point goals with `progress`/`target_hit`.

### Adding a task command
Add an entry to `CommandIndex`; its lowercase name becomes available in manifests.

If the policy needs a transformed value, register a transform with `@command(<indexes>)`, following `point_goal`. Add a reward that reads the new command from `c.commands`.

## Terminations

The environment supports the following termination terms:

| Name | Description |
| --- | --- |
| `unhealthy` | Root falls below `height_threshold` (default 0.55) |
| `deviation_from_motion` | Summed squared joint error exceeds `threshold` (default 1.5) |

Configure terms per motion under `termination` in the manifest. Episodes also time out after `episode_length_s` (default 10 seconds, in `env_cfg.py`).

```yaml
    termination:
      unhealthy:
      deviation_from_motion:
        threshold: 5.0
```

### Adding a termination
Add a function in `mdp/terminations.py`, then enable it by name under `termination` in the manifest.

```python
@termination(max_speed=10.0)
def too_fast(c: StepContext, max_speed: float) -> torch.Tensor:
    return c.state.root_linear_velocities.norm(dim=-1) > max_speed
```

## Behavior Triggers

A motion becomes a **behavior** when its manifest entry has a `trigger`. Motions without one are **steady-state / locomotion**: the motion the character returns to between behaviors.

```yaml
  jump:
    file: data/jump/standing_broad_jump.npz
    sample_prob: 0.4
    trigger: binary_key_0   # binary_key_0, binary_key_1 or binary_key_2
```

To add a behavior, add a motion with a `trigger`. Give each behavior its own key.

**Random key presses (training).** During training, trigger keys are pressed automatically:



1. **Starting from locomotion or standing.** 

The behavior starts at its first frame when the key is pressed.

2. **Finishing a behavior.** 

At the end of the clip, the environment returns to locomotion.

3. **Starting an episode.** 

A motion is sampled using `sample_prob`. If a behavior is selected, the episode starts at a random point in its clip, or at its first frame with `random_start: false`. A locomotion motion is also sampled for the character to return to afterward.

4. **Only behaviors.** 

If every motion has a trigger, there is no locomotion or random key pressing. At the end of each behavior, another is selected.

**Keyboard control.** In play mode with `--joystick velocity` or `--joystick point_goal`, keyboard input replaces random presses: `LEFT_SHIFT` triggers `binary_key_0`, `RIGHT_SHIFT` triggers `binary_key_1`, and `C` triggers `binary_key_2`. Without `--joystick`, play uses random presses as in training.

## Motion Settings

Two per-motion settings control how the reference clip is played:

- `random_start` (default `true`): when an episode starts in this motion, it starts at a random point in the clip. Set it to `false` to always start at the first frame.

```yaml
  walk_to_goal:
    file: data/locomotion/walk.npz
    random_start: true
```