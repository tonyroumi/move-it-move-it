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

`src/configs/goal_manifest.yaml` trains walking toward a random world-space goal, sampled per
environment as an offset from the robot (`goal_x`/`goal_y`) and held fixed in the world until the
next resample. The policy observes the goal as an offset in its heading frame; `progress` rewards
closing the distance and is 0 on the first step after a reset or a new goal.

```bash
./isaaclab.sh -p scripts/train.py --algo AMP --manifest src/configs/goal_manifest.yaml
```

Drive the goal from the keyboard while playing a checkpoint trained this way. The arrow keys move
the goal (a red sphere) in world X/Y from a point 1 m ahead of the robot's start position; `L`
puts it back there:

```bash
./isaaclab.sh -p scripts/play.py --checkpoint <path> --joystick --joystick-mode point_goal
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
- **Transitions.** Commands resample every `command_resample_steps` and on reset, with a zero
  command at the motion's `zero_command_prob`, so episodes switch between modes. Resets start
  from clip frames, so standing envs also learn to stop from a walking pose. Goal envs also switch
  when they arrive at a goal or get a new one.
- **Terms.** `root_lin_vel_l2`, `root_ang_vel_l2`, `upright` and `support_centered` (root over the
  midpoint of the feet) are bounded in [0, 1], like `joint_vel` and `action_rate_l2`. They reward
  standing still; they can't guarantee the robot is perfectly motionless.
- A motion without a `standing` block behaves as before.
