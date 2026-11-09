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

Drive the goal from the keyboard while playing a checkpoint trained this way. The arrow keys move
the goal (a red sphere) in world X/Y from a point 1 m ahead of the robot's start position; `L`
puts it back there:

```bash
./isaaclab.sh -p scripts/play.py --checkpoint <path> --joystick --joystick-mode point_goal
```

A policy only walks to goals like those it trained on, so goals far outside the sampled range,
or behind obstacles, may not be reached.

### Nearby goal

A motion with a `nearby_goal` block, in place of `goal_x`/`goal_y` commands, ties the goal to the
robot. A direction is held, the way an arrow key is held: forward, back, left, right, or a
diagonal normalized to unit length. Every step the env re-places the goal `distance` m away along
that direction, in the robot's current heading frame (yaw only). So the goal follows the robot as
it moves and turns, and the policy always observes the offset `distance * direction`. With no
direction held, the goal is at the root, and a motion with a `standing` block stands.

```yaml
nearby_goal:
  distance: 1.0     # m; must exceed standing.goal_radius, or loading fails (default 1.0)
  idle_prob: 0.25   # chance a sampled direction is none (default 0.25)
  switch_prob: 0.02 # per-step chance of a new direction between resamples (default 0)
```

In training, the direction is sampled on reset and every `command_resample_steps`, uniformly
over the eight directions or none at `idle_prob`, and switches in between at `switch_prob`, so
episodes see standing, every direction, and the transitions between them. `progress` measures
each step from where it started, which for a nearby goal is the distance covered along the held
direction. `configs/nearby_goal_manifest.yaml` is a starting point with untuned weights:

```bash
./isaaclab.sh -p scripts/train.py --algo AMP --manifest configs/nearby_goal_manifest.yaml
```

Drive it from the keyboard with `--joystick-mode nearby_goal`. The held arrow keys set the
direction (up is forward, left the robot's own left), and the red sphere shows the goal:

```bash
./isaaclab.sh -p scripts/play.py --checkpoint <path> --joystick --joystick-mode nearby_goal
```

The goal moves as soon as a key is pressed or released. The policy sees it on its next
observation, and how the robot actually moves is still up to the policy and physics. Play uses
the same code path as training, at the distance the checkpoint trained with; for a fixed-goal
checkpoint the distance defaults to 1.0 m. `--nearby-goal-distance` overrides it, and is also
checked against `goal_radius`. The goal turns with the robot and never arrives, so left/right
and back ask for sidestepping or walking backwards. A policy trained on forward-walking clips
may only manage those clumsily, or by curving.

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
