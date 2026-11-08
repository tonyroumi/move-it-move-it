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
