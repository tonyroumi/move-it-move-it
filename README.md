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
./isaaclab.sh -p scripts/train.py --task <task_name> --algorithm PPO
```

Play a checkpoint:

```bash
./isaaclab.sh -p scripts/play.py --task <task_name> --algorithm PPO --checkpoint <path>
```
