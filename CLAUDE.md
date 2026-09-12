# Project Overview

`moveitmoveit` is a lightweight motion learning framework built around IsaacLab.
It implements RL algorithms from scratch and trains a single humanoid model on
a variety of motions/behaviors, where each motion's rewards, commands, and
termination conditions are declared in a **motion manifest** (YAML) rather than
hardcoded per-behavior.

Current focus:
- Implement PPO and AMP (Adversarial Motion Priors) from scratch
- Train a single `MotionLearningEnv(DirectRLEnv)` across multiple reward schemes and motions
- Drive per-motion behavior via declarative motion manifests, not env subclasses
- Maintain explicit control over rollout storage, GAE, PPO updates,
  discriminator training, and diagnostics

# Architecture

- `moveitmoveit/agents/`
  - RL agents: PPO, AMP
- `moveitmoveit/models/`
  - neural network architectures such as MLP and GaussianMLP
- `moveitmoveit/storage/`
  - rollout storage, `RolloutBuffer`/`CircularBuffer` for discriminator training
- `moveitmoveit/env.py` / `moveitmoveit/env_cfg.py`
  - the single `MotionLearningEnv(DirectRLEnv)` — no `envs/` package, since there is and will only
    ever be one environment (the same robot, driven by the motion manifest)
- `moveitmoveit/motion_manager.py`
  - per-motion command/reward/termination tables built from the motion manifest, per-env clip assignment
- `moveitmoveit/motion/`
  - `motion_loader.py` — reference motion clip loading and state sampling
  - `motion_viewer.py` — standalone motion clip visualization
- `scripts/train.py`
  - CLI entry point: `--algo {ppo,amp}`, `--reward {tracking,joystick,joystick-shaped,manifest}`, `--motions <name(s)>`
  - single registered gym task `MoveIt-Humanoid-v0` (no per-combination task IDs)

IsaacLab environments use `DirectRLEnv`. Package uses src-layout
(`src/moveitmoveit/`), installed via `pip install -e .`; imports always use the
`moveitmoveit.env`/`moveitmoveit.motion.*` (etc.) namespace, never relative package-root imports.

## Motion Manifest

Each motion has a manifest YAML (`data/motions/<motion>.yaml`) that
declares its reward composition and termination condition.
# Coding Conventions

- Python 3.12
- PyTorch
- Prefer explicit tensor shapes.
- Include tensor shapes in comments when operations are non-obvious.
- Avoid unnecessary abstractions.
- Do not modify unrelated code.
- Preserve existing type hints.
- Prefer small focused classes/functions.
- New per-motion behavior should be expressed as a manifest YAML + registered
  reward terms, not a new `RewardScheme`/`CommandGenerator`/env subclass, unless
  the behavior genuinely needs new mechanics the registry can't express.

# Environment

IsaacLab 3.0
Isaac Sim 6.0
PyTorch
Ubuntu Linux

Run IsaacLab Python scripts through:

./isaaclab.sh -p <script>

# Working Guidelines

## Scope Control

- Do exactly what was requested and no more.
- Make the smallest change necessary to satisfy the request.
- Do not refactor, rename, reorganize, optimize, or clean up unrelated code.
- Do not change APIs, interfaces, tensor shapes, configuration structure, or behavior unless explicitly requested.
- Do not introduce new abstractions, helper classes, utilities, or dependencies unless they are necessary for the requested change.
- Do not modify nearby code merely because it could be improved.
- Preserve the existing coding style and architecture unless the task specifically asks to change them.

If you notice an unrelated bug, design issue, or possible improvement:
- Mention it separately.
- Do not fix it unless explicitly asked.

If the requested change appears to require a broader modification than expected:
- Explain why.
- Describe the minimum broader change required.
- Do not proceed with the broader change unless it is clearly necessary to complete the request.

## Before Editing

Before making significant changes:

1. Inspect the relevant implementation and its immediate dependencies.
2. Determine the exact scope of the requested change.
3. Identify important tensor shapes, data flow, and behavioral assumptions.
4. Prefer modifying existing code over introducing new architecture.
5. Make the smallest necessary modification.

## Behavioral Preservation

Unless explicitly requested otherwise:

- Preserve existing behavior outside the requested change.
- Preserve function signatures and public APIs.
- Preserve configuration defaults.
- Preserve tensor shapes and device/dtype behavior.
- Preserve existing logging and diagnostics.
- Preserve comments and documentation that remain correct.
- Do not silently change algorithmic semantics.

## Response Discipline

When asked to implement a specific change:

- Focus the response on that change.
- Do not provide a rewritten version of an entire file when a localized patch is sufficient.
- Do not propose multiple alternative architectures unless asked.
- Clearly identify any assumptions you had to make.