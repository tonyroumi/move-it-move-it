# Project Overview

`moveitmoveit` is a lightweight motion learning framework built around IsaacLab.
It implements RL algorithms from scratch and trains a single humanoid model on
a variety of motions/behaviors, where each motion's rewards, commands, and
termination conditions are declared in a **motion manifest** (YAML) rather than
hardcoded per-behavior.

# Architecture

- `moveitmoveit/agents/`
  - RL agents: PPO, AMP
- `moveitmoveit/models/`
  - neural network architectures such as MLP and GaussianMLP
- `moveitmoveit/storage/`
  - rollout storage, `RolloutBuffer`/`CircularBuffer` 
- `moveitmoveit/env.py` / `moveitmoveit/env_cfg.py`
  - the single `MotionLearningEnv(DirectRLEnv)` — no `envs/` package, since there is and will only
    ever be one environment (the same robot, driven by the motion manifest)
- `moveitmoveit/manifest.py`
  - parses and validates the motion manifest, with no torch dependency
- `moveitmoveit/motion_set.py`
  - `MotionSet`: per-motion command/selection tables built from the motion manifest, and the `MotionLibrary` holding their clips. Per-env clip assignment lives on the env
- `moveitmoveit/motion/`
  - `library.py` — `MotionLibrary`: reference motion clip loading and state sampling by motion id
  - `reference.py` — `ReferenceMotions`: what the env reset, tracking and AMP consumers ask of the reference motion
  - `motion_viewer.py` — standalone motion clip visualization

## Motion Manifest

The manifest YAML (`configs/manifest.yaml`) declares each motion's clip,
selection weight, commands, reward composition and termination condition.
# Coding Conventions

- Python 3.12
- PyTorch
- Avoid unnecessary abstractions.
- Do not modify unrelated code.
- Preserve existing type hints.
- Prefer small focused classes/functions.

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

1. Inspect the relevant implementation and its dependencies.
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
- Do not propose multiple alternative architectures unless asked.
- Clearly identify any assumptions you had to make.
