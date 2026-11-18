# CLAUDE.md

## Project

`moveitmoveit` is an IsaacLab RL environment for learning task-driven humanoid behaviors guided by
reference motions (AMP and DeepMimic-style tracking). A single YAML **manifest** declares the
motions, their commands, rewards, terminations, behavior triggers, and AMP style weight.

## Layout

- `src/moveitmoveit/env.py`, `env_cfg.py` — the gym env `MoveIt-Humanoid-v0` (registered in `__init__.py`).
- `src/moveitmoveit/mdp/` — observation, command, reward, and termination terms. Each is a plain
  function registered with a decorator (`@reward(...)`, `@termination(...)`, `@command(...)`,
  `@observation(dim)`); manifests reference them by function name.
- `src/moveitmoveit/motion/` — manifest parsing (`manifest.py`, `specs.py`), motion library, behaviors.
- `src/moveitmoveit/agents/` — PPO and AMP trainers; `runners/`, `storage/`, `models/` support them.
- `configs/agents/*.yaml` — algorithm hyperparameters. `configs/manifests/*.yaml` — experiments.
- `scripts/` — `train.py`, `play.py`, `play_motion.py`, plus helper shell scripts.
- `data/` (motion `.npz` clips) and `logs/` (checkpoints, TensorBoard) are local and not committed.

## Running

**Never run IsaacLab.** Do not invoke `isaaclab.sh`, the scripts below, or any wrapper that launches
it (`run_seq.sh`, `scripts/play_latest.sh`, `scripts/record_motions.py`). The user runs these
themselves; the commands are listed for reference only.

If a change needs a sim run to verify, give the user the exact command to run instead.

## Conventions

- Keep docstrings short and descriptive of *what* a value is (match existing style).
- There is no test suite or pytest config.

## Working practices for agents

- **Plan non-trivial changes.** For multi-file work, outline the approach before writing code and
  confirm with the user if the design is ambiguous.
- **Small, focused diffs.** Change only what the task needs; no drive-by refactors, renames, or
  reformatting of unrelated code.
- **Ask when blocked.** If a requirement is unclear or a change affects training behavior (reward
  scales, observation layout, checkpoint compatibility), flag it rather than guessing.

## Commits and branches

- Work on a feature branch; do not commit directly to `main`.
- Keep commits focused and atomic.
- Use an imperative, capitalized commit subject with no trailing period.
- Inspect staged changes before committing.
- Do not add AI co-author or attribution lines.
- Prefer follow-up commits over amending commits while addressing review feedback.
