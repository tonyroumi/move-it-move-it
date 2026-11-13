from dataclasses import dataclass

from moveitmoveit.agents.amp import AMPCfg


@dataclass(kw_only=True)
class HILCfg(AMPCfg):
    warm_start_steps: int = 0
    """Environment steps, summed over all envs like `runner.timesteps`, that every env trains on the
    tracking reward alone before the second half of the envs switch to AMP."""

    resample_commands: bool = True
    """Whether the env keeps drawing new commands. Off, the commands stay as they are."""

    sample_behaviors: bool = True
    """Whether training's key presses keep starting behaviors. Off, every behavior key stays released."""

    observe_commands: bool = True
    """Whether the policy and value function see the command and behavior slots, phase included. Off,
    those slots read 0; the observation keeps its size."""
