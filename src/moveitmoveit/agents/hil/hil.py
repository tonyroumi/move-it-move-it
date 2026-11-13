from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from moveitmoveit.utils.logger import Logger

from moveitmoveit.agents.amp import AMP
from moveitmoveit.agents.hil.hil_cfg import HILCfg
from moveitmoveit.agents.ppo import PPO


TRACKING_TERMS = ("motion_tracking", "heading_motion_tracking")
"""The manifest reward terms HIL counts as tracking; every other manifest reward term is a task reward."""


class HIL(AMP):
    """
    Hybrid Imitation Learning (HIL): one policy and value function, trained first with PPO on the
    manifest's tracking terms alone, then on two fixed halves of the envs.

    Until `warm_start_steps` env steps have completed every env earns only the tracking terms and
    the discriminator neither trains nor shapes the reward. After that, the first half of the env
    indices (the extra env of an odd count included) keep the tracking reward, and the second half
    earn what AMP would give them: the tracking and task terms, weighted by `goal_reward_lambda`, plus
    the style reward. The discriminator then trains on agent motion from the AMP half alone.
    """

    cfg: HILCfg

    def __init__(self, cfg: HILCfg, logger: Logger):
        super().__init__(cfg=cfg, logger=logger)

        self._env_steps = 0

    def init(self, env: DirectRLEnv, cfg: dict):
        reward_names = env.unwrapped.mdp.reward_names
        self._tracking_terms = [name for name in reward_names if name in TRACKING_TERMS]
        self._task_terms = [name for name in reward_names if name not in TRACKING_TERMS]
        if not self._tracking_terms:
            raise ValueError(
                f"HIL trains on the manifest's tracking reward ({' or '.join(TRACKING_TERMS)}), but the"
                f" manifest declares no tracking term; its reward terms are {reward_names}."
            )

        env.unwrapped.sample_commands = self.cfg.resample_commands
        env.unwrapped.sample_behaviors = self.cfg.sample_behaviors

        num_envs = env.unwrapped.num_envs
        self._amp_half = torch.arange(num_envs, device=env.unwrapped.device) >= (num_envs + 1) // 2

        super().init(env, cfg)

        self._half_reward_sums = torch.zeros(2, device=env.unwrapped.device)
        self._half_reward_counts = torch.zeros(2, device=env.unwrapped.device)

    @property
    def warm_starting(self) -> bool:
        """Whether every env still trains on the tracking reward alone."""
        return self._env_steps < self.cfg.warm_start_steps

    def _amp_envs(self) -> torch.Tensor:
        """The envs on AMP this step: none during the warm start, the second half after it."""
        return self._amp_half & (not self.warm_starting)

    def _preprocess_observations(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        if not self.cfg.observe_commands:
            observations = observations.clone()
            observations[..., -self._cmd_dim:] = 0.0
        return super()._preprocess_observations(observations, train=train)

    def _rewards(self, infos: dict, shape: torch.Size, amp_envs: torch.Tensor) -> torch.Tensor:
        """Each env's reward from the env's weighted per-term rewards, with the style reward on the AMP
        half recorded for logging, and zero elsewhere."""
        terms = infos["reward_terms"]
        tracking = sum(terms[name] for name in self._tracking_terms).view(shape)
        task = sum((terms[name] for name in self._task_terms), torch.zeros_like(tracking)).view(shape)

        style_reward, weighted_style_reward = self._style_reward(infos, shape)
        style_reward = torch.where(amp_envs.view(shape), style_reward, torch.zeros_like(style_reward))
        weighted_style_reward = torch.where(amp_envs.view(shape), weighted_style_reward, torch.zeros_like(weighted_style_reward))

        infos["reward_terms_raw"]["style"] = style_reward
        infos["reward_terms"]["style"] = weighted_style_reward

        amp_rewards = self.cfg.goal_reward_lambda * (tracking + task) + weighted_style_reward
        return torch.where(amp_envs.view(shape), amp_rewards, tracking)

    def process_env_step(
        self,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        amp_envs = self._amp_envs()
        rewards = self._rewards(infos, rewards.shape, amp_envs)

        self._amp_observation_buffer.append(infos["amp_obs"][amp_envs])

        halves = self._amp_half.long()
        self._half_reward_sums.index_add_(0, halves, rewards.view(-1))
        self._half_reward_counts.index_add_(0, halves, torch.ones_like(rewards.view(-1)))

        self._env_steps += rewards.shape[0]

        # AMP's own step would add the style reward to every env; the reward here already has it
        PPO.process_env_step(self, next_observations, rewards, terminated, truncated, infos)

    def update(self) -> None:
        self._log_phase()

        super().update()

    def _update_discriminator(self) -> None:
        # before the switch the buffer holds no agent motion: the discriminator waits for the AMP half
        if len(self._amp_observation_buffer) == 0:
            return

        super()._update_discriminator()

    def _log_phase(self) -> None:
        """Log the phase and each half's mean reward per step over the rollout, then restart them."""
        timestep = self.logger.timestep
        self.logger.track_data("HIL/Phase (0 warm start, 1 split)", float(not self.warm_starting), timestep)

        means = self._half_reward_sums / self._half_reward_counts.clamp_min(1)
        for index, half in enumerate(("PPO", "AMP")):
            if self._half_reward_counts[index] > 0:
                self.logger.track_data(f"HIL/{half} Half Reward per Step (mean)", means[index].item(), timestep)

        self._half_reward_sums.zero_()
        self._half_reward_counts.zero_()
