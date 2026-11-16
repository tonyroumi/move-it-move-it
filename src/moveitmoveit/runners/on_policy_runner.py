from __future__ import annotations

from isaaclab.envs import DirectRLEnv

from moveitmoveit.agents import resolve_agent
from moveitmoveit.utils.logger import Logger


class OnPolicyRunner:
    """Generic on-policy training loop."""

    def __init__(
        self,
        *,
        cfg: dict,
        env: DirectRLEnv,
        logger: Logger,
    ):
        self.env = env
        self.logger = logger

        agent_cls, cfg_cls = resolve_agent(cfg["agent"]["class_type"])

        self.agent = agent_cls(cfg=cfg_cls(**cfg["agent"]), logger=self.logger)
        self.agent.init(self.env, cfg)

    def learn(self, timesteps: int) -> None:
        """Run the training loop."""
        num_envs = self.env.unwrapped.num_envs
        observations, infos = self.env.reset()

        timestep = 0
        while timestep < timesteps:
            self.agent.inference()

            with self.logger.timing("Collection Time"):
                for _ in range(self.agent.cfg.num_transitions_per_env):
                    actions = self.agent.act(observations)

                    observations, rewards, terminated, truncated, infos = self.env.step(actions)

                    self.agent.process_env_step(
                        next_observations=observations,
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        infos=infos
                    )

                    timestep += num_envs

                    self.logger.step(num_envs)

            self.agent.train()
            with self.logger.timing("Learning Time"):
                self.agent.update()

            self.logger.log(self.agent.write_checkpoint, mean_episode_reward=self.agent.mean_episode_reward)
