from typing import Dict
import gymnasium
import gym
import numpy as np
import torch
import yaml

from rl_games.algos_torch.players import (
    BasicPpoPlayerContinuous,
    BasicPpoPlayerDiscrete,
)

class RLGamesInferenceRunner(BaseInferenceRunner):
    """
    This class implements a wrapper for the RLGames model."""

    def __init__(
        self,
        logdir: str | None = None,
        observation_space: spaces_in.Space | None = None,
        action_space: spaces_in.Space | None = None,
        checkpoint_path: str | None = None,
        device: str = "auto",
        use_mix_precision: bool = False,
        *args,
        **kwargs,
    ) -> None:
        # Set the device & mixed precision
        if device == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        self._device_type = torch.device(self._device).type
        self._mixed_precision = use_mix_precision
        self._observation_space = self._convert_observation_space(observation_space)
        self._action_space = self._convert_action_space(action_space)

        self.load_model(logdir, checkpoint_path)

    def _convert_action_space(self, action_space: gymnasium.spaces.Space) -> gym.spaces.Space:
        if isinstance(action_space, gymnasium.spaces.Discrete):
            return gym.spaces.Discrete(action_space.n)
        elif isinstance(action_space, gymnasium.spaces.Tuple):
            return gym.spaces.Tuple([gym.spaces.Discrete(n.n) for n in action_space.spaces])
        elif isinstance(action_space, gymnasium.spaces.Box):
            return gym.spaces.Box(-self._clip_actions, self._clip_actions, action_space.shape)
        elif isinstance(action_space, gymnasium.spaces.MultiDiscrete):
            return gym.spaces.Tuple([gym.spaces.Discrete(n) for n in action_space.nvec])
        
    def _convert_observation_space(self, observation_space: gymnasium.spaces.Space) -> gym.spaces.Space:
        if not isinstance(observation_space, gymnasium.spaces.Box):
            raise NotImplementedError(
                f"The RL-Games wrapper does not currently support observation space: '{type(observation_space)}'."
                f" If you need to support this, please modify the wrapper: {self.__class__.__name__},"
                " and if you are nice, please send a merge-request."
            )
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(None, None, observation_space.shape)


    def load_model(
        self,
        log_dir: str | None = None,
        checkpoint_path: str | None = None,
        **kwargs,
    ) -> None:

        # Get model and agent configuration
        env_params_path = f"{log_dir}/params/env.yaml"
        with open(env_params_path) as f:
            env_cfg = yaml.load(f, Loader=yaml.FullLoader)

        agent_params_path = f"{log_dir}/params/agent.yaml"
        with open(agent_params_path) as f:
            agent_params = yaml.safe_load(f)

        self._env = env_cfg
        self._cfg = agent_params

        self.buildModel()
        self.load_weigths(checkpoint_path)

    def buildModel(self) -> None:
        """
        Build the RLGames model."""

        if isinstance(self._action_space, gym.spaces.Tuple):
            self.player = BasicPpoPlayerDiscrete(
                self._cfg, self._observation_space, self._action_space, clip_actions=False, deterministic=True
            )
        else:
            self.player = BasicPpoPlayerContinuous(
                self._cfg, self._observation_space, self._action_space, clip_actions=False, deterministic=True
            )

    def load_weigths(self, model_name: str) -> None:
        """
        Restore the weights of the RLGames model.

        Args:
            model_name (str): A string containing the path to the checkpoint of an RLGames model matching the configuation file.
        """
        self.player.restore(model_name)

    def getAction(self, state, is_deterministic=True, **kwargs) -> np.ndarray:

        actions = (self.player.get_action(state, is_deterministic=is_deterministic))
        return actions