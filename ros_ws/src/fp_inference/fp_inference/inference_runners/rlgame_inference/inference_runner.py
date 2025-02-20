from typing import Dict
from gymnasium import spaces as spaces_in
from gym import spaces
import numpy as np
import torch
import yaml

from rl_games.algos_torch.players import (
    BasicPpoPlayerContinuous,
    BasicPpoPlayerDiscrete,
)

class BaseInferenceRunner:
    def __init__(
        self,
        logdir: str | None = None,
        action_space: spaces.Space | None = None,
        checkpoint_path: str | None = None,
        device: str = "auto",
        use_mix_precision: bool = False,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError("Init method not implemented")

    def act(self, states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Act method not implemented")

    def load_model(self, path: str) -> None:
        raise NotImplementedError("Load method not implemented")

    def build(self) -> None:
        raise NotImplementedError("Build method not implemented")

    def reset(self) -> None:
        raise NotImplementedError("Reset method not implemented")


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
        self._observation_space = observation_space
        self._action_space = action_space

        self.load_model(logdir, checkpoint_path)


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

        if isinstance(action_space, spaces_in.Discrete) or isinstance(action_space, spaces_in.MultiDiscrete):
            self.player = BasicPpoPlayerDiscrete(
                self._cfg, obs_space, action_space, clip_actions=False, deterministic=True
            )
        else:
            self.player = BasicPpoPlayerContinuous(
                self._cfg, obs_space, action_space, clip_actions=False, deterministic=True
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