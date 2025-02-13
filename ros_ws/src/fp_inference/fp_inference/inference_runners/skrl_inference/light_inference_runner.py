import skrl
from packaging import version
import yaml
from gymnasium import spaces

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

import copy
import torch
from collections.abc import Mapping
from typing import Any

from skrl import logger
from skrl.utils import set_seed

from . import generate_models, get_state_preprocessor


class LightInferenceRunner:

    def __init__(
        self, logdir: str | None = None, device: str = "auto", use_mix_precision: bool = False
    ) -> None:
        """Experiment runner

        Class that configures and instantiates skrl components to execute training/evaluation workflows in a few lines of code

        :param env: Environment to train on
        :param cfg: Runner configuration
        """

        if logdir is not None:
            self.load_model(logdir)
        else:
            raise ValueError("logdir must be provided")

        # Set the device & mixed precision
        if device == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        self._device_type = torch.device(self._device).type
        self._mixed_precision = use_mix_precision

        # Checkpoint modules to restore the weights of the models
        self.checkpoint_modules = {}

        # RNN initial state
        self._rnn_states = {"rnn": []}

        # set random seed
        set_seed(self._cfg.get("seed"))

    def load_model(self, log_dir: str | None = None, action_space: spaces | None = None, **kwargs) -> None:
        """Load the RL model from a given path."""

        env_params_path = f"{log_dir}/params/env.yaml"
        with open(env_params_path) as f:
            env_cfg = yaml.load(f, Loader=yaml.FullLoader)

        agent_params_path = f"{log_dir}/params/agent.yaml"
        with open(agent_params_path) as f:
            agent_params = yaml.safe_load(f)

        self._env = env_cfg
        self._cfg = agent_params

        if spaces is None:
            raise ValueError("action_space must be provided")

        self.load_weigths(f"{log_dir}/checkpoints/best_agent.pt")
        self.action_space = spaces.Tuple([spaces.Discrete(2)] * 8)

    def load_weigths(self, path: str) -> None:
        """Load the model from the specified path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str
        """
        if version.parse(torch.__version__) >= version.parse("1.13"):
            modules = torch.load(path, map_location=self._device, weights_only=False)  # prevent torch:FutureWarning
        else:
            modules = torch.load(path, map_location=self._device)
        if type(modules) is dict:
            for name, data in modules.items():
                module = self.checkpoint_modules.get(name)
                if module is not None:
                    if hasattr(module, "load_state_dict"):
                        module.load_state_dict(data)
                        if hasattr(module, "eval"):
                            module.eval()
                    else:
                        raise NotImplementedError
                else:
                    logger.warning(f"Cannot load the {name} module. The agent doesn't have such an instance")

    def initialize_rnn_state(self) -> None:
        for size in enumerate(self._policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn_states["rnn"].append(torch.zeros(size, dtype=torch.float32, device=self._device))

    def build_actor(self, env: dict[str, Any], cfg: dict[str, Any], device: str):

        agent_class = cfg.get("agent", {}).get("class", "").lower()

        # Assign the act method based on the agent class
        if agent_class == "ppo":
            if "rnn" in self._policy.get_specification():
                self.initialize_rnn_state()
                self.act = self.ppo_rnn_act
            else:
                self.act = self.ppo_act
        elif agent_class == "sac":
            if "rnn" in self._policy.get_specification():
                self.initialize_rnn_state()
                self.act = self.sac_rnn_act
            else:
                self.act = self.sac_act

    def ppo_act(
        self, states: torch.Tensor, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, Mapping[str, torch.Tensor | Any]]:
        # sample random actions
        if timestep < 0:
            return self._policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self._policy.act({"states": self._state_preprocessor(states)}, role="policy")

        return actions, log_prob, outputs

    def ppo_rnn_act(
        self, states: torch.Tensor, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, Mapping[str, torch.Tensor | Any]]:
        # sample random actions
        if timestep < 0:
            return self._policy.random_act(
                {"states": self._state_preprocessor(states), **self._rnn_states}, role="policy"
            )

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self._policy.act(
                {"states": self._state_preprocessor(states), **self._rnn_states}, role="policy"
            )

        # Update the rnn states
        self._rnn_states["rnn"] = outputs.get("rnn", [])

        return actions, log_prob, outputs

    def sac_act(
        self, states: torch.Tensor, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, Mapping[str, torch.Tensor | Any]]:
        # sample random actions
        if timestep < 0:
            return self._policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, _, outputs = self._policy.act({"states": self._state_preprocessor(states)}, role="policy")

        return actions, None, outputs

    def sac_rnn_act(
        self, states: torch.Tensor, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, Mapping[str, torch.Tensor | Any]]:
        # sample random actions
        if timestep < 0:
            return self._policy.random_act(
                {"states": self._state_preprocessor(states), **self._rnn_states}, role="policy"
            )

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, _, outputs = self._policy.act(
                {"states": self._state_preprocessor(states), **self._rnn_states}, role="policy"
            )

        # Update the rnn states
        self._rnn_states["rnn"] = outputs.get("rnn", [])

        return actions, None, outputs

    def build(self, resume_path: str) -> None:
        """Build the runner. Generate the models, preprocessor, and the actor.

        :param resume_path: Path to the checkpoint to resume from."""

        # Generate the models
        self._models = generate_models(copy.deepcopy(self._env), copy.deepcopy(self._cfg), self._device)
        # Generate the preprocessor
        self._state_preprocessor = get_state_preprocessor(
            copy.deepcopy(self._env), copy.deepcopy(self._cfg), self._device
        )
        # Get the policy model
        self._policy = self._models["policy"]
        # Set the policy model to evaluation mode
        self._policy.set_mode("eval")
        # Generate the act method
        self.build_actor(copy.deepcopy(self._env), copy.deepcopy(self._cfg), self._device)
        # Add the modules to the list of checpointed modules
        self.checkpoint_modules = {"policy": self._policy, "state_preprocessor": self._state_preprocessor}
        # Fetch the weights from the checkpoint
        self.load(resume_path)
