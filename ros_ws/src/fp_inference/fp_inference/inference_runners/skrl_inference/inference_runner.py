import skrl
from packaging import version
import yaml
from gymnasium import spaces
from fp_inference.inference_runners import Registerable


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


class SKRLInferenceRunner(Registerable):
    def __init__(
        self,
        logdir: str | None = None,
        observation_space: spaces.Space | None = None,
        action_space: spaces.Space | None = None,
        checkpoint_path: str | None = None,
        device: str = "auto",
        use_mix_precision: bool = False,
    ) -> None:
        """SKRL Inference Runner, used to perform inference using the skrl library.

        Args:
            logdir: The directory where the model is configuration and weights are stored.
            observation_space: The observation space of the agent.
            action_space: The action space of the agent.
            checkpoint_path: The path to the checkpoint to restore the model from.
            device: The device to run the inference on.
            use_mix_precision: Whether to use mixed precision for inference."""

        # Set the device & mixed precision
        if device == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        self._device_type = torch.device(self._device).type
        self._mixed_precision = use_mix_precision

        self._observation_space = observation_space
        self._action_space = action_space

        # Checkpoint modules to restore the weights of the models
        self.checkpoint_modules = {}

        # RNN initial state
        self._rnn_states = {"rnn": []}

        self.load_model(logdir, action_space, checkpoint_path)

        # set random seed
        set_seed(self._cfg.get("seed"))

    def load_model(
        self,
        log_dir: str | None = None,
        action_space: spaces.Space | None = None,
        checkpoint_path: str | None = None,
        **kwargs,
    ) -> None:
        """Load the RL model from a given path.

        Args:
            log_dir: The directory where the model is configuration and weights are stored.
            action_space: The action space of the agent.
            checkpoint_path: The path to the checkpoint to restore the model from."""

        # Get model and agent configuration
        print(log_dir)
        env_params_path = f"{log_dir}/params/env.yaml"
        print(env_params_path)
        with open(env_params_path) as f:
            env_cfg = yaml.load(f, Loader=yaml.FullLoader)
        print(env_cfg)

        agent_params_path = f"{log_dir}/params/agent.yaml"
        with open(agent_params_path) as f:
            agent_params = yaml.safe_load(f)

        self._env = env_cfg
        self._cfg = agent_params

        # Get the action space
        if action_space is None:
            raise ValueError("action_space must be provided")
        self._action_space = action_space

        # Get the checkpoint path
        self._checkpoint_path = (
            checkpoint_path if checkpoint_path is not None else f"{log_dir}/checkpoints/best_agent.pt"
        )

        # Build the model
        self.build()

    def load_weigths(self, path: str) -> None:
        """Load the model from the specified path

        The final storage device is determined by the constructor of the model

        Args:
            path: The path to the checkpoint to load the model from."""

        if version.parse(torch.__version__) >= version.parse("1.13"):
            modules = torch.load(path, map_location=self._device, weights_only=False)  # prevent torch:FutureWarning
        else:
            modules = torch.load(path, map_location=self._device)
        if type(modules) is dict:
            for name, data in modules.items():
                module = self.checkpoint_modules.get(name)
                print("module: ", module)
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
        """Initialize the RNN state for the model."""

        for size in enumerate(self._policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn_states["rnn"].append(torch.zeros(size, dtype=torch.float32, device=self._device))

    def build_actor(self, env: dict[str, Any], cfg: dict[str, Any], device: str) -> None:
        """Build the actor based on the agent class.

        Args:
            env: The environment configuration.
            cfg: The agent configuration.
            device: The device to run the inference on."""

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

    def ppo_act(self, states: torch.Tensor, timestep: int = 0, timesteps: int = 0, **kwargs) -> torch.Tensor:
        """Perform the PPO action.

        Args:
            states: The states to perform the action on.
            timestep: The current timestep.
            timesteps: The total number of timesteps.

        Returns:
            torch.Tensor: The action to perform."""

        # sample random actions
        if timestep < 0:
            return self._policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self._policy.act({"states": self._state_preprocessor(states)}, role="policy")

        return actions

    def ppo_rnn_act(self, states: torch.Tensor, timestep: int = 0, timesteps: int = 0, **kwargs) -> torch.Tensor:
        """Perform the PPO action with RNN.

        Args:
            states: The states to perform the action on.
            timestep: The current timestep.
            timesteps: The total number of timesteps.

        Returns:
            torch.Tensor: The action to perform."""

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

        return actions

    def sac_act(self, states: torch.Tensor, timestep: int = 0, timesteps: int = 0, **kwargs) -> torch.Tensor:
        """Perform the SAC action.

        Args:
            states: The states to perform the action on.
            timestep: The current timestep.
            timesteps: The total number of timesteps.

        Returns:
            torch.Tensor: The action to perform."""

        # sample random actions
        if timestep < 0:
            return self._policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, _, outputs = self._policy.act({"states": self._state_preprocessor(states)}, role="policy")

        return actions

    def sac_rnn_act(self, states: torch.Tensor, timestep: int = 0, timesteps: int = 0, **kwargs) -> torch.Tensor:
        """Perform the SAC action with RNN.

        Args:
            states: The states to perform the action on.
            timestep: The current timestep.
            timesteps: The total number of timesteps.

        Returns:
            torch.Tensor: The action to perform."""

        # sample random action
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

        return actions

    def build(self) -> None:
        """Build the runner. Generate the models, preprocessor, and the actor.

        Args:
            resume_path: The path to the checkpoint to restore the model from."""

        # Generate the models
        self._models = generate_models(
            copy.deepcopy(self._env), copy.deepcopy(self._cfg), self._action_space, self._device
        )
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
        self.load_weigths(self._checkpoint_path)
