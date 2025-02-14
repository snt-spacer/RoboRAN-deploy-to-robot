from .import Registerable,BaseInferenceRunner


from gymnasium import spaces
import torch

class RandomInferenceRunner(Registerable, BaseInferenceRunner):
    def __init__(
        self,
        action_space: spaces.Space | None = None,
        device: str = "auto",
        **kwargs,
    ) -> None:
        """Random Inference Runner, used to perform inference using the skrl library.

        Args:
            action_space: The action space of the agent.
            device: The device to run the inference on."""
        
        # Set the device & mixed precision
        if device == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        self._device_type = torch.device(self._device).type

        self.load_model(action_space=action_space)

    def load_model(
        self,
        action_space: spaces.Space | None = None,
        **kwargs,
    ) -> None:
        """Load the RL model from a given path.

        Args:
            action_space: The action space of the agent."""

        # Get the action space
        if action_space is None:
            raise ValueError("action_space must be provided")
        self._action_space = action_space

        # Build the model
        self.build()

    def act(self, *args, **kwargs) -> torch.Tensor:
        """Act method for the runner.

        Args:
            states: The states to act on.

        Returns:
            The actions to take."""

        
        return torch.tensor(self._action_space.sample()).unsqueeze(0)

    def build(self) -> None:
        """Build the runner. Generate the models, preprocessor, and the actor.

        Args:
            resume_path: The path to the checkpoint to restore the model from."""

        pass

    def reset(self):
        pass