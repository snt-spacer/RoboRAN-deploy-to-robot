from gymnasium import spaces
import torch


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
