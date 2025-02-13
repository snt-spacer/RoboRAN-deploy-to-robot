import torch

class BaseInferenceRunner:
    def __init__(self, **args, **kwargs) -> None:
        raise NotImplementedError("Init method not implemented")

    def act(self, states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Act method not implemented")

    def load(self, path: str) -> None:
        raise NotImplementedError("Load method not implemented")

    def build(self, resume_path: str) -> None:
        raise NotImplementedError("Build method not implemented")