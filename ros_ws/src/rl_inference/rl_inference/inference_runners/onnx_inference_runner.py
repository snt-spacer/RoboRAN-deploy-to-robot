from . import Registerable, BaseInferenceRunner

from gymnasium import spaces
import onnxruntime as ort
import torch


class ONNXInferenceRunner(Registerable, BaseInferenceRunner):
    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "auto",
        *args,
        **kwargs,
    ) -> None:
        if device == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        self._provider = "CUDAExecutionProvider" if "cuda" in self._device else "CPUExecutionProvider"
        self._checkpoint_path = checkpoint_path

        self.build()

    def act(self, states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self._device == "cpu":
            output = self.ort_model.run(None, dict({"obs": states.numpy()}))[0]
            output = torch.from_numpy(output)
        # print(output)
        return output

    def load_model(self, path: str) -> None:
        """
        Loads the ONNX model from the given path.

        Args:
            path: The path to the ONNX model file.
        """
        self.ort_model = ort.InferenceSession(path, providers=[self._provider])

    def build(self) -> None:
        """
        Builds the ONNX inference runner by loading the model.
        """
        self.load_model(self._checkpoint_path)

    def reset(self) -> None:
        pass
