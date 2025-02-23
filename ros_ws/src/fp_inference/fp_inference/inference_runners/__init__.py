from .base_inference_runner import BaseInferenceRunner


class Registerable:
    def __init_subclass__(cls: BaseInferenceRunner):
        cls_name = cls.__name__[:-15]  # Remove "InferenceRunner" from the class name
        InferenceRunnerFactory.register(cls_name, cls)


class InferenceRunnerFactory:
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name, *args, **kwargs) -> BaseInferenceRunner:

        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered. Available modules: {cls.registry.keys()}")

        print("=============================================")
        print("Creating inference runner {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f" +{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls.registry[cls_name](*args, **kwargs)


from .rlgame_inference import RLGamesInferenceRunner  # noqa: F401, E402, F403
from .skrl_inference import SKRLInferenceRunner  # noqa: F401, E402, F403
from .random_inference_runner import RandomInferenceRunner  # noqa: F401, E402, F403
from .onnx_inference_runner import ONNXInferenceRunner  # noqa: F401, E402, F403
