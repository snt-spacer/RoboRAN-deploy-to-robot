from .base_inference_runner import BaseInferenceRunner


class Registerable:
    """
    Registerable is a base class that automatically registers subclasses of BaseInferenceRunner
    with the InferenceRunnerFactory. It removes the "InferenceRunner" suffix from the subclass name
    and uses the remaining part as the registration key.
    """
    def __init_subclass__(cls: BaseInferenceRunner):
        cls_name = cls.__name__[:-15]  # Remove "InferenceRunner" from the class name
        InferenceRunnerFactory.register(cls_name, cls)


class InferenceRunnerFactory:
    """
    InferenceRunnerFactory is a factory class that manages the registration and creation of
    inference runners. It provides methods to register, create, and retrieve registered runners.
    """

    _registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """
        Registers a subclass with a given name.

        Args:
            name: The name to register the subclass under.
            sub_class: The subclass to register.

        Raises:
            ValueError: If the name is already registered.
        """
        if name in cls._registry:
            raise ValueError(f"Module {name} already registered.")
        cls._registry[name] = sub_class

    @classmethod
    def create(cls, cls_name, *args, **kwargs) -> BaseInferenceRunner:
        """
        Creates an instance of a registered inference runner.

        Args:
            cls_name: The name of the registered runner.
            *args: Positional arguments to pass to the runner's constructor.
            **kwargs: Keyword arguments to pass to the runner's constructor.

        Returns:
            An instance of the registered inference runner.

        Raises:
            ValueError: If the name is not registered.
        """

        if cls_name not in cls._registry:
            raise ValueError(f"Module {cls_name} not registered. Available modules: {cls._registry.keys()}")

        print("=============================================")
        print("Creating inference runner {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f" +{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls._registry[cls_name](*args, **kwargs)
    
    def get_registry_keys(self) -> list[str]:
        return self._registry.keys()


from .rlgame_inference import RLGamesInferenceRunner  # noqa: F401, E402, F403
from .skrl_inference import SKRLInferenceRunner  # noqa: F401, E402, F403
from .random_inference_runner import RandomInferenceRunner  # noqa: F401, E402, F403
from .onnx_inference_runner import ONNXInferenceRunner  # noqa: F401, E402, F403
from .debug_vel_inference_runner import DebugVelInferenceRunner
