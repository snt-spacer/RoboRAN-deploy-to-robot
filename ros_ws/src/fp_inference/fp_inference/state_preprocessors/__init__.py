"""State preprocessors for the robot state."""

from .base_state_preprocessor import BaseStatePreProcessor


class Registerable:
    """Registerable class for factory pattern.

    All subclasses will be registered in the factory and must follow the naming convention of ending with
    "StatePreProcessor".
    """

    def __init_subclass__(cls: BaseStatePreProcessor) -> None:
        """Register the class in the factory. Remove "StatePreProcessor" from the class name."""
        name = cls.__name__[:-17]  # Remove "StatePreProcessor" from the class name
        StatePreprocessorFactory.register(name, cls)


class StatePreprocessorFactory:
    """Factory class for creating state preprocessors.

    State preprocessors receive state information from a robot, and generate a standardized interface to query
    robot-state related information from. This is very similar to the ArticulationData object from IsaacLab.
    """

    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a new state preprocessor class.

        Args:
            name (str): The name of the state preprocessor.
            sub_class (Registerable): The state preprocessor class.
        """
        # Check if the class is already registered
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name: str, *args, **kwargs) -> BaseStatePreProcessor:
        """Create a new state preprocessor instance.

        Args:
            cls_name (str): The name of the state preprocessor to create.
        """
        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered. Available preprocessors: {cls.registry.keys()}")

        # Print the arguments of the state preprocessor
        print("=============================================")
        print("Creating state preprocessor {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f" +{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls.registry[cls_name](*args, **kwargs)


from .optitrack_state_preprocessor import OptitrackStatePreProcessor  # noqa: F401, E402, F403
from .odometry_state_preprocessor import OdometryStatePreProcessor  # noqa: F401, E402, F403
from .debug_state_preprocessor import DebugStatePreProcessor  # noqa: F401, E402, F403
from .odometry_global_state_preprocessor import OdometryGlobalStatePreProcessor  # noqa: F401, E402, F403
