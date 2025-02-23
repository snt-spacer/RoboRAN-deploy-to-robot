"""Observation formater package."""

from .base_formater import BaseFormater, BaseFormaterCfg  # noqa: F401, E402, F403


class Registerable:
    """Registerable class for factory pattern.

    All subclasses will be registered in the factory and must follow the naming convention of ending with "Formater".
    """

    def __init_subclass__(cls: BaseFormater) -> None:
        """Register the class in the factory. Remove "Formater" from the class name."""
        cls_name = cls.__name__[:-8]  # Remove "Formater" from the class name
        ObservationFormaterFactory.register(cls_name, cls)


class ObservationFormaterFactory:
    """Factory class for creating observation formaters.

    Observation formaters take in a state preprocessor,
    and using its standard interface, return an observation space and logs for the desired task.
    The output of the observation formater is the observation, it can be acquired by calling the
    `observation` property of the formater.
    """

    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a new observation formater class.

        Args:
            name (str): The name of the observation formater.
            sub_class (Registerable): The observation formater class.
        """
        # Check if the class is already registered
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name: str, *args, **kwargs) -> BaseFormater:
        """Create a new observation formater instance.

        Args:
            cls_name (str): The name of the observation formater to create.
        """
        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered. Available formaters: {cls.registry.keys()}")

        # Print the arguments of the formater
        print("=============================================")
        print("Creating observation formater {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f" +{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls.registry[cls_name](*args, **kwargs)


from .go_to_position_formater import GoToPositionFormater  # noqa: F401, E402, F403
from .go_to_pose_formater import GoToPoseFormater  # noqa: F401, E402, F403
from .go_through_position_formater import GoThroughPositionFormater  # noqa: F401, E402, F403
from .go_through_positions_formater import GoThroughPositionsFormater # noqa: F401, E402, F403
from .track_velocities_formater import TrackVelocitiesFormater # noqa: F401, E402, F403
