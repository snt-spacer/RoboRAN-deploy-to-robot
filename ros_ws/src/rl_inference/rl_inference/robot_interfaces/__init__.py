"""Robot interfaces package."""

from .base_robot_interface import BaseRobotInterface


class Registerable:
    """Registerable class for factory pattern.
    
    All subclasses will be registered in the factory and must follow the naming convention of ending with "Interface".
    """

    def __init_subclass__(cls: BaseRobotInterface) -> None:
        """Register the class in the factory. Remove "Interface" from the class name."""

        cls_name = cls.__name__[:-9]  # Remove "Interface" from the class name
        RobotInterfaceFactory.register(cls_name, cls)


class RobotInterfaceFactory:
    """Factory class for creating robot interfaces.
    
    Robot interfaces take in an action and convert it into the ROS message to send to the robot. The robot interface
    also logs the actions taken by the robot. It provides the action space for robot, as well as a logging interface.
    """

    _registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a new robot interface class.
        
        Args:
            name (str): The name of the robot interface.
            sub_class (Registerable): The robot interface class.
        """

        if name in cls._registry:
            raise ValueError(f"Module {name} already registered.")
        cls._registry[name] = sub_class

    @classmethod
    def create(cls, cls_name: str, *args, **kwargs) -> BaseRobotInterface:
        """Create a new robot interface instance.
        
        Args:
            cls_name (str): The name of the robot interface to create.
        """

        if cls_name not in cls._registry:
            raise ValueError(f"Module {cls_name} not registered. Available interfaces: {cls._registry.keys()}")

        # Print the arguments of the robot interface
        print("=============================================")
        print("Creating robot interface {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f" +{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls._registry[cls_name](*args, **kwargs)

    def get_registry_keys(self) -> list[str]:
        return self._registry.keys()
        

from .virtual_floating_platform import VirtualFloatingPlatformInterface  # noqa: F401, E402, F403
from .floating_platform import FloatingPlatformInterface  # noqa: F401, E402, F403
from .kingfisher import KingfisherInterface  # noqa: F401, E402, F403
from .turtlebot import TurtlebotInterface  # noqa: F401, E402, F403
from .leatherback import LeatherbackInterface  # noqa: F401, E402, F403