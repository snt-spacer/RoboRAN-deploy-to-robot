from .base_robot_interface import BaseRobotInterface


class Registerable:
    def __init_subclass__(cls: BaseRobotInterface):
        cls_name = cls.__name__[:-9]  # Remove "Interface" from the class name
        RobotInterfaceFactory.register(cls_name, cls)


class RobotInterfaceFactory:
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name, *args, **kwargs) -> BaseRobotInterface:

        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered.")

        print("=============================================")
        print("Creating robot interface {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f" +{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls.registry[cls_name](*args, **kwargs)


from .virtual_floating_platform import VirtualFloatingPlatformInterface  # noqa: F401, E402, F403
from .floating_platform import FloatingPlatformInterface  # noqa: F401, E402, F403
from .kingfisher import KingfisherInterface  # noqa: F401, E402, F403
from .turtlebot import TurtlebotInterface  # noqa: F401, E402, F403