from .base_robot_visualizer import BaseRobotVisualizer

class Registerable:
    """Registerable class.
    
    All classes that inherit from this class are automatically registered in the RobotVisualizerFactory.
    """

    def __init_subclass__(cls: BaseRobotVisualizer) -> None:
        """Register the class in the factory.
        
        When a class inherits from this class, it is automatically registered in the factory.

        Args:
            cls (BaseRobotVisualizer): The robot visualizer class to register
        """
        cls_name = cls.__name__[:-10]  # Remove "Visualizer" from the class name
        RobotVisualizerFactory.register(cls_name, cls)


class RobotVisualizerFactory:
    """Robot visualizer factory class.

    The factory is used to create robot visualizer objects. Robot visualizer objects visualize data from a pandas
    DataFrame. They generate plots and save them in the same folder as the logged data.
    """
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a robot visualizer class in the factory.

        Robot visualizer classes are used to visualize data from a pandas DataFrame.

        Args:
            name (str): The name of the goal formater class.
            sub_class (Registerable): The goal formater class to register.
        """
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name:str, *args, **kwargs) -> BaseRobotVisualizer:
        """Create a robot visualizer object.

        Args:
            cls_name (str): The name of the goal formater class.
        """
        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered. Available modules: {cls.registry.keys()}")

        return cls.registry[cls_name](*args, **kwargs)

from .turtlebot_visualizer import TurtlebotVisualizer
from .kingfisher_visualizer import KingfisherVisualizer
from .leatherback_visualizer import LeatherbackVisualizer
from .floating_platform_visualizer import FloatingPlatformVisualizer
from .virtual_floating_platform_visualizer import VirtualFloatingPlatformVisualizer