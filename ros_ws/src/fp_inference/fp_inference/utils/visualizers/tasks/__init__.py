from .base_task_visualizer import BaseTaskVisualizer

class Registerable:
    """Registerable class.
    
    All classes that inherit from this class are automatically registered in the TaskVisualizerFactory.
    """

    def __init_subclass__(cls: BaseTaskVisualizer) -> None:
        """Register the class in the factory.
        
        When a class inherits from this class, it is automatically registered in the factory.

        Args:
            cls (BaseTaskVisualizer): The task visualizer class to register
        """
        cls_name = cls.__name__[:-10]  # Remove "Visualizer" from the class name
        TaskVisualizerFactory.register(cls_name, cls)


class TaskVisualizerFactory:
    """Task visualizer factory class.

    The factory is used to create task visualizer objects. Task visualizer objects visualize data from a pandas
    DataFrame. They generate plots and save them in the same folder as the logged data.
    """
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a task visualizer class in the factory.

        Task visualizer classes are used to visualize data from a pandas DataFrame.

        Args:
            name (str): The name of the goal formater class.
            sub_class (Registerable): The goal formater class to register.
        """
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name:str, *args, **kwargs) -> BaseTaskVisualizer:
        """Create a task visualizer object.

        Args:
            cls_name (str): The name of the goal formater class.
        """
        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered. Available modules: {cls.registry.keys()}")

        print("=============================================")
        print("Creating goal formater {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f" +{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls.registry[cls_name](*args, **kwargs)

from .go_to_position_visualizer import GoToPositionVisualizer