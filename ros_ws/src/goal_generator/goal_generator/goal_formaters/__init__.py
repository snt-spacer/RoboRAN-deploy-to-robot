from .base_formater import BaseFormater


class Registerable:
    """Registerable class.
    
    All classes that inherit from this class are automatically registered in the GoalFormaterFactory.
    """

    def __init_subclass__(cls: BaseFormater) -> None:
        """Register the class in the factory.
        
        When a class inherits from this class, it is automatically registered in the factory.

        Args:
            cls (BaseFormater): The goal formater class to register
        """
        cls_name = cls.__name__[:-8]  # Remove "Formater" from the class name
        GoalFormaterFactory.register(cls_name, cls)


class GoalFormaterFactory:
    """Goal formater factory class.

    The factory is used to create goal formater objects. Goal formater objects format goals from a YAML file into ROS messages.
    """
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a goal formater class in the factory.

        Goal formater classes are used to format goals from a YAML file into ROS messages.

        Args:
            name (str): The name of the goal formater class.
            sub_class (Registerable): The goal formater class to register.
        """
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name:str, *args, **kwargs) -> BaseFormater:
        """Create a goal formater object.

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

from .go_through_positions_formater import GoThroughPositionsFormater
from .go_to_position_formater import GoToPositionFormater
from .go_to_pose_formater import GoToPoseFormater
from .track_velocities_formater import TrackVelocitiesFormater
