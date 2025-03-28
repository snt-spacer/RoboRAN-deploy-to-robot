"""Observation formater package."""
import yaml
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

    _registry = {}
    _cfg_registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a new observation formater class.

        Args:
            name (str): The name of the observation formater.
            sub_class (Registerable): The observation formater class.
        """
        # Check if the class is already registered
        if name in cls._registry:
            raise ValueError(f"Module {name} already registered.")
        cls._registry[name] = sub_class

    @classmethod
    def register_cfg(cls, name: str, cfg_class: BaseFormaterCfg) -> None:
        if name in cls._registry:
            raise ValueError(f"Module {name} already registered.")
        cls._cfg_registry[name] = cfg_class

    @classmethod
    def create(cls, cls_name: str, robot_name: str, *args, **kwargs) -> BaseFormater:
        """Create a new observation formater instance.

        Args:
            cls_name (str): The name of the observation formater to create.
        """
        if cls_name not in cls._registry:
            raise ValueError(f"Module {cls_name} not registered. Available formaters: {cls._registry.keys()}")
        
        cls_cfg_name = cls_name + "Cfg"
        if cls_cfg_name not in cls._cfg_registry:
            raise ValueError(f"Module {cls_name} not registered. Available formaters: {cls._cfg_registry.keys()}")
        
        if "FloatingPlatform" in robot_name:
            robot_name = "floating_platform"
        elif "TurtleBot2" in robot_name:
            robot_name = "turtlebot2"
        elif "Kingfisher" in robot_name:
            robot_name = "kingfisher"
        else:
            raise ValueError(f"Robot {robot_name} not supported.")
        
        # Load cfg
        try:
            cfg_path = f'/RANS_DeployToRobot/ros_ws/src/rl_inference/config/{robot_name}/{cls_name}.yaml'
            with open(cfg_path, 'r') as file:
                cfg_data = yaml.safe_load(file)
                cfg_class = cls._cfg_registry[cls_cfg_name]
                kwargs["task_cfg"] = cfg_class(**cfg_data)
        except Exception as e:
            print(f"Error loading {cls_name} config: {e}")
            cfg_data = {}


        # Print the arguments of the formater
        print("=============================================")
        print("Creating observation formater {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f" +{key}: {value}") for key, value in kwargs.items()]
        print("Cfg Path:")
        print(f" {cfg_path}")
        print("=============================================")

        return cls._registry[cls_name](*args, **kwargs)
    
    def get_registry_keys(self) -> list[str]:
        return self._registry.keys()


from .go_to_position_formater import GoToPositionFormater, GoToPositionTaskCfg  # noqa: F401, E402, F403
from .go_to_pose_formater import GoToPoseFormater, GoToPoseTaskCfg  # noqa: F401, E402, F403
from .go_through_positions_formater import GoThroughPositionsFormater, GoThroughPositionsTaskCfg # noqa: F401, E402, F403
from .track_velocities_formater import TrackVelocitiesFormater, TrackVelocitiesFormaterCfg # noqa: F401, E402, F403

# Check why is not possible to use a similar class like registerable for the cfg classes todo automagic
ObservationFormaterFactory.register_cfg("GoToPositionCfg", GoToPositionTaskCfg)
ObservationFormaterFactory.register_cfg("GoToPoseCfg", GoToPoseTaskCfg)
ObservationFormaterFactory.register_cfg("GoThroughPositionsCfg", GoThroughPositionsTaskCfg)
ObservationFormaterFactory.register_cfg("TrackVelocitiesCfg", TrackVelocitiesFormaterCfg)
