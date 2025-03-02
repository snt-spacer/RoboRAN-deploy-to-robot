from .base_trajectory import BaseTrajectory, BaseTrajectoryCfg


class Registerable:
    """Registerable class.
    
    All classes that inherit from this class are automatically registered in the TrajectoryFactory.
    """

    def __init_subclass__(cls: BaseTrajectory) -> None:
        """Register the class in the factory.
        
        When a class inherits from this class, it is automatically registered in the factory.
        
        Args:
            cls (BaseTrajectory): The trajectory generator class to register.
        """
        cls_name = cls.__name__[:-10]  # Remove "Trajectory" from the class name
        TrajectoryFactory.register(cls_name, cls)


class TrajectoryFactory:
    """Trajectory factory class.
    
    The factory is used to create trajectory generator objects. Trajectory generators objects create
    trajectories with different shapes."""
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a trajectory generator class in the factory.

        Trajectory generator classes are used to generate trajectories.
        
        Args:
            name (str): The name of the trajectory generator class.
            sub_class (Registerable): The trajectory generator class to register.
        """
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name: str, *args, **kwargs) -> BaseTrajectory:
        """Create a trajectory generator object.
        
        Args:
            cls_name (str): The name of the trajectory generator class.
        """
        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered. Available modules: {cls.registry.keys()}")

        print("=============================================")
        print("Creating trajectory generator {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f" +{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls.registry[cls_name](*args, **kwargs)


class RegisterableCfg:
    """Registerable configuration class.

    All classes that inherit from this class are automatically registered in the TrajectoryCfgFactory.
    """

    def __init_subclass__(cls: BaseTrajectory) -> None:
        """Register the class in the factory.
        
        When a class inherits from this class, it is automatically registered in the factory.
        
        Args:
            cls (BaseTrajectory): The trajectory generator configuration class to register.
        """
        cls_name = cls.__name__[:-13]  # Remove "TrajectoryCfg" from the class name
        TrajectoryCfgFactory.register(cls_name, cls)


class TrajectoryCfgFactory:
    """Trajectory configuration factory class.

    The factory is used to create trajectory generator configuration objects. Trajectory generator configuration
    objects are used to configure the trajectory generator objects."""
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: RegisterableCfg) -> None:
        """Register a trajectory generator configuration class in the factory.

        Trajectory generator configuration classes are used to configure the trajectory generator classes.
        
        Args:
            name (str): The name of the trajectory generator configuration class.
            sub_class (RegisterableCfg): The trajectory generator configuration class to register.
        """
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name: str, *args, **kwargs) -> BaseTrajectoryCfg:
        """Create a trajectory generator configuration object.
        
        Args:
            cls_name (str): The name of the trajectory generator configuration class.
        """
        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered. Available modules: {cls.registry.keys()}")

        print("=============================================")
        print("Creating trajectory generator {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f" +{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls.registry[cls_name](*args, **kwargs)


# Add all trajectory generators to the factory
from .accelerating_sinewave_trajectory import AcceleratingSinewaveTrajectory
from .bernouilli_lemniscate_trajectory import BernouilliLemniscateTrajectory
from .besace_trajectory import BesaceTrajectory
from .circle_trajectory import CircleTrajectory
from .gerono_lemniscate_trajectory import GeronoLemniscateTrajectory
from .hippopede_trajectory import HippopedeTrajectory
from .hypotrochoid_trajectory import HypotrochoidTrajectory
from .infinite_square_trajectory import InfiniteSquareTrajectory
from .lissajous_trajectory import LissajousTrajectory
from .ngon_trajectory import NGonTrajectory
from .sinewave_trajectory import SinewaveTrajectory
from .square_trajectory import SquareTrajectory
from .spiral_trajectory import SpiralTrajectory

# Add all the trajectory generators configurations to the factory
from .accelerating_sinewave_trajectory import AcceleratingSinewaveTrajectoryCfg
from .bernouilli_lemniscate_trajectory import BernouilliLemniscateTrajectoryCfg
from .besace_trajectory import BesaceTrajectoryCfg
from .circle_trajectory import CircleTrajectoryCfg
from .gerono_lemniscate_trajectory import GeronoLemniscateTrajectory
from .hippopede_trajectory import HippopedeTrajectoryCfg
from .hypotrochoid_trajectory import HypotrochoidTrajectoryCfg
from .infinite_square_trajectory import InfiniteSquareTrajectoryCfg
from .lissajous_trajectory import LissajousTrajectoryCfg
from .ngon_trajectory import NGonTrajectoryCfg
from .sinewave_trajectory import SinewaveTrajectoryCfg
from .square_trajectory import SquareTrajectoryCfg
from .spiral_trajectory import SpiralTrajectoryCfg
