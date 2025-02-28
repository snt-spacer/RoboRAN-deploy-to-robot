from .base_trajectory import BaseTrajectory, BaseTrajectoryCfg


class Registerable:
    def __init_subclass__(cls: BaseTrajectory) -> None:
        cls_name = cls.__name__[:-10]  # Remove "Trajectory" from the class name
        TrajectoryFactory.register(cls_name, cls)


class TrajectoryFactory:
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name, *args, **kwargs) -> BaseTrajectory:

        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered.")

        print("=============================================")
        print("Creating trajectory generator {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f" +{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls.registry[cls_name](*args, **kwargs)


class RegisterableCfg:
    def __init_subclass__(cls: BaseTrajectory) -> None:
        cls_name = cls.__name__[:-13]  # Remove "Trajectory" from the class name
        TrajectoryCfgFactory.register(cls_name, cls)


class TrajectoryCfgFactory:
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: RegisterableCfg) -> None:
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name, *args, **kwargs) -> BaseTrajectoryCfg:

        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered.")

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
from .infinite_square_trajectory import InfiniteSquareTrajectory
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
from .infinite_square_trajectory import InfiniteSquareTrajectoryCfg
from .ngon_trajectory import NGonTrajectoryCfg
from .sinewave_trajectory import SinewaveTrajectoryCfg
from .square_trajectory import SquareTrajectoryCfg
from .spiral_trajectory import SpiralTrajectoryCfg
