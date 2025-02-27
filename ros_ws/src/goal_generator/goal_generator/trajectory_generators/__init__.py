from .base_trajectory import BaseTrajectory, BaseTrajectoryCfg

class Registerable:
    def __init_subclass__(cls: BaseTrajectory) -> None:
        cls_name = cls.__name__[:-10]  # Remove "Trajectory" from the class name
        print("Factory")
        print("Registering class: ", cls_name)
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
        print("Factory")
        print("Registering class: ", cls_name)
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
from .circle_trajectory import CircleTrajectory
# Add all the trajectory generators configurations to the factory
from .circle_trajectory import CircleTrajectoryCfg