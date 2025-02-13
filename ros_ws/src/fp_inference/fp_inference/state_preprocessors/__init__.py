from .base_state_preprocessor import BaseStatePreProcessor

class Registerable:
    def __init_subclass__(cls: BaseStatePreProcessor):
        StatePreprocessorFactory.register(cls.__name__, cls)

class StatePreprocessorFactory:
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable):
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name, *args, **kwargs) -> BaseStatePreProcessor:

        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered.")

        print("=============================================")
        print("Creating state preprocessor {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f"{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls.registry[cls_name](*args, **kwargs)

from optitrack_state_preprocessor import OptitrackStatePreProcessor