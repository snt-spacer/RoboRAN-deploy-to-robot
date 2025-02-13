from .base_formater import BaseFormater, BaseFormaterCfg

class Registerable:
    def __init_subclass__(cls: BaseFormater):
        cls_name = cls.__name__[:-8] # Remove "Formater" from the class name
        ObservationFormaterFactory.register(cls_name, cls)


class ObservationFormaterFactory:
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable):
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name, *args, **kwargs) -> BaseFormater:

        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered.")

        print("=============================================")
        print("Creating observation formater {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f"{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls.registry[cls_name](*args, **kwargs)

from .go_to_position_formater import GoToPositionFormater
from .go_to_pose_formater import GoToPoseFormater