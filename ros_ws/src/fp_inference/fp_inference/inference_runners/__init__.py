
from .base_inference_runner import BaseInferenceRunner

class Registerable:
    def __init_subclass__(cls: BaseInferenceRunner):
        cls_name = cls.__name__[:-15] # Remove "InferenceRunner" from the class name
        ObservationFormaterFactory.register(cls_name, cls)


class ObservationFormaterFactory:
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable):
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name, *args, **kwargs):

        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered.")

        print("=============================================")
        print("Creating Inference Runner {} with args:".format(cls_name))
        print("Args:")
        [print(arg) for arg in args]
        print("Kwargs:")
        [print(f"{key}: {value}") for key, value in kwargs.items()]
        print("=============================================")

        return cls.registry[cls_name](*args, **kwargs)
