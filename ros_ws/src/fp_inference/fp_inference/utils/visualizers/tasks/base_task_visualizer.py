import pandas as pd

class AutoRegister:
    def __init_subclass__(cls, **kwargs):
        """Ensure each subclass gets its own independent registry."""
        super().__init_subclass__(**kwargs)
        cls._registry: dict[str, callable] = {}  # Unique for each subclass

        for name, value in cls.__dict__.items():
            # If an attribute is a function and has our marker, register it.
            if callable(value) and getattr(value, '_auto_register', False):
                cls._registry[name] = value

    @staticmethod
    def register(func: callable) -> callable:
        """
        Decorator that simply marks a function so that __init_subclass__
        knows it should be placed in the registry.
        """
        func._auto_register = True
        return func

    @classmethod
    def get_registered_methods(cls) -> dict[str, callable]:
        """Retrieve registered methods."""
        return cls._registry


class BaseTaskVisualizer(AutoRegister):
    def __init__(self, data: pd.DataFrame, folder: str) -> None:
        self._data = data
        self._folder = folder

    def generate_plots(self) -> None:
        for plot in self.get_registered_methods().values():
            plot(self)