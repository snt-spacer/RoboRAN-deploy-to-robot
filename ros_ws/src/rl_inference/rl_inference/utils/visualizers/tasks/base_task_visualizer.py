import pandas as pd
import numpy as np
import math

class AutoRegister:
    def __init_subclass__(cls, **kwargs):
        """Ensure each subclass gets its own independent registry."""
        super().__init_subclass__(**kwargs)
        cls._registry: dict[str, callable] = {}  # Unique for each subclass
        cls._video_registery: dict[str, callable] = {} # Unique for each subclass

        for name, value in cls.__dict__.items():
            # If an attribute is a function and has our marker, register it.
            if callable(value) and getattr(value, '_auto_register', False):
                cls._registry[name] = value
                if getattr(value, '_is_video', False):
                    cls._video_registery[name] = value

    @staticmethod
    def register(func: callable) -> callable:
        """Decorator that simply marks a function so that __init_subclass__
        knows it should be placed in the registry.
        """
        func._auto_register = True
        return func
    
    @staticmethod
    def register_video(func: callable) -> callable:
        """Decorator that simply marks a function so that __init_subclass__
        knows it is a video plot.
        """
        func._is_video = True
        return func

    @classmethod
    def get_registered_methods(cls, get_videos=False) -> dict[str, callable]:
        """Retrieve registered methods."""
        if get_videos:
            return cls._registry
        else:
            # Filter out the video methods
            return {k: v for k, v in cls._registry.items() if k not in cls._video_registery}


class BaseTaskVisualizer(AutoRegister):
    def __init__(self, data: pd.DataFrame, folder: str) -> None:
        self._data = data
        self._folder = folder

    def generate_plots(self, generate_videos:bool = False) -> None:
        for plot in self.get_registered_methods(get_videos=generate_videos).values():
            plot(self)

    def generate_zero_traj(self):
        x = np.array(self._data['position_world.x.m'])
        y = np.array(self._data['position_world.y.m'])
        # Compute the limits of the plot
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        # Add 30% padding
        dx = x_max - x_min
        dy = y_max - y_min
        x_min -= 0.15 * dx
        x_max += 0.15 * dx
        y_min -= 0.15 * dy
        y_max += 0.15 * dy
        # Equalize X and Y limits
        dx = x_max - x_min
        dy = y_max - y_min
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        dxy = max(dx, dy)
        x_min = x_center - dxy / 2
        x_max = x_center + dxy / 2
        y_min = y_center - dxy / 2
        y_max = y_center + dxy / 2
        return x, y, x_min, x_max, y_min, y_max

    @staticmethod
    def auto_ceil(x):
        a = math.ceil(math.log10(abs(x)))
        precision = -a + 1
        return round(x + 0.5 * 10**(-precision), precision)
    
    @staticmethod
    def auto_floor(x):
        a = math.ceil(math.log10(abs(x)))
        precision = -a + 1
        return round(x , precision)

    def generate_grid(self, n_major_cells, n_minor_cells, ax, limits, equalize=True, mode='floor'):
        # Major cells are roundish numbers i.e. 2, 1, 0.5, 0.2
        dx = limits[1] - limits[0]
        if mode == 'ceil':
            x_tick_size = self.auto_ceil(dx / n_major_cells)
        else:
            x_tick_size = self.auto_floor(dx / n_major_cells)
        dy = limits[3] - limits[2]
        if mode == 'ceil':
            y_tick_size = self.auto_ceil(dy / n_major_cells)
        else:
            y_tick_size = self.auto_floor(dy / n_major_cells)
        # Adjust the ticks to be the same for X and Y
        if equalize:
            tick_size = min(x_tick_size, y_tick_size)
            x_tick_size = tick_size
            y_tick_size = tick_size
        x_start_tick = math.floor(limits[0] / x_tick_size) * x_tick_size - x_tick_size
        x_end_tick = math.ceil(limits[1] / x_tick_size) * x_tick_size + x_tick_size
        x_range = np.arange(x_start_tick, x_end_tick + x_tick_size, x_tick_size)
        y_start_tick = math.floor(limits[2] / y_tick_size) * y_tick_size - y_tick_size
        y_end_tick = math.ceil(limits[3] / y_tick_size) * y_tick_size + y_tick_size
        y_range = np.arange(y_start_tick, y_end_tick + y_tick_size, y_tick_size)
        # Minor cells are 1/n of the major cells
        x_minor_tick_size = x_tick_size / n_minor_cells
        x_minor_range = np.arange(x_start_tick, x_end_tick, x_minor_tick_size)
        y_minor_tick_size = y_tick_size / n_minor_cells
        y_minor_range = np.arange(y_start_tick, y_end_tick, y_minor_tick_size)
        # Set the ticks
        ax.set_xticks(x_range)
        ax.set_yticks(y_range)
        ax.set_xticks(x_minor_range, minor=True)
        ax.set_yticks(y_minor_range, minor=True)
        # Set the grid
        ax.grid(which='major', color='black', linewidth=1)
        ax.grid(which='minor', color='gray', linewidth=0.5, linestyle='--')
    