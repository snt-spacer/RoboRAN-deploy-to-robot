from . import TrajectoryCfgFactory, TrajectoryFactory

import numpy as np
from matplotlib import pyplot as plt
from colorsys import hsv_to_rgb

# Create accelerating sinewave trajectory
name = "AcceleratingSinewave"
cfg = {"amplitude": 1.0, "frequency": 1.0, "num_periods": 5, "acceleration_rate": 1.2, "max_amplitude": 2.0, "min_amplitude": 1.0}
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)
