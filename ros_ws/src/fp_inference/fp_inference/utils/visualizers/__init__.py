from .robots import RobotVisualizerFactory
from .tasks import TaskVisualizerFactory

import pandas as pd

def generate_plots(data: pd.DataFrame, folder: str, task_name: str, robot_name: str) -> None:
    RVF = RobotVisualizerFactory.create(robot_name, data, folder)
    TVF = TaskVisualizerFactory.create(task_name, data, folder)
    RVF.generate_plots()
    TVF.generate_plots()