from .robots import RobotVisualizerFactory
from .tasks import TaskVisualizerFactory

import pandas as pd
import os

def generate_plots(data: pd.DataFrame, folder_path: str, task_name: str, robot_name: str) -> None:
    RVF = RobotVisualizerFactory.create(robot_name, data, folder_path)
    TVF = TaskVisualizerFactory.create(task_name, data, folder_path)
    # Create folder if needed
    os.makedirs(folder_path, exist_ok=True)
    # Generate the plots
    RVF.generate_plots()
    TVF.generate_plots()