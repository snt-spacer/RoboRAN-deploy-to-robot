from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from . import BaseRobotVisualizer, Registerable

class KingfisherVisualizer(BaseRobotVisualizer, Registerable):
    def __init__(self, data: pd.DataFrame, folder: str) -> None:
        super().__init__(data, folder)

    def compute_energy(self) -> None:
        dt = np.array(self._data['elapsed_time.s'].diff())
        dt[0] = dt[1]
        energy = ((self._data['actions.left'] + self._data['actions.right']) / 2) * dt
        return energy

    @BaseRobotVisualizer.register
    def plot_actions(self) -> None:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self._data['elapsed_time.s'], self._data['actions.left'], label='Left Command')
        ax.plot(self._data['elapsed_time.s'], self._data['actions.right'], label='Right Command')
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Command')
        ax.set_title('Robot Commands')
        ax.legend()
        plt.savefig(f'{self._folder}/commands.png')

    @BaseRobotVisualizer.register
    def plot_commands(self):
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self._data['elasped_time.s'], self._data['commands.left'], label='Left Command')
        ax.plot(self._data['elapsed_time.s'], self._data['commands.right'], label='Right Command')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Command')
        ax.set_title('Robot Commands')
        ax.legend()
        plt.savefig(f'{self._folder}/commands.png')

    @BaseRobotVisualizer.register
    def plot_energy(self):
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self._data['elapsed_time.s'], self.compute_energy(), label='Energy')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy')
        ax.set_title('Robot Normalized Energy Consumption')
        ax.legend()
        ax.grid(visible=True)
        plt.savefig(f'{self._folder}/normalized_energy.png')