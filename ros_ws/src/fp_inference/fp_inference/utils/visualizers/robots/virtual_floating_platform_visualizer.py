from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from . import BaseRobotVisualizer, Registerable
from colorsys import hsv_to_rgb

class VirtualFloatingPlatformVisualizer(BaseRobotVisualizer, Registerable):
    def __init__(self, data: pd.DataFrame, folder: str) -> None:
        super().__init__(data, folder)

    def compute_energy(self) -> None:
        dt = np.array(self._data['elapsed_time.s'].diff())
        dt[0] = dt[1]
        actions = [self._data['actions.t' + str(i)] for i in range(0, 8)]
        energy = (sum(actions) / 8.0) * dt
        return energy

    @BaseRobotVisualizer.register
    def plot_actions(self) -> None:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        for i in range(0, 8):
            ax.plot(self._data['elapsed_time.s'], self._data['actions.t' + str(i)], label='Action ' + str(i))
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Command')
        ax.set_title('Robot Commands')
        ax.legend()
        plt.savefig(f'{self._folder}/actions.png')

    @BaseRobotVisualizer.register
    def plot_actions_multi(self) -> None:
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Thruster Actions')
        for i in range(0, 8):
            ax = fig.add_subplot(4, 2, i+1)
            color = hsv_to_rgb(i/8.0, 1.0, 1.0)
            ax.plot(self._data['elapsed_time.s'], self._data['actions.t' + str(i)], label='Action ' + str(i), color=color)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Command')
            ax.set_title('Thruster ' + str(i) + ' Command')
        fig.tight_layout()
        plt.savefig(f'{self._folder}/actions_multi.png')

    @BaseRobotVisualizer.register
    def plot_commands(self):
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        for i in range(0, 8):
            ax.plot(self._data['elapsed_time.s'], self._data['commands.t' + str(i)], label='Thruster ' + str(i) + ' Command')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Command')
        ax.set_title('Robot Commands')
        ax.legend()
        plt.savefig(f'{self._folder}/commands.png')

    @BaseRobotVisualizer.register
    def plot_commands_multi(self):
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Thruster Commands')
        for i in range(0, 8):
            ax = fig.add_subplot(4, 2, i+1)
            color = hsv_to_rgb(i/8.0, 1.0, 1.0)
            ax.plot(self._data['elapsed_time.s'], self._data['commands.t' + str(i)], label='Thruster ' + str(i) + ' Command', color=color)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Command')
            ax.set_title('Thruster ' + str(i) + ' Command')
        fig.tight_layout()
        plt.savefig(f'{self._folder}/commands_multi.png')

    @BaseRobotVisualizer.register
    def plot_energy(self):
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self._data['elapsed_time.s'], self.compute_energy(), label='Energy')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy')
        ax.set_title('Robot Normalized Energy Consumption')
        ax.legend()
        plt.savefig(f'{self._folder}/normalized_energy.png')