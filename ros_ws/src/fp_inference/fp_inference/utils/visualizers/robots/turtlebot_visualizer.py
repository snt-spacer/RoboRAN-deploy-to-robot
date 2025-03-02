from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from . import BaseRobotVisualizer, Registerable

class TurtlebotVisualizer(BaseRobotVisualizer, Registerable):
    def __init__(self, data: pd.DataFrame, folder: str) -> None:
        super().__init__(data, folder)

    def compute_energy(self) -> None:
        # Compute time difference
        dt = np.array(self._data['elapsed_time.s'].diff())
        dt[0] = dt[1]
        # Inverse kinematics:
        wheel_radius = 0.038
        wheel_base = 0.12
        linear_vel = self._data['commands.linear_velocity.m/s']
        angular_vel = self._data['commands.angular_velocity.rad/s']
        left_wheel_target_velocity = (linear_vel + angular_vel * wheel_base) / wheel_radius
        right_wheel_target_velocity = (linear_vel - angular_vel * wheel_base) / wheel_radius
        # Compute energy
        energy = ((left_wheel_target_velocity + right_wheel_target_velocity) / 2) * dt
        return energy

    @BaseRobotVisualizer.register
    def plot_actions(self) -> None:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self._data['elapsed_time.s'], self._data['actions.linear_velocity'], label='Linear Velocity Cmd')
        ax.plot(self._data['elapsed_time.s'], self._data['actions.angular_velocity'], label='Angular Velocity Cmd')
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
        ax.plot(self._data['elasped_time.s'], self._data['commands.linear_velocity.m/s'], label='Linear Velocity Cmd')
        ax.plot(self._data['elapsed_time.s'], self._data['commands.angular_velocity.rad/s'], label='Angular Velocity Cmd')
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
        plt.savefig(f'{self._folder}/normalized_energy.png')