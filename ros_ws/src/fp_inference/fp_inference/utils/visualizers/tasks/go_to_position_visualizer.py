from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import math

from . import BaseTaskVisualizer, Registerable

class GoToPositionVisualizer(BaseTaskVisualizer, Registerable):
    def __init__(self, data: pd.DataFrame, folder: str) -> None:
        super().__init__(data, folder)

    @staticmethod
    def auto_ceil(x):
        a = math.ceil(math.log10(abs(x)))
        precision = -a + 1
        return round(x + 0.5 * 10**(-precision), precision)

    def generate_grid(self, n_major_cells, n_minor_cells, ax, limits):
        # Major cells are roundish numbers i.e. 2, 1, 0.5, 0.2
        dx = limits[1] - limits[0]
        x_tick_size = self.auto_ceil(dx / n_major_cells)
        x_start_tick = math.floor(limits[0] / x_tick_size) * x_tick_size
        x_end_tick = math.ceil(limits[1] / x_tick_size) * x_tick_size
        x_range = np.arange(x_start_tick, x_end_tick + x_tick_size, x_tick_size)
        dy = limits[3] - limits[2]
        y_tick_size = self.auto_ceil(dy / n_major_cells)
        y_start_tick = math.floor(limits[2] / y_tick_size) * y_tick_size
        y_end_tick = math.ceil(limits[3] / y_tick_size) * y_tick_size
        y_range = np.arange(y_start_tick, y_end_tick + y_tick_size, y_tick_size)
        # Minor cells are 1/n of the major cells
        x_minor_tick_size = x_tick_size / n_minor_cells
        x_minor_range = np.arange(x_start_tick, x_end_tick + x_minor_tick_size, x_minor_tick_size)
        y_minor_tick_size = y_tick_size / n_minor_cells
        y_minor_range = np.arange(y_start_tick, y_end_tick + y_minor_tick_size, y_minor_tick_size)
        # Set the ticks
        ax.set_xticks(x_range)
        ax.set_yticks(y_range)
        ax.set_xticks(x_minor_range, minor=True)
        ax.set_yticks(y_minor_range, minor=True)
        # Set the grid
        ax.grid(which='major', color='black', linewidth=1)
        ax.grid(which='minor', color='gray', linewidth=0.5, linestyle='--')
    

    def generate_zero_traj(self):
        x = np.array(self._data['position_world.x.m'])
        y = np.array(self._data['position_world.y.m'])
        # Compute the limits of the plot
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        # Add 30% padding
        dx = x_max - x_min
        dy = y_max - y_min
        x_min -= np.floor(0.15 * dx)
        x_max += np.floor(0.15 * dx)
        y_min -= np.ceil(0.15 * dy)
        y_max += np.ceil(0.15 * dy)
        # Equalize X and Y limits
        dx = x_max - x_min
        dy = y_max - y_min
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        dxy = max(dx, dy)
        x_min = np.ceil(x_center - dxy / 2)
        x_max = np.floor(x_center + dxy / 2)
        y_min = np.floor(y_center - dxy / 2)
        y_max = np.ceil(y_center + dxy / 2)
        return x, y, x_min, x_max, y_min, y_max

    @BaseTaskVisualizer.register
    def plot_trajectory(self) -> None:
        fig = plt.figure(figsize=(8,8))
        # Compute the limits of the plot
        x, y, x_min, x_max, y_min, y_max = self.generate_zero_traj()
        # Plot the trajectory of the robot
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, label='Robot Trajectory', color='royalblue', zorder=3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectory')
        # Add the goal position
        target_pos_x = self._data['target_position.x.m'].iloc[-1]
        target_pos_y = self._data['target_position.y.m'].iloc[-1]
        plt.scatter(target_pos_x, target_pos_y, color='r', facecolor='none', label='Goal Position', zorder=4)
        self.generate_grid(8, 4, ax, [x_min, x_max, y_min, y_max])
        # Set scientific notation
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax.axis('equal')
        ax.legend()
        plt.savefig(f'{self._folder}/trajectory.png')
        plt.close(fig)

    @BaseTaskVisualizer.register
    def plot_trajectory_with_heading(self) -> None:
        fig = plt.figure(figsize=(8,8))
        # Compute the limits of the plot
        x, y, x_min, x_max, y_min, y_max = self.generate_zero_traj()
        # Plot the trajectory of the robot
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, label='Robot Trajectory', color='royalblue', zorder=3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectory with Heading')
        self.generate_grid(8, 4, ax, [x_min, x_max, y_min, y_max])
        # Set scientific notation
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        # Add the goal position
        target_pos_x = self._data['target_position.x.m'].iloc[-1]
        target_pos_y = self._data['target_position.y.m'].iloc[-1]
        plt.scatter(target_pos_x, target_pos_y, color='r', facecolor='none', label='Goal Position', zorder=4)
        # Add the legend before adding the robot heading
        ax.legend()
        # Add the robot heading 20 points only
        idx = np.arange(0, len(self._data), len(self._data) // 20)
        ax.quiver(x[idx], y[idx], 
                  np.cos(self._data['heading_world.rad'].iloc[idx]), np.sin(self._data['heading_world.rad'].iloc[idx]),
                  color='royalblue', label='Robot Heading', zorder=3)
        ax.axis('equal')
        plt.savefig(f'{self._folder}/trajectory_with_heading.png')
        plt.close(fig)

    @BaseTaskVisualizer.register
    def make_trajectory_video(self):
        # Compute the limits of the plot
        x, y, x_min, x_max, y_min, y_max = self.generate_zero_traj()
        # Make a blank canvas
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectory with Heading')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axis('equal')
        # Make the update function
        target_pos_x = self._data['target_position.x.m'].iloc[-1]
        target_pos_y = self._data['target_position.y.m'].iloc[-1]
        def update_trajectory(i):
            ax.clear()
            ax.plot(x,y, label='Robot Trajectory', color='b', zorder=3)
            ax.scatter(target_pos_x, target_pos_y, color='r', facecolor='none', label='Goal Position', zorder=4)
            ax.quiver(x[i], y[i], 
                      np.cos(self._data['heading_world.rad'].iloc[i]), np.sin(self._data['heading_world.rad'].iloc[i]),
                      color='b', label='Robot Heading',zorder=3)
            self.generate_grid(8, 4, ax, [x_min, x_max, y_min, y_max])
            # Set scientific notation
            ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.axis('equal')

        ani = animation.FuncAnimation(fig, update_trajectory, frames=len(self._data), interval=5)
        ani.save(f'{self._folder}/trajectory.mp4')
        plt.close(fig)
        

    @BaseTaskVisualizer.register
    def plot_position_error_with_helpers(self):
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self._data['elapsed_time.s'], self._data['distance_error.m'], label='Distance to Goal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (m)')
        ax.set_title('Position Error')
        ax.grid(visible=True)
        ax.grid(linestyle = '--', linewidth = 0.5)

        ax.legend()
        # Draw a horizontal line at 0.25m
        ax.axhline(y=0.25, color='k', linestyle='--', label='25cm threshold')
        # Draw a horizontal line at 0.10m
        ax.axhline(y=0.10, color='k', linestyle='--', label='10cm threshold')
        # Draw a horizontal line at 0.05m
        ax.axhline(y=0.05, color='k', linestyle='--', label='5cm threshold')
        plt.savefig(f'{self._folder}/position_error.png')
        plt.close(fig)

    @BaseTaskVisualizer.register
    def plot_position_error_with_helpers_log(self):
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self._data['elapsed_time.s'], self._data['distance_error.m'], label='Distance to Goal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (m)')
        ax.set_title('Position Error')
        ax.grid(visible=True)
        ax.legend()
        # Draw a horizontal line at 0.25m
        ax.axhline(y=0.25, color='grey', linestyle='--', label='25cm threshold')
        # Draw a horizontal line at 0.10m
        ax.axhline(y=0.10, color='grey', linestyle='--', label='10cm threshold')
        # Draw a horizontal line at 0.05m
        ax.axhline(y=0.05, color='grey', linestyle='--', label='5cm threshold')
        ax.set_yscale('log')
        plt.savefig(f'{self._folder}/position_error_log.png')
        plt.close(fig)
    
    @BaseTaskVisualizer.register
    def plot_velocity(self):
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('Robot Body Velocities')
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(self._data['elapsed_time.s'], self._data['linear_velocities_body.x.m/s'], label='body linear velocity')
        ax.plot(self._data['elapsed_time.s'], self._data['linear_velocities_body.y.m/s'], label='body lateral velocity')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Linear Velocities (m/s)')
        ax.set_title('Linear Velocities')
        # Add a grid
        ax.grid(visible=True)
        ax.legend()
        # Add a 0 line in a grey color
        ax.axhline(y=0, color='grey', linestyle='--')
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(self._data['elapsed_time.s'], self._data['angular_velocities_body.z.rad/s'], label='body angular velocity')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Angular Velocity')
        ax.grid(visible=True)
        ax.legend()
        # Add a 0 line in a grey color
        ax.axhline(y=0, color='grey', linestyle='--')
        fig.tight_layout()

        plt.savefig(f'{self._folder}/linear_velocity.png')
        plt.close(fig)