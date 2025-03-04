from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np

from . import BaseTaskVisualizer, Registerable

class GoThroughPositionsVisualizer(BaseTaskVisualizer, Registerable):
    def __init__(self, data: pd.DataFrame, folder: str) -> None:
        super().__init__(data, folder)

    @BaseTaskVisualizer.register
    def plot_trajectory(self) -> None:
        fig = plt.figure(figsize=(8,8))
        # Compute the limits of the plot
        x_min, x_max = self._data['position_world.x.m'].min(), self._data['position_world.x.m'].max()
        y_min, y_max = self._data['position_world.y.m'].min(), self._data['position_world.y.m'].max()
        # Add 30% padding
        dx = x_max - x_min
        dy = y_max - y_min
        x_min -= np.floor(0.15 * dx)
        x_max += np.floor(0.15 * dx)
        y_min -= np.ceil(0.15 * dy)
        y_max += np.ceil(0.15 * dy)
        # Plot the trajectory of the robot
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self._data['position_world.x.m'], self._data['position_world.y.m'], label='Robot Trajectory', color='royalblue', zorder=3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectory')
        # Add the goal position
        target_pos_x = np.array(self._data['target_position.x.m'])
        target_pos_y = np.array(self._data['target_position.y.m'])
        target_pos = np.stack([target_pos_x, target_pos_y], axis=1)
        # Get unique target positions
        unique_target_pos = np.unique(target_pos, axis=0)
        plt.scatter(unique_target_pos[:,0], unique_target_pos[:,1], color='r', facecolor='none', label='Goal Position', zorder=4)
        ax.set_xticks(np.arange(x_min, x_max + 1, 1))
        ax.set_yticks(np.arange(y_min, y_max + 1, 1))
        ax.grid(which='major', color='black', linewidth=1)
        ax.set_xticks(np.arange(x_min, x_max + 0.25, 0.25), minor=True)
        ax.set_yticks(np.arange(y_min, y_max + 0.25, 0.25), minor=True)
        ax.grid(which='minor', color='gray', linewidth=0.5, linestyle='--')
        ax.axis('equal')
        ax.legend()
        plt.savefig(f'{self._folder}/trajectory.png')
        plt.close(fig)

    @BaseTaskVisualizer.register
    def plot_trajectory_with_heading(self) -> None:
        fig = plt.figure(figsize=(8,8))
        # Compute the limits of the plot
        x_min, x_max = self._data['position_world.x.m'].min(), self._data['position_world.x.m'].max()
        y_min, y_max = self._data['position_world.y.m'].min(), self._data['position_world.y.m'].max()
        # Add 30% padding
        dx = x_max - x_min
        dy = y_max - y_min
        x_min -= np.floor(0.15 * dx)
        x_max += np.floor(0.15 * dx)
        y_min -= np.ceil(0.15 * dy)
        y_max += np.ceil(0.15 * dy)
        # Plot the trajectory of the robot
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self._data['position_world.x.m'], self._data['position_world.y.m'], label='Robot Trajectory', color='royalblue', zorder=3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectory with Heading')
        ax.set_xticks(np.arange(x_min, x_max + 1, 1))
        ax.set_yticks(np.arange(y_min, y_max + 1, 1))
        ax.grid(which='major', color='black', linewidth=1)
        ax.set_xticks(np.arange(x_min, x_max + 0.25, 0.25), minor=True)
        ax.set_yticks(np.arange(y_min, y_max + 0.25, 0.25), minor=True)
        ax.grid(which='minor', color='gray', linewidth=0.5, linestyle='--')
        # Add the goal position
        target_pos_x = np.array(self._data['target_position.x.m'])
        target_pos_y = np.array(self._data['target_position.y.m'])
        target_pos = np.stack([target_pos_x, target_pos_y], axis=1)
        # Get unique target positions
        unique_target_pos = np.unique(target_pos, axis=0)
        plt.scatter(unique_target_pos[:,0], unique_target_pos[:,1], color='r', facecolor='none', label='Goal Position', zorder=4)
        # Add the legend before adding the robot heading
        ax.legend()
        # Add the robot heading 20 points only
        idx = np.arange(0, len(self._data), len(self._data) // 20)
        ax.quiver(self._data['position_world.x.m'].iloc[idx], self._data['position_world.y.m'].iloc[idx], 
                  np.cos(self._data['heading_world.rad'].iloc[idx]), np.sin(self._data['heading_world.rad'].iloc[idx]),
                  color='royalblue', label='Robot Heading', zorder=3)
        ax.axis('equal')
        plt.savefig(f'{self._folder}/trajectory_with_heading.png')
        plt.close(fig)

    @BaseTaskVisualizer.register
    def make_trajectory_video(self):
        # Compute the limits of the plot
        x_min, x_max = self._data['position_world.x.m'].min(), self._data['position_world.x.m'].max()
        y_min, y_max = self._data['position_world.y.m'].min(), self._data['position_world.y.m'].max()
        # Add 30% padding
        dx = x_max - x_min
        dy = y_max - y_min
        x_min -= np.floor(0.15 * dx)
        x_max += np.floor(0.15 * dx)
        y_min -= np.ceil(0.15 * dy)
        y_max += np.ceil(0.15 * dy)
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
        x_pos = np.array(self._data['position_world.x.m'])
        y_pos = np.array(self._data['position_world.y.m'])
        target_pos_x = np.array(self._data['target_position.x.m'])
        target_pos_y = np.array(self._data['target_position.y.m'])
        target_pos = np.stack([target_pos_x, target_pos_y], axis=1)
        # Get unique target positions
        unique_target_pos = np.unique(target_pos, axis=0)
        def update_trajectory(i):
            ax.clear()
            ax.plot(x_pos[:i], y_pos[:i], label='Robot Trajectory', color='b', zorder=3)
            ax.scatter(unique_target_pos[:,0], unique_target_pos[:,1], color='r', facecolor='none', label='Goal Position', zorder=4)
            ax.quiver(self._data['position_world.x.m'].iloc[i], self._data['position_world.y.m'].iloc[i], 
                      np.cos(self._data['heading_world.rad'].iloc[i]), np.sin(self._data['heading_world.rad'].iloc[i]),
                      color='b', label='Robot Heading',zorder=3)
            ax.set_xticks(np.arange(x_min, x_max + 1, 1))
            ax.set_yticks(np.arange(y_min, y_max + 1, 1))
            ax.grid(which='major', color='black', linewidth=1)
            ax.set_xticks(np.arange(x_min, x_max + 0.25, 0.25), minor=True)
            ax.set_yticks(np.arange(y_min, y_max + 0.25, 0.25), minor=True)
            ax.grid(which='minor', color='gray', linewidth=0.5, linestyle='--')
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

    @BaseTaskVisualizer.register
    def plot_goal_distances(self):
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self._data['elapsed_time.s'], self._data['distance_error.m'], label='Goal 0 Distance')
        ax.plot(self._data['elapsed_time.s'], self._data['task_data.goal_1_dist.m'], label='Goal 1 Distance')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (m)')
        ax.set_title('Position Error')
        ax.grid(visible=True)
        ax.grid(linestyle = '--', linewidth = 0.5)

        ax.legend()
        plt.savefig(f'{self._folder}/Goal_distance.png')
        plt.close(fig)
