import matplotlib.pyplot as plt
import numpy as np
import datetime

def plot_episode_data(episode_data, save_path=None):
    """Plots the trajectory data from an episode and saves the figure."""

    if save_path is None:
        raise ValueError("save path cannot be None.")
    
    save_path += "/episode_data.png"
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # Position Trajectory in 2D
    positions = np.array(episode_data["position"])
    axes[0, 0].plot(positions[:, 0], positions[:, 1], marker="o", linestyle="-")
    axes[0, 0].set_title("2D Position Trajectory")
    axes[0, 0].set_xlabel("X Position")
    axes[0, 0].set_ylabel("Y Position")
    axes[0, 0].grid(True)

    # Linear Velocities
    lin_vels = np.array(episode_data["lin_vel"])
    axes[0, 1].plot(lin_vels[:, 0], label="vx")
    axes[0, 1].plot(lin_vels[:, 1], label="vy")
    axes[0, 1].set_title("Linear Velocities")
    axes[0, 1].set_xlabel("Time Steps")
    axes[0, 1].set_ylabel("Velocity (m/s)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Angular Velocity (Yaw Rate)
    ang_vels = np.array(episode_data["ang_vel"])
    axes[1, 0].plot(ang_vels, label="Yaw Rate (rad/s)")
    axes[1, 0].set_title("Angular Velocity")
    axes[1, 0].set_xlabel("Time Steps")
    axes[1, 0].set_ylabel("Yaw Rate (rad/s)")
    axes[1, 0].grid(True)

    # Distance to Goal
    dists = np.array(episode_data["dist_to_goal"])
    axes[1, 1].plot(dists, label="Distance to Goal")
    axes[1, 1].set_title("Distance to Target")
    axes[1, 1].set_xlabel("Time Steps")
    axes[1, 1].set_ylabel("Distance (m)")
    axes[1, 1].grid(True)

    # Heading Error
    heading_errors = np.array(episode_data["heading_error"])
    axes[2, 0].plot(heading_errors, label="Heading Error (rad)")
    axes[2, 0].set_title("Heading Error")
    axes[2, 0].set_xlabel("Time Steps")
    axes[2, 0].set_ylabel("Error (rad)")
    axes[2, 0].grid(True)

    # Actions Taken
    actions = np.array(episode_data["actions"])
    for i in range(actions.shape[1]):
        axes[2, 1].plot(actions[:, i], label=f"Thruster {i}")
    axes[2, 1].set_title("Actions Taken (Thrusters)")
    axes[2, 1].set_xlabel("Time Steps")
    axes[2, 1].set_ylabel("Action (0 or 1)")
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
