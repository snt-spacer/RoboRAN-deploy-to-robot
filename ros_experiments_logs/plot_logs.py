import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("/home/admin51/Documents/RANS_DeployToRobot/ros_experiments_logs/logs/SKRL_VirtualFloatingPlatform_TrackVelocities_2025-02-24-08-10-03.csv")

# Define the time axis
time = df["elapsed_time.s"]

# Create a figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot linear velocities
axes[0].plot(time, df["linear_velocities_bodyx.m/s"], label="Linear Vel X (m/s)", linestyle="-")
axes[0].plot(time, df["linear_velocities_bodyy.m/s"], label="Linear Vel Y (m/s)", linestyle="--")
axes[0].plot(time, df["linear_velocities_bodyz.m/s"], label="Linear Vel Z (m/s)", linestyle="-.")
axes[0].set_ylabel("Linear Velocity (m/s)")
axes[0].legend()
axes[0].set_title("Linear Velocities Over Time")

# Plot angular velocities
axes[1].plot(time, df["angular_velocities_bodyx.rad/s"], label="Angular Vel X (rad/s)", linestyle="-")
axes[1].plot(time, df["angular_velocities_bodyy.rad/s"], label="Angular Vel Y (rad/s)", linestyle="--")
axes[1].plot(time, df["angular_velocities_bodyz.rad/s"], label="Angular Vel Z (rad/s)", linestyle="-.")
axes[1].set_ylabel("Angular Velocity (rad/s)")
axes[1].legend()
axes[1].set_title("Angular Velocities Over Time")

# Plot target velocities vs. errors
axes[2].plot(time, df["target_linear_vel.m.s"], label="Target Linear Vel (m/s)", linestyle="-", alpha=0.8)
axes[2].plot(time, df["task_data.lin_vel_error.m/s"], label="Linear Vel Error (m/s)", linestyle="--", alpha=0.8)
axes[2].plot(time, df["target_lateral_vel.m/s"], label="Target Lateral Vel (m/s)", linestyle="-.", alpha=0.8)
axes[2].plot(time, df["task_data.lat_vel_error.m/s"], label="Lateral Vel Error (m/s)", linestyle=":", alpha=0.8)
axes[2].set_ylabel("Velocity & Errors (m/s)")
axes[2].legend()
axes[2].set_title("Target Velocities and Errors Over Time")

# Label x-axis
axes[2].set_xlabel("Elapsed Time (s)")

# # plt.tight_layout()
# # plt.show()

# breakpoint()

# Compute position errors
position_error_x = df["position_world.x.m"] - df["target_position.x.m"]
position_error_y = df["position_world.y.m"] - df["target_position.y.m"]

# Angle error
angle_error = df["task_data.ang_vel_error.rad/s"]

# Create the figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot position errors
axes[0].plot(time, position_error_x, label="Position Error X (m)", linestyle="-")
axes[0].plot(time, position_error_y, label="Position Error Y (m)", linestyle="--")
axes[0].set_ylabel("Position Error (m)")
axes[0].legend()
axes[0].set_title("Position Errors Over Time")

# Plot angle errors
axes[1].plot(time, angle_error, label="Angular Velocity Error (rad/s)", linestyle="-", color="red")
axes[1].set_ylabel("Angular Error (rad/s)")
axes[1].legend()
axes[1].set_title("Angular Velocity Error Over Time")

# Label x-axis
axes[1].set_xlabel("Elapsed Time (s)")


# Extract position and target position data
position_x = df["position_world.x.m"]
position_y = df["position_world.y.m"]
target_x = df["target_position.x.m"]
target_y = df["target_position.y.m"]

# Create a figure for the trajectory plot
plt.figure(figsize=(8, 8))

# Plot actual position trajectory
plt.plot(position_x, position_y, label="Actual Position", linestyle="-", marker="o", alpha=0.7)

# Plot target position trajectory
plt.plot(target_x, target_y, label="Target Position", linestyle="--", marker="x", alpha=0.7)

# Quiver plot showing the velocity vectors

vx = df["linear_velocities_bodyx.m/s"]
vy = df["linear_velocities_bodyy.m/s"]
norm = (vx**2 + vy**2)**0.5
u = vx / norm
v = vy / norm

plt.quiver(position_x, position_y, u, v, color="red", alpha=0.5)

# Labels and title
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Position Trajectory vs Target")
plt.legend()
plt.axis("equal")
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()