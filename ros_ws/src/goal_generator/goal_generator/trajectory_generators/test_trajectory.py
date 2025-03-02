from goal_generator import TrajectoryCfgFactory, TrajectoryFactory

import numpy as np
from matplotlib import pyplot as plt
from colorsys import hsv_to_rgb

# Create accelerating sinewave trajectory
name = "AcceleratingSinewave"
cfg = {"min_frequency": 1.0, "max_frequency": 3.0, "num_periods": 5, "max_amplitude": 2.0, "min_amplitude": 1.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Accelerating Sinewave " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("accelerating_sinewave.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Accelerating Sinewave " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("accelerating_sinewave_angle.png")
plt.close("all")

# Create bernouilli lemniscate trajectory
name = "BernouilliLemniscate"
cfg = {"a": 1.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Bernouilli Lemniscate " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("bernouilli_lemniscate.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Bernouilli Lemniscate " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("bernouilli_lemniscate_angle.png")
plt.close("all")

# Create besace trajectory
name = "Besace"
cfg = {"a": 2.0, 'b': 1.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Besace " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("besace.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Besace " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("besace_angle.png")
plt.close("all")

# Create circle trajectory
name = "Circle"
cfg = {"radius": 1.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Circle")
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("circle.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Circle")
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("circle_angle.png")
plt.close("all")

# Create gerono lemniscate trajectory
name = "GeronoLemniscate"
cfg = {"a": 1.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Gerono Lemniscate " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("gerono_lemniscate.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Gerono Lemniscate " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("gerono_lemniscate_angle.png")
plt.close("all")

# hippopede trajectory
name = "Hippopede"
cfg = {"a": 1.0, "b": 1.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Hippopede " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("hippopede.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Hippopede " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("hippopede_angle.png")
plt.close("all")

# Create Hypotrochoid trajectory
name = "Hypotrochoid"
cfg = {"R": 7, "r": 4, "d": 1.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Hypotrochoid " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("hypotrochoid.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Hypotrochoid " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("hypotrochoid_angle.png")
plt.close("all")

# Create Infinite Square trajectory
name = "InfiniteSquare"
cfg = {"x_dim": 1.0, "y_dim": 1.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Infinite Square " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("infinite_square.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Infinite Square " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("infinite_square_angle.png")
plt.close("all")

# Create Lissajous trajectory
name = "Lissajous"
cfg = {"A": 1.0, "B": 1.0, "a": 5.0, "b": 4.0, "omega_x": np.pi / 2.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Lissajous " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("lissajous.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Lissajous " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("lissajous_angle.png")
plt.close("all")

# Create NGon trajectory
name = "NGon"
cfg = {"num_sides": 5, "size": 1.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("NGon " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("ngon.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("NGon " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("ngon_angle.png")
plt.close("all")

# Create Sinewave trajectory
name = "Sinewave"
cfg = {"amplitude": 1.0, "frequency": 1.0, "num_periods": 5}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Sinewave " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("sinewave.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Sinewave " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("sinewave_angle.png")
plt.close("all")

# Create Spiral trajectory
name = "Spiral"
cfg = {"min_radius": 0.25, "max_radius": 2.0, "num_turns": 5.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Spiral " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("spiral.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Spiral " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("spiral_angle.png")
plt.close("all")

# Create Square trajectory
name = "Square"
cfg = {"x_dim": 1.0, "y_dim": 1.0}
string_cfg = ", ".join([f"{key}: {value}" for key, value in cfg.items()])
cfg = TrajectoryCfgFactory.create(name, **cfg)
traj = TrajectoryFactory.create(name, cfg)

# Generate trajectory
xy, angle = traj.trajectory
t = np.linspace(0, 1, num=xy.shape[0])
rgb = np.array([hsv_to_rgb(i, 1.0, 1.0) for i in t])
rgba = np.concatenate((rgb, np.ones((xy.shape[0], 1))), axis=1)
# Plot trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Square " + string_cfg)
plt.scatter(xy[:, 0], xy[:, 1], color=rgba)
plt.axis("equal")
fig.savefig("square.png")
# Plot angular trajectory
fig = plt.figure(figsize=(8, 8))
plt.title("Square " + string_cfg)
plt.quiver(xy[:, 0], xy[:, 1], np.cos(angle), np.sin(angle), color=rgba)
plt.axis("equal")
fig.savefig("square_angle.png")
plt.close("all")
