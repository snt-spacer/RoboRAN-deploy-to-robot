from . import generate_plots

import pandas as pd


go_to_positon_data  = pd.read_csv("/RANS_DeployToRobot/ros_experiments_logs/logs/SKRL_VirtualFloatingPlatform_GoToPosition_2025-03-04-08-45-12.csv")
go_to_pose_data = pd.read_csv("/RANS_DeployToRobot/ros_experiments_logs/logs/SKRL_VirtualFloatingPlatform_GoToPose_2025-03-04-08-50-40.csv")
go_through_positions_data = pd.read_csv("/RANS_DeployToRobot/ros_experiments_logs/logs/SKRL_VirtualFloatingPlatform_GoThroughPositions_2025-03-04-09-40-13.csv")
track_velocities_data = pd.read_csv("/RANS_DeployToRobot/ros_experiments_logs/logs/SKRL_VirtualFloatingPlatform_TrackVelocities_2025-03-04-17-43-51.csv")
generate_plots(go_to_positon_data, "GoToPositon_plots", "GoToPosition", "VirtualFloatingPlatform")
generate_plots(go_to_pose_data, "GoToPose_plots", "GoToPose", "VirtualFloatingPlatform")
generate_plots(go_through_positions_data, "GoThroughPositions_plots", "GoThroughPositions", "VirtualFloatingPlatform")
generate_plots(track_velocities_data, "TrackVelocities_plots", "TrackVelocities", "VirtualFloatingPlatform")