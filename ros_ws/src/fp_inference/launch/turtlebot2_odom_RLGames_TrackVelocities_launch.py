from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="fp_inference",
                executable="rl_task_node_v2",
                name="rl_task_node_v2",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "task_name": "TrackVelocities",
                        "state_preprocessor_name": "Odometry",
                        "robot_interface_name": "Turtlebot",
                        "inference_runner_name": "DebugVel",
                        "enable_logging": True,
                        "device": "cpu",
                        "max_steps": 3000,
                        "dt": 1 / 10,
                        "nn_checkpoint_path": "/RANS_DeployToRobot/models/2025-02-26_18-08-07_RLGames_TurtleBot2_TrackVelocities/nn/Turtlebot2-TrackVelocities.pth",
                        "nn_log_dir": "/RANS_DeployToRobot/models/2025-02-26_18-08-07_RLGames_TurtleBot2_TrackVelocities",
                        "terminate_on_completion": True,
                        "logs_save_path": "/RANS_DeployToRobot/ros_experiments_logs",
                    }
                ],
                remappings=[
                    ("state_preprocessor_input", "/odom"),
                    ("observation_formater_input", "/goal"),
                    ("robot_interface_commands", "/velocity_smoother/input"),
                ],
            )
        ]
    )
