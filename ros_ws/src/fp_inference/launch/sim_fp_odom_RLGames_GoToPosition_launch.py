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
                        "task_name": "GoToPosition",
                        "state_preprocessor_name": "Odometry",
                        "robot_interface_name": "VirtualFloatingPlatform",
                        "inference_runner_name": "RLGames",
                        "enable_logging": True,
                        "device": "cuda:0",
                        "max_steps": 200,
                        "dt": 1 / 15.0,
                        "nn_log_dir": "/RANS_DeployToRobot/models/rl_games_FP_gotoposition_2025-02-19_08-47-09",
                        "nn_checkpoint_path": "/RANS_DeployToRobot/models/rl_games_FP_gotoposition_2025-02-19_08-47-09/nn/last_FloatingPlatform-GoToPosition_ep_500_rew__197.19853_.pth",
                        "terminate_on_completion": True,
                        "logs_save_path": "/RANS_DeployToRobot/ros_experiments_logs",
                    }
                ],
                remappings=[
                    ("state_preprocessor_input", "/omniFPS/Robots/FloatingPlatform/odom"),
                    ("robot_interface_commands", "/omniFPS/Robots/FloatingPlatform/thrusters/input"),
                ],
            )
        ]
    )
