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
                        "robot_interface_name": "Kingfisher",
                        "inference_runner_name": "RLGames",
                        "enable_logging": True,
                        "device": "cpu",
                        "max_steps": 500,
                        "dt": 1 / 12.5,
                        "terminate_on_completion": False,
                        "nn_checkpoint_path": "/RANS_DeployToRobot/models/RLGames/GoToPosition/2025-02-22_17-12-34/nn/kingfisher_direct.pth",
                        "nn_log_dir": "/RANS_DeployToRobot/models/RLGames/GoToPosition/2025-02-22_17-12-34/",
                        "logs_save_path": "/RANS_DeployToRobot/ros_experiments_logs",
                    }
                ],
                remappings=[
                    ("state_preprocessor_input", "/state"),
                    ("observation_formater_input", "/goal"),
                    ("robot_interface_commands", "/action"),
                ],
            )
        ]
    )
