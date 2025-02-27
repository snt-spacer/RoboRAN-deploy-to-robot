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
                        "inference_runner_name": "SKRL",
                        "enable_logging": True,
                        "device": "cuda:0",
                        "max_steps": 500,
                        "dt": 1 / 5.0,
                        "nn_log_dir": "/RANS_DeployToRobot/models/skrl/Single/IROS_2025/Single/2025-02-23_15-03-17_ppo-discrete_torch_FloatingPlatform-GoToPosition_seed-42_massRnd-False_comRnd-False_wrenchRnd-False_noisyActRnd-False_actRescalerRnd-False_obsRnd-False_",
                        "nn_checkpoint_path": "None",
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
