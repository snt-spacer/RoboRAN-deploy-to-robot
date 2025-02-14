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
                        "state_preprocessor_name": "Optitrack",
                        "robot_interface_name": "FloatingPlatform",
                        "inference_runner_name": "SKRL",
                        "enable_logging": False,
                        "device": "cuda:0",
                        "max_steps": 250,
                        "dt": 1 / 5.0,
                        "nn_log_dir": "None",
                        "nn_checkpoint_path": "None",
                        "terminate_on_completion": False,
                        "logs_save_path": "./ros_data_logs",
                    }
                ],
            )
        ]
    )
