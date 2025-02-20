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
                        "inference_runner_name": "Random",
                        "enable_logging": True,
                        "device": "cuda:0",
                        "max_steps": 100,
                        "dt": 1 / 10.0,
                        "nn_log_dir": "None",
                        "nn_checkpoint_path": "None",
                        "terminate_on_completion": True,
                        "logs_save_path": "/RANS_DeployToRobot/ros_experiments_logs",
                    }
                ],
                remappings=[
                    ("state_preprocessor_input", "/vrpn_client_node/FP_exp_RL/PoseStamped"),
                   # ("robot_interface_commands", "/omniFPS/Robots/FloatingPlatform/thrusters/input"),
                   ("robot_interface_commands", "/spacer_floating_platform/valves/input"),
                ],
            )
        ]
    )
