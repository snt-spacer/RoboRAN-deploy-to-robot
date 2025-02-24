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
                        "enable_logging": True,
                        "device": "cuda:0",
                        "max_steps": 200,
                        "dt": 1 / 5.0,
                        "nn_log_dir": "/home/admin51/Documents/RANS_DeployToRobot/models/skrl/Single/IROS_2025/Single/2025-02-23_15-03-17_ppo-discrete_torch_FloatingPlatform-GoToPosition_seed-42_massRnd-False_comRnd-False_wrenchRnd-False_noisyActRnd-False_actRescalerRnd-False_obsRnd-False_",
                        "nn_checkpoint_path": "None",
                        "terminate_on_completion": True,
                        "logs_save_path": "/RANS_DeployToRobot/ros_experiments_logs",
                    }
                ],
                remappings=[
                    ("state_preprocessor_input", "/vrpn_mocap/RigidBody_005/pose"),
                    ("robot_interface_commands", "spacer_floating_platform/valves/input"),
                ],
            )
        ]
    )
