from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    goal_generator_dir = get_package_share_directory("goal_generator")
    goals_file_path = LaunchConfiguration(
        "goals_file", default=PathJoinSubstitution([goal_generator_dir, "config", "GoThroughPositions_fp_sim.yaml"])
    )

    return LaunchDescription(
        [
            Node(
                package="goal_generator",
                executable="goal_generator_node",
                name="goal_generator_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "task_name": "GoToPose",
                        "goals_file_path": goals_file_path,
                    }
                ],
                remappings=[
                    ("task_is_done", "task_available_interface"),
                ],
            )
        ]
    )
