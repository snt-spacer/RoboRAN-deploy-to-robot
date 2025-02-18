from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    goal_generator_dir = get_package_share_directory('goal_generator')
    goals_file_path = LaunchConfiguration('goals_file', default=PathJoinSubstitution([
        goal_generator_dir, 'config', 'goals.yaml'
    ]))

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
                        "task_name": "GoToPosition",
                        "goals_file_path": goals_file_path,
                        "state_preprocessor_name": "Optitrack",
                        "device": "cuda:0",
                    }
                ],
            )
        ]
    )
