from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    goal_generator_dir = get_package_share_directory("goal_generator")

    # Launch configuration variables
    goals_file_path = LaunchConfiguration(
        "goals_file", default=PathJoinSubstitution([goal_generator_dir, "config", "lake", "velocity_trajectories", "circle.yaml"])
    )
    is_local = LaunchConfiguration("is_local", default=True)
    task_name = LaunchConfiguration("task_name", default="TrackVelocities")
    # Launch arguments
    goals_file_path_arg = DeclareLaunchArgument(
        "goals_file_path",
        default_value=PathJoinSubstitution([goal_generator_dir, "config", "lake","velocity_trajectories","circle.yaml"]),
        description="The path to the goals file",
    )
    is_local_arg = DeclareLaunchArgument(
        "is_local",
        default_value="True",
        description="Whether the goals are local or not",
        choices=["True", "False"]
    )
    task_name_arg = DeclareLaunchArgument(
        "task_name",
        default_value="TrackVelocities",
        description="The name of the task",
        choices=["GoThroughPositions", "GoToPosition", "GoToPose", "TrackVelocities"]
    )
    args = [goals_file_path_arg, is_local_arg, task_name_arg]

    node = Node(
        package="goal_generator",
        executable="goal_generator_node",
        name="goal_generator_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "task_name": task_name,
                "goals_file_path": goals_file_path,
                "is_local": is_local,
            }
        ],
        remappings=[
            ("task_is_done", "task_available_interface"),
            ("state_preprocessor_input", "/vrpn_mocap/RigidBody_005/pose"),
        ]
    )

    return LaunchDescription(
        [
            *args,
            node,
        ]
    )
