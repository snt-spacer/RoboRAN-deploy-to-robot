import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription

import launch_ros.actions
import launch_ros.descriptions


def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(
            package='joy', executable='joy_node', name='joy',
            parameters=[
                {'autorepeat_rate': 10.},
                {'dev': "/dev/input/js0"},
                ],
            arguments= [ "--ros-args", "--log-level", "joy:=debug"],
            output='screen'),

        launch_ros.actions.Node(
            package='vrep_ros_teleop', executable='teleop_node', name='teleop',
            parameters=[
                {'~/axis_linear_x': 1},
                {'~/axis_angular': 0},
                {'~/scale_linear_x': 1.0},
                {'~/scale_angular': 1.0},
                {'~/timeout': 1.0}
                ],
            remappings=[
                ('~/twistCommand', '/velocity_smoother/input'),
                ],
            output='screen'),
    ])
