from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from rl_inference import InferenceRunnerFactory
from rl_inference import ObservationFormaterFactory
from rl_inference import RobotInterfaceFactory
from rl_inference import StatePreprocessorFactory


def generate_launch_description():
    # Launch configuration variables
    task_name = LaunchConfiguration("task_name", default="GoThroughPositions")
    state_preprocessor_name = LaunchConfiguration("state_preprocessor_name", default="Optitrack")
    robot_interface_name = LaunchConfiguration("robot_interface_name", default="FloatingPlatform")
    inference_runner_name = LaunchConfiguration("inference_runner_name", default="SKRL")
    enable_logging = LaunchConfiguration("enable_logging", default=True)
    enable_plotting = LaunchConfiguration("enable_plotting", default=True)
    device = LaunchConfiguration("device", default="cpu")
    max_run_time = LaunchConfiguration("max_run_time", default=30)
    dt = LaunchConfiguration("dt", default=1 / 5.0)
    terminate_on_completion = LaunchConfiguration("terminate_on_completion", default=False)
    nn_checkpoint_path = LaunchConfiguration("nn_checkpoint_path", default="None")
    nn_log_dir = LaunchConfiguration("nn_log_dir", default="None")
    logs_save_path = LaunchConfiguration("logs_save_path", default="/RANS_DeployToRobot/ros_experiments_logs")
    # Launch arguments
    task_name_arg = DeclareLaunchArgument(
        "task_name",
        default_value="GoThroughPositions",
        description="The name of the task.",
        choices=ObservationFormaterFactory._registry.keys(),
    )
    state_preprocessor_name_arg = DeclareLaunchArgument(
        "state_preprocessor_name",
        default_value="Optitrack",
        description="The name of the state preprocessor.",
        choices=StatePreprocessorFactory._registry.keys(),
    )
    robot_interface_name_arg = DeclareLaunchArgument(
        "robot_interface_name",
        default_value="FloatingPlatform",
        description="The name of the robot interface.",
        choices=RobotInterfaceFactory._registry.keys(),
    )
    inference_runner_name_arg = DeclareLaunchArgument(
        "inference_runner_name",
        default_value="SKRL",
        description="The name of the inference runner.",
        choices=InferenceRunnerFactory._registry.keys(),
    )
    enable_logging_arg = DeclareLaunchArgument(
        "enable_logging",
        default_value="True",
        description="Whether to enable logging. If logging is disabled no logs will be generated.",
        choices=["True", "False"],
    )
    enable_plotting_arg = DeclareLaunchArgument(
        "enable_plotting",
        default_value="True",
        description="Whether to enable plotting. If logging is disabled no plots will be generated.",
        choices=["True", "False"],
    )
    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cpu",
        description="The device to run the inference on.",
        choices=["cpu", "cuda", "auto"],
    )
    max_run_time_arg = DeclareLaunchArgument(
        "max_run_time",
        default_value="30",
        description="The maximum run time of the task in seconds.",
    )
    dt_arg = DeclareLaunchArgument(
        "dt",
        default_value="0.2",
        description="The time step of the task. In seconds.",
    )
    terminate_on_completion_arg = DeclareLaunchArgument(
        "terminate_on_completion",
        default_value="False",
        description="Whether to terminate the task on completion. If true the task will terminate after the first run. If false the task will run indefinitely.",
        choices=["True", "False"],
    )
    nn_checkpoint_path_arg = DeclareLaunchArgument(
        "nn_checkpoint_path",
        default_value="None",
        description="The path to the neural network checkpoint.",
    )
    nn_log_dir_arg = DeclareLaunchArgument(
        "nn_log_dir",
        default_value="None",
        description="The directory to save the neural network logs.",
    )
    logs_save_path_arg = DeclareLaunchArgument(
        "logs_save_path",
        default_value="/RANS_DeployToRobot/ros_experiments_logs",
        description="The directory to save the logs.",
    )
    args = [
        task_name_arg,
        state_preprocessor_name_arg,
        robot_interface_name_arg,
        inference_runner_name_arg,
        enable_logging_arg,
        enable_plotting_arg,
        device_arg,
        max_run_time_arg,
        dt_arg,
        terminate_on_completion_arg,
        nn_checkpoint_path_arg,
        nn_log_dir_arg,
        logs_save_path_arg,
    ]

    # Node    
    node = Node(
        package="rl_inference",
        executable="rl_task_node_v2",
        name="rl_task_node_v2",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "task_name": task_name,
                "state_preprocessor_name": state_preprocessor_name,
                "robot_interface_name": robot_interface_name,
                "inference_runner_name": inference_runner_name,
                "enable_logging": enable_logging,
                "enable_plotting": enable_plotting,
                "device": device,
                "max_run_time": max_run_time,
                "dt": dt,
                "terminate_on_completion": terminate_on_completion,
                "nn_checkpoint_path": nn_checkpoint_path,
                "nn_log_dir": nn_log_dir,
                "logs_save_path": logs_save_path,
            }
        ],
        remappings=[
            ("state_preprocessor_input", "/vrpn_mocap/RigidBody_005/pose"),
            ("robot_interface_commands", "/fp_cmd_mux/autonomous_control_input"),
        ],
    )
    return LaunchDescription(
        [
            node,
            *args,
        ]
    )
