#!/usr/bin/env python3

import threading
import rclpy
import rclpy.logging
from rclpy.node import Node
import rclpy.time
from std_msgs.msg import Bool

from .state_preprocessors import StatePreprocessorFactory
from .observation_formaters import ObservationFormaterFactory
from .robot_interfaces import RobotInterfaceFactory
from .inference_runners import InferenceRunnerFactory
from .utils import Logger

from rcl_interfaces.msg import ParameterDescriptor


class RLTaskNode(Node):
    """ROS2 node that runs a reinforcement learning task."""

    def __init__(self) -> None:
        """Initialize the node by setting the parameters and building the task."""
        super().__init__("rl_task_node")

        # Parameters
        task_name_desc = ParameterDescriptor(
            description='The name of the task to be executed.\
                Currently the following tasks are supported: {}".'.format(
                ", ".join(ObservationFormaterFactory.registry.keys())
            )
        )
        self.declare_parameter("task_name", "GoToPosition", task_name_desc)
        self._task_name = self.get_parameter("task_name").get_parameter_value().string_value
        state_preprocessor_name_desc = ParameterDescriptor(
            description='The name of the state preprocessor to be used.\
                Currently the following state preprocessors are supported: {}".'.format(
                ", ".join(StatePreprocessorFactory.registry.keys())
            )
        )
        self.declare_parameter("state_preprocessor_name", "Optitrack", state_preprocessor_name_desc)
        self._state_preprocessor_name = self.get_parameter("state_preprocessor_name").get_parameter_value().string_value
        robot_interface_name_desc = ParameterDescriptor(
            description='The name of the robot interface to be used.\
                Currently the following robot interfaces are supported: {}".'.format(
                ", ".join(RobotInterfaceFactory.registry.keys())
            )
        )
        self.declare_parameter("robot_interface_name", "FloatingPlatform", robot_interface_name_desc)
        self._robot_interface_name = self.get_parameter("robot_interface_name").get_parameter_value().string_value
        inference_runner_name_desc = ParameterDescriptor(
            description='The name of the inference runner to be used.\
                Currently the following inference runners are supported: {}".'.format(
                ", ".join(InferenceRunnerFactory.registry.keys())
            )
        )
        self.declare_parameter("inference_runner_name", "SKRLInferenceRunner", inference_runner_name_desc)
        self._inference_runner_name = self.get_parameter("inference_runner_name").get_parameter_value().string_value
        enable_logging_desc = ParameterDescriptor(description="Enable logging of the task execution.")
        self.declare_parameter("enable_logging", False, enable_logging_desc)
        self._enable_logging = self.get_parameter("enable_logging").get_parameter_value().bool_value
        device_desc = ParameterDescriptor(
            description='The device to be used for the task.\
                If set to "auto", the device will be selected automatically.'
        )
        self.declare_parameter("device", "auto", device_desc)
        self._device = self.get_parameter("device").get_parameter_value().string_value
        max_steps_desc = ParameterDescriptor(description="The number of steps to be executed in each episode.")
        self.declare_parameter("max_steps", 200, max_steps_desc)
        self._max_steps = self.get_parameter("max_steps").get_parameter_value().integer_value
        dt_desc = ParameterDescriptor(description="The time step with which the task will be executed.")
        self.declare_parameter("dt", 1 / 15.0, dt_desc)
        self._dt = self.get_parameter("dt").get_parameter_value().double_value
        nn_log_dir_desc = ParameterDescriptor(description="The directory where the neural network model is stored.")
        self.declare_parameter("nn_log_dir", "None", nn_log_dir_desc)
        self._nn_log_dir = self.get_parameter("nn_log_dir").get_parameter_value().string_value
        if self._nn_log_dir == "None":
            self._nn_log_dir = None
        nn_checkpoint_path_desc = ParameterDescriptor(description="The path to the neural network model checkpoint.")
        self.declare_parameter("nn_checkpoint_path", "None", nn_checkpoint_path_desc)
        self._nn_checkpoint_path = self.get_parameter("nn_checkpoint_path").get_parameter_value().string_value
        if self._nn_checkpoint_path == "None":
            self._nn_checkpoint_path = None
        terminate_on_completion_desc = ParameterDescriptor(description="Terminate the node when the goal is reached.")
        self.declare_parameter("terminate_on_completion", False, terminate_on_completion_desc)
        self._terminate_on_completion = self.get_parameter("terminate_on_completion").get_parameter_value().bool_value
        logs_save_path_desc = ParameterDescriptor(description="The path where the logs will be saved.")
        self.declare_parameter("logs_save_path", "None", logs_save_path_desc)
        self._logs_save_path = self.get_parameter("logs_save_path").get_parameter_value().string_value
        if self._logs_save_path == "None":
            self._logs_save_path

        # Build the task
        self.build()

    def build(self) -> None:
        """Build the task.

        This is done by creating the state preprocessor, observation formater, inference runner, and robot
        interface.
        """
        self.get_logger().info("Building task...")

        self._state_preprocessor = StatePreprocessorFactory.create(self._state_preprocessor_name, device=self._device)
        self._robot_interface = RobotInterfaceFactory.create(self._robot_interface_name, device=self._device)
        self._observation_formater = ObservationFormaterFactory.create(
            self._task_name,
            self._state_preprocessor,
            num_actions=self._robot_interface.num_actions,
            max_steps=self._max_steps,
            device=self._device,
        )
        self._inference_runner = InferenceRunnerFactory.create(
            self._inference_runner_name,
            logdir=self._nn_log_dir,
            observation_space=self._observation_formater.observation_space,
            action_space=self._robot_interface.action_space,
            checkpoint_path=self._nn_checkpoint_path,
            device=self._device,
        )

        self.get_logger().info("Initializing logger...")
        objects = [self._state_preprocessor, self._observation_formater, self._robot_interface]
        self._data_logger = Logger(objects, self._enable_logging, self._logs_save_path)

        self.get_logger().info("Opening ROS2 interfaces...")
        # ROS2 Subscriptions
        self.create_subscription(
            self._state_preprocessor.ROS_TYPE,
            "state_preprocessor_input",
            self._state_preprocessor.ROS_CALLBACK,
            self._state_preprocessor.QOS_PROFILE,
        )
        self.create_subscription(
            self._observation_formater.ROS_TYPE,
            "observation_formater_input",
            self._observation_formater.ROS_CALLBACK,
            self._observation_formater.QOS_PROFILE,
        )
        self.create_subscription(Bool, "reset_task", self.reset_task, 1)

        # ROS2 Publishers
        self.action_pub = self.create_publisher(
            self._robot_interface.ROS_ACTION_TYPE,
            "robot_interface_commands",
            self._robot_interface.QOS_PROFILE,
        )
        self.task_available_pub = self.create_publisher(
            Bool,
            "task_available_interface",
            1,
        )
        self._done_msg = Bool()
        self._done_msg.data = True
        self.kill_signal = False

        self.get_logger().info("Task built!")

    def reset_task(self, *args, **kwargs) -> None:
        """Reset the task.

        This is done by resetting the state preprocessor, observation formater, inference runner, and robot
        interface.
        """
        self.get_logger().info("Reset received. Terminating task.")

        # Reset all the components
        self._state_preprocessor.reset()
        self._observation_formater.reset()
        self._inference_runner.reset()
        self._robot_interface.reset()

    def advertize_task_status(self) -> None:
        """Advertize the task status by publishing the task status every second."""
        advertize_rate = self.create_rate(1.0)
        while rclpy.ok() and (not self.kill_signal):
            self._done_msg.data = not self._observation_formater.task_is_live
            self.task_available_pub.publish(self._done_msg)
            advertize_rate.sleep()

    def run_task(self) -> None:
        """Run the task by executing the main task loop."""
        self.get_logger().info("Running task...")
        # Create a rate object
        rate = self.create_rate(1.0 / self._dt)
        self._observation_formater._task_is_live = True
        # Main task loop
        while rclpy.ok():
            # Get the current observation
            self._observation_formater.format_observation(self._robot_interface.last_actions)
            # Get the action from the inference runner
            action = self._inference_runner.act(self._observation_formater.observation)
            # Publish the action
            self.action_pub.publish(self._robot_interface.cast_actions(action))
            # Log the data
            self._data_logger.update()
            # Check if the task is completed
            if (self._observation_formater.task_completed) or (not self._observation_formater.task_is_live):
                break
            # Sleep
            rate.sleep()

    def run(self) -> None:
        """Run the task node by waiting for a goal to be received and then running the task."""
        # Check every second if a goal has been received
        wait_rate = self.create_rate(1.0)
        while rclpy.ok():
            # Wait for a goal to be received
            if self._observation_formater.task_is_live:
                self.get_logger().info("Goal received!")
                self.prime_state_preprocessor()
                self.run_task()
                self.get_logger().info("Task completed!")
                # Reset the robot interface
                self._robot_interface.reset()
                self.action_pub.publish(self._robot_interface.pre_kill_action)
                # Send immobilization command
                self.action_pub.publish(self._robot_interface.kill_action)
                # Task is completed, save the logs.
                self._data_logger.save(self._robot_interface_name, self._inference_runner_name, self._task_name)
                # Returns if the task is completed
                if self._terminate_on_completion:
                    self.get_logger().info("In run once mode. Terminating node.")
                    break
                else:
                    self.get_logger().info("Resetting task.")
                    self._observation_formater.reset()
            self.get_logger().info("Waiting for goal...")
            wait_rate.sleep()

    def prime_state_preprocessor(self) -> None:
        """Wait for the state preprocesseor to be ready."""
        self.get_logger().info("Warming up the state preprocessor...")
        obs_rate = self.create_rate(1.0 / self._dt)
        while rclpy.ok():
            if self._state_preprocessor.is_primed:
                break
            obs_rate.sleep()
        self.get_logger().info("State preprocessor is ready!")

    def shutdown(self) -> None:
        """Shut down the node."""
        self._kill_signal = True

    def clean_termination(self) -> None:
        """Terminate the node."""
        self._kill_signal = True
        self.destroy_node()
        rclpy.shutdown()

    def on_interupt(self) -> None:
        """Handle the interrupt signal."""
        self.action_pub.publish(self._robot_interface.kill_action)
        self.get_logger().info("Received interrupt signal. Trying to save and terminating node.")
        self._data_logger.save(self._robot_interface_name, self._inference_runner_name, self._task_name)
        self.clean_termination()


def main(args=None):
    """Start the ROS2 node."""
    import signal
    import sys

    rclpy.init(args=args)

    task_node = RLTaskNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(task_node,), daemon=True)
    spin_thread.start()
    task_status_thread = threading.Thread(target=task_node.advertize_task_status, daemon=True)
    task_status_thread.start()

    def signal_handler(sig, frame):
        task_node.kill_signal = True
        task_status_thread.join()
        task_node.on_interupt()
        spin_thread.join()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)

    task_node.run()

    task_node.kill_signal = True
    task_status_thread.join()
    task_node.clean_termination()
    spin_thread.join()

if __name__ == "__main__":
    main()
