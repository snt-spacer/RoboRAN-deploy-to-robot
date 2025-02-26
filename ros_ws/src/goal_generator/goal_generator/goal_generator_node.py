#!/usr/bin/env python3

import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
import copy

from .goal_formaters import GoalFormaterFactory

from rcl_interfaces.msg import ParameterDescriptor

class GoalPublisherNode(Node):
    def __init__(self):
        super().__init__("goal_publisher_node")

        # Task name
        task_name_desc = ParameterDescriptor(
            description='The name of the task to be executed. Currently the following tasks are supported: {}".'.format(
                ", ".join(GoalFormaterFactory.registry.keys())
            )
        )
        self.declare_parameter("task_name", "GoToPosition", task_name_desc)
        self._task_name = self.get_parameter("task_name").get_parameter_value().string_value

        # Goals file path
        goals_file_path_desc = ParameterDescriptor(description="The path to the file containing the goals.")
        self.declare_parameter("goals_file_path", "", goals_file_path_desc)
        self._goals_file_path = self.get_parameter("goals_file_path").get_parameter_value().string_value

        # Device
        device_desc = ParameterDescriptor(
            description='The device to be used for the task. If set to "auto", the device will be selected automatically.'
        )
        self.declare_parameter("device", "auto", device_desc)
        self._device = self.get_parameter("device").get_parameter_value().string_value

        self.task_is_live = False

        self.build()

    def build(self):
        self.goal_formater = GoalFormaterFactory.create(self._task_name, self._goals_file_path)

        self._a = self._prev_a = self._b = self._prev_b = 0
        self._a_was_pressed = self._a_was_released = self._b_was_pressed = self._b_was_released = False
        self._waiting_for_a = self._waiting_for_b = False

        # ROS2 Subscriptions
        self.create_subscription(
            Bool,
            "task_is_done",
            self.task_is_done_callback,
            1
        )
        self.create_subscription(
            Joy,
            "joystick",
            self.joy_callback,
            1
        )

        # ROS2 Publishers
        self.goal_pub = self.create_publisher(
            self.goal_formater.ROS_TYPE,
            "observation_formater_input",
            self.goal_formater.QOS_PROFILE,
        )

    def wait_for_A_button(self):
        self._waiting_for_a = True
        while rclpy.ok():
            if self._a_was_pressed:
                self._waiting_for_a = False
                self._a = self._prev_a = 0
                break
    
    def wait_for_A_or_B_button(self) -> bool:
        self._waiting_for_a = True
        self._waiting_for_b = True
        while rclpy.ok():
            if self._a_was_pressed:
                self._waiting_for_a = False
                self._a = self._prev_a = 0
                return True
            if self._b_was_pressed:
                self._waiting_for_b = False
                self._b = self._prev_b = 0
                return False

    def task_is_done_callback(self, msg: Bool) -> None:
        self._task_is_done = msg.data

    def joy_callback(self, msg: Joy) -> None:
        self._buttons = msg.buttons
        # A button
        self._prev_a = copy.copy(self._a)
        self._a = self._buttons[0]
        if self._waiting_for_a:
            self._a_was_pressed = self._prev_a == 0 and self._a == 1
            self._a_was_released = self._prev_a == 1 and self._a == 0
        # B button
        self._prev_b = copy.copy(self._b)
        self._b = self._buttons[1]
        if self._waiting_for_b:
            self._b_was_pressed = self._prev_b == 0 and self._b == 1
            self._b_was_released = self._prev_b == 1 and self._b == 0

    def run(self):
        self.get_logger().info("Press 'A' to start!")
        self.wait_for_A_button()        
        self.get_logger().info("Starting the goal publisher node...") 
        while rclpy.ok():
            self.run_goal_loop()
            if self.goal_formater.is_done:
                self.get_logger().info("Exhausted all goals. Press 'B' to exit, press 'A' to restart.")
                repeat = self.wait_for_A_or_B_button()
                if repeat:
                    self.goal_formater.reset()
                else:
                    break
            else:
                self.get_logger().info("Press the 'A' key to send the next goal!")
                self.wait_for_A_button()


    def run_goal_loop(self) -> None:
        rate = self.create_rate(0.5)
        self.goal_pub.publish(self.goal_formater.goal)
        self.get_logger().info(self.goal_formater.log_publish())
        while rclpy.ok():
            if self.goal_pub.get_subscription_count() != 0:
                self.get_logger().info("Waiting for subscriber...")

            if self._task_is_done:
                self.get_logger().info("Task completed!")
                break
            rate.sleep()

    def clean_termination(self):
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    goal_publisher_node = GoalPublisherNode()

    thread = threading.Thread(target=rclpy.spin, args=(goal_publisher_node,), daemon=True)
    thread.start()

    goal_publisher_node.run()
    goal_publisher_node.clean_termination()
    thread.join()


if __name__ == "__main__":
    main()
