#!/usr/bin/env python3

import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

from .goal_formaters import GoalFormaterFactory

from rcl_interfaces.msg import ParameterDescriptor
import time


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


        # ROS2 Publishers
        self.goal_pub = self.create_publisher(
            self.goal_formater.ROS_TYPE,
            "observation_formater_input",
            self.goal_formater.ROS_QUEUE_SIZE,
        )

    def run(self):
        while rclpy.ok():
            if self.goal_pub.get_subscription_count() != 0:
                self.task_is_live = True

            if self.task_is_live:
                self.run_task()
                if self.goal_formater.task_completed:
                    break

            self.get_logger().info("Waiting for subscriber to connect")
            time.sleep(1)

    def run_task(self):
        while rclpy.ok():

            if self.goal_formater.send_goal and self.goal_pub.get_subscription_count() != 0:
                self.goal_pub.publish(self.goal_formater.goal)
                self.get_logger().info(self.goal_formater.log_publish())

            if self.goal_formater.task_completed:
                break

            self.get_logger().info("Waiting to publish new goal")
            time.sleep(1)

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
