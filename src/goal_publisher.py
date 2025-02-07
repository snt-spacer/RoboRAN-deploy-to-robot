#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point


class GoalPublisherNode(Node):
    def __init__(self):
        super().__init__("goal_publisher_node")

        # Publisher for the goal position
        self.goal_pub = self.create_publisher(Point, "/spacer_floating_platform/goal", 10)

        # Timer to periodically publish the goal (every 1 second for now)
        self.timer = self.create_timer(1.0, self.publish_goal)

        # Default goal position
        self.goal = Point()
        self.goal.x = 2.5  # Example goal x
        self.goal.y = 1.5  # Example goal y
        self.goal.z = 0.0  # Example goal z (not needed for 2D tasks)

        self.get_logger().info("Goal Publisher Node Initialized.")

    def publish_goal(self):
        # Publish the goal
        self.goal_pub.publish(self.goal)
        self.get_logger().info(f"Published goal: x={self.goal.x}, y={self.goal.y}, z={self.goal.z}")


def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
