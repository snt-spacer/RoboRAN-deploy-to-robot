#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np


class OptiTrackSimulator(Node):
    def __init__(self):
        super().__init__("optitrack_simulator")
        self.pose_pub = self.create_publisher(PoseStamped, "/vrpn_client_node/FP_exp_RL/PoseStamped", 10)
        self.timer = self.create_timer(0.1, self.publish_pose)  # 10 Hz
        self.t = 0.0

    def publish_pose(self):
        # Generate a circular trajectory for testing
        x = 2.0 * np.cos(self.t)
        y = 2.0 * np.sin(self.t)
        z = 1.0  # Fixed height

        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.w = 1.0
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0

        self.pose_pub.publish(pose_msg)

        self.t += 0.1


def main(args=None):
    rclpy.init(args=args)
    node = OptiTrackSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
