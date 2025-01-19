#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Float32MultiArray
import numpy as np


class StateCreationNode(Node):
    def __init__(self):
        super().__init__("state_creation_node")

        # Subscriptions
        self.pose_sub = self.create_subscription(
            PoseStamped, "/vrpn_client_node/FP_exp_RL/pose", self.pose_callback, 10
        )
        self.goal_sub = self.create_subscription(
            Point, "/spacer_floating_platform/goal", self.goal_callback, 10
        )

        # Publisher
        self.state_pub = self.create_publisher(Float32MultiArray, "/rl_task_state", 10)

        # Timer (replaces rospy.Rate)
        self.timer = self.create_timer(0.1, self.publish_state)  # 10 Hz

        # Internal state variables
        self.robot_pose = None
        self.robot_quat = None
        self.robot_vel = np.zeros(6)  # Linear (x, y, z) and Angular (roll, pitch, yaw)
        self.goal_position = None
        self.pose_history = []

    def pose_callback(self, msg: PoseStamped):
        # Update pose and quaternion
        pos = msg.pose.position
        quat = msg.pose.orientation
        self.robot_pose = np.array([pos.x, pos.y, pos.z])
        self.robot_quat = np.array([quat.w, quat.x, quat.y, quat.z])

        # Store pose history for velocity calculation
        self.pose_history.append((self.robot_pose, self.get_clock().now().nanoseconds))
        if len(self.pose_history) > 30:  # Keep the last 30 messages
            self.pose_history.pop(0)

        # Update linear and angular velocities
        self.robot_vel[:3], self.robot_vel[3:] = self.derive_velocities()

    def goal_callback(self, msg: Point):
        # Update goal position
        self.goal_position = np.array([msg.x, msg.y, msg.z])

    def quaternion_to_heading(self, quat):
        """
        Converts a quaternion to a heading (yaw angle in radians).
        """
        _, x, y, z = quat
        heading = np.arctan2(2.0 * (x * y + z), 1.0 - 2.0 * (y**2 + z**2))
        return heading

    def derive_velocities(self):
        """
        Computes linear and angular velocities using pose history.
        Returns:
            lin_vel (np.ndarray): Linear velocity [vx, vy, vz].
            ang_vel (np.ndarray): Angular velocity [roll_rate, pitch_rate, yaw_rate].
        """
        if len(self.pose_history) < 2:
            return np.zeros(3), np.zeros(3)

        # Calculate velocity using the last two poses
        (pos1, t1), (pos2, t2) = self.pose_history[-2:]
        dt = (t2 - t1) * 1e-9  # Nanoseconds to seconds
        lin_vel = (pos2 - pos1) / dt

        # Placeholder for angular velocity (can be expanded)
        ang_vel = np.zeros(3)

        return lin_vel, ang_vel

    def compute_state(self):
        if self.robot_pose is None:
            self.get_logger().warn("Pose not received yet.")
            return None

        if self.goal_position is None:
            self.get_logger().warn("Goal not received yet.")
            return None

        # Compute state for `go_to_position` task
        position_error = self.goal_position[:2] - self.robot_pose[:2]
        position_dist = np.linalg.norm(position_error)

        # Compute heading and target heading
        heading = self.quaternion_to_heading(self.robot_quat)
        target_heading = np.arctan2(position_error[1], position_error[0])
        target_heading_error = np.arctan2(np.sin(target_heading - heading), np.cos(target_heading - heading))

        state = np.zeros(7)
        state[0] = position_dist
        state[1] = np.cos(target_heading_error)
        state[2] = np.sin(target_heading_error)
        state[3:5] = self.robot_vel[:2]  # Linear velocity
        state[5] = self.robot_vel[5]  # Yaw angular velocity
        state[6] = 0.0  # Placeholder for previous action

        return state


    def publish_state(self):
        state = self.compute_state()
        if state is not None:
            # Publish state
            state_msg = Float32MultiArray(data=state.tolist())
            self.state_pub.publish(state_msg)
            self.get_logger().info(f"Publishing state: {state}")


def main(args=None):
    rclpy.init(args=args)
    node = StateCreationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
