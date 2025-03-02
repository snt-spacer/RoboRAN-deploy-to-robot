#!/usr/bin/env python3

import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Pose, PoseStamped, PointStamped, PoseArray
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry
import numpy as np
import copy

from .goal_formaters import GoalFormaterFactory

from rcl_interfaces.msg import ParameterDescriptor


class GoalPublisherNode(Node):
    """Goal publisher node.
    
    The goal publisher node is used to publish goals to the observation formater node. The goals are read from a YAML
    file, and published to the observation formater node. Depending on the configuration, the goals can be published in
    the local or global frame. To iterate through the goals, the 'A' button on the joystick must be pressed. To exit the
    node, the 'B' button must be pressed.
    """

    def __init__(self) -> None:
        """Initialize the goal publisher node.
        
        This method initializes the goal publisher node with the following parameters:
        - task_name: The name of the task to be executed. Default is 'GoToPosition'.
        - goals_file_path: The path to the file containing the goals. Default is ''. The goals are read from this file.
            It must follow the format associated with the type of task.
        - wait_for_task_timeout: The time to wait for the task to be completed before exiting. Default is 600 seconds.
            If the task is not completed within this time, the node will unlock the current goal.
        - is_local: If the goals should be published in the local or global frame. Default is True. If True, the goals
            are published in the local frame. That is the frame of the robot. This is useful when sending multiple goals
            this way the goals are sent relative to the robot. If False, the goals are published in the global frame.
        - odom_msg_type: The type of the odometry message. Default is 'Odometry'. The odometry message is used to
            perform the transformation between the local and global frames.
        """

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

        # Wait for task timeout
        wait_for_task_timeout_desc = ParameterDescriptor(
            description="The time to wait for the task to be completed before exiting."
        )
        self.declare_parameter("wait_for_task_timeout", 600, wait_for_task_timeout_desc)
        self._wait_for_task_timeout = self.get_parameter("wait_for_task_timeout").get_parameter_value().integer_value

        # Local or global frame
        is_local_desc = ParameterDescriptor(description="If the goals should be published in the local or global frame.")
        self.declare_parameter("is_local", True, is_local_desc)
        self._is_local = self.get_parameter("is_local").get_parameter_value().bool_value

        # Message types
        self._odom_msg_type_desc = ParameterDescriptor(description="The type of the odometry message.")
        self.declare_parameter("odom_msg_type", "Odometry", self._odom_msg_type_desc)
        self._odom_msg_type = self.get_parameter("odom_msg_type").get_parameter_value().string_value

        self._task_is_done = False

        self.build()

    def build(self) -> None:
        """Build the goal publisher node."""

        self.goal_formater = GoalFormaterFactory.create(self._task_name, self._goals_file_path)

        self._a = self._prev_a = self._b = self._prev_b = 0
        self._a_was_pressed = self._a_was_released = self._b_was_pressed = self._b_was_released = False
        self._permanent_a_press = self._permanent_a_release = self._permanent_b_press = self._permanent_b_release = (
            False
        )
        self._goals_exhausted = False

        # ROS2 Subscriptions
        frame_type = None
        frame_callback = None
        if self._odom_msg_type == "Odometry":
            frame_type = Odometry
            frame_callback = self.odometry_callback
        elif self._odom_msg_type == "PoseStamped":
            frame_type = PoseStamped
            frame_callback = self.pose_stamped_callback
        elif self._odom_msg_type == "TF":
            raise NotImplementedError("TF is not supported yet.")
        else:
            raise ValueError("Invalid odometry message type.")
        
        self.create_subscription(frame_type, "state_preprocessor_input", frame_callback, 1)
        self.create_subscription(Bool, "task_is_done", self.task_is_done_callback, 1)
        self.create_subscription(Joy, "/joy", self.joy_callback, 1)

        # ROS2 Publishers
        self.goal_pub = self.create_publisher(
            self.goal_formater.ROS_TYPE,
            "observation_formater_input",
            self.goal_formater.QOS_PROFILE,
        )

    def wait_for_A_button(self) -> None:
        """Wait for the 'A' button to be pressed.
        
        The method blocks until the 'A' button is pressed.
        """
        while rclpy.ok():
            if self._permanent_a_press:
                self._permanent_a_press = False
                break

    def wait_for_A_or_B_button(self) -> bool:
        """Wait for the 'A' or 'B' button to be pressed.

        The method blocks until the 'A' or 'B' button is pressed.
        
        Returns:
            bool: True if the 'A' button was pressed, False if the 'B' button was pressed.
        """
        while rclpy.ok():
            if self._permanent_a_press:
                self._permanent_a_press = False
                return True
            if self._b_was_pressed:
                self._permanent_b_press = False
                return False
            
    def odometry_callback(self, msg: Odometry) -> None:
        """Odometry callback.
        
        The method is called when an odometry message is received. The odometry message is used to transform the goals
        from the global to the local frame.

        Note: The odometry message is converted to a PoseStamped message. Ideally we should use the TF tree to
        perform the transformation. However, some form of robot localization do not provide the TF tree. Hence,
        for now, we perform the transformation manually.
        
        Args:
            msg (Odometry): The odometry message.
        """
        self._global_frame = msg.header.frame_id
        self._local_pose = PoseStamped()
        self._local_pose.header = msg.header
        self._local_pose.pose = msg.pose.pose

    def pose_stamped_callback(self, msg: PoseStamped) -> None:
        """PoseStamped callback.

        The method is called when a PoseStamped message is received. The PoseStamped message is used to transform the
        goals from the global to the local frame.

        Note: The odometry message is converted to a PoseStamped message. Ideally we should use the TF tree to
        perform the transformation. However, some form of robot localization do not provide the TF tree. Hence,
        for now, we perform the transformation manually.
        
        Args:
            msg (PoseStamped): The PoseStamped message.
        """
        self._global_frame = msg.header.frame_id
        self._local_pose = msg

    def convert_point_to_local_frame(self, point: PointStamped) -> PointStamped:
        """Convert a point to the local frame.
        
        The method converts a point from the global to the local frame. The transformation is performed using the
        pose of the robot in the global frame.

        Args:
            point (PointStamped): The point to convert.
        
        Returns:
            PointStamped: The point in the local frame.
        """
        if self._local_pose is not None:
            # Build the transform
            q = [self._local_pose.pose.orientation.x, self._local_pose.pose.orientation.y, self._local_pose.pose.orientation.z, self._local_pose.pose.orientation.w]
            R = Rotation.from_quat(q,scalar_first=False)
            T = np.array([self._local_pose.pose.position.x, self._local_pose.pose.position.y, self._local_pose.pose.position.z])
            # Transform the point
            point = np.array([point.point.x, point.point.y, point.point.z])
            point = R.apply(point) + T
            point = PointStamped()
            point.point.x = point[0]
            point.point.y = point[1]
            point.point.z = point[2]
            return point
        else:
            return point

    def convert_pose_to_local_frame(self, pose: PoseStamped) -> PoseStamped:
        """Convert a pose to the local frame.

        The method converts a pose from the global to the local frame. The transformation is performed using the
        pose of the robot in the global frame.

        Args:
            pose (PoseStamped): The pose to convert.
        
        Returns:
            PoseStamped: The pose in the local frame.
        """
        if self._local_pose is not None:
            # Build the transform
            q = [self._local_pose.pose.orientation.x, self._local_pose.pose.orientation.y, self._local_pose.pose.orientation.z, self._local_pose.pose.orientation.w]
            R = Rotation.from_quat(q,scalar_first=False)
            T = np.array([self._local_pose.pose.position.x, self._local_pose.pose.position.y, self._local_pose.pose.position.z])
            # Transform the pose
            point = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
            point = R.apply(point) + T
            
            pose = PoseStamped()
            pose.pose.position.x = pose[0]
            pose.pose.position.y = pose[1]
            pose.pose.position.z = pose[2]
            q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            R2 = Rotation.from_quat(q, scalar_first=False).as_matrix()
            R2 = R.as_matrix() @ R2
            q = Rotation.from_matrix(R2).as_quat()
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            return pose
        else:
            return pose

    def convert_pose_array_to_local_frame(self, pose_array: PoseArray) -> PoseArray:
        """Convert a pose array to the local frame.

        The method converts a pose array from the global to the local frame. The transformation is performed using the
        pose of the robot in the global frame.

        Args:
            pose_array (PoseArray): The pose array to convert.
        
        Returns:
            PoseArray: The pose array in the local frame.
        """
        if self._local_pose is not None:
            # Build the transform
            q = [self._local_pose.pose.orientation.x, self._local_pose.pose.orientation.y, self._local_pose.pose.orientation.z, self._local_pose.pose.orientation.w]
            R = Rotation.from_quat(q,scalar_first=False)
            T = np.array([self._local_pose.pose.position.x, self._local_pose.pose.position.y, self._local_pose.pose.position.z])
            # Transform the pose array
            new_pose_array = PoseArray()
            new_pose_array.header = pose_array.header
            new_pose_array.header.frame_id = self._global_frame
            for pose in pose_array.poses:
                new_pose = Pose()
                point = np.array([pose.position.x, pose.position.y, pose.position.z])
                point = R.apply(point) + T
                new_pose.position.x = point[0]
                new_pose.position.y = point[1]
                new_pose.position.z = point[2]
                q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                R2 = Rotation.from_quat(q, scalar_first=False).as_matrix()
                R2 = R.as_matrix() @ R2
                q = Rotation.from_matrix(R2).as_quat()
                new_pose.orientation.x = q[0]
                new_pose.orientation.y = q[1]
                new_pose.orientation.z = q[2]
                new_pose.orientation.w = q[3]
                new_pose_array.poses.append(new_pose)
            return new_pose_array
        else:
            return pose_array
        
    def auto_project(self, msg: PointStamped | PoseStamped | PoseArray) -> PointStamped | PoseStamped | PoseArray:
        if self._is_local:
            if isinstance(msg, PointStamped):
                return self.convert_point_to_local_frame(msg)
            elif isinstance(msg, PoseStamped):
                return self.convert_pose_to_local_frame(msg)
            elif isinstance(msg, PoseArray):
                return self.convert_pose_array_to_local_frame(msg)
            else:
                raise ValueError("Invalid message type.")
        else:
            return

    def task_is_done_callback(self, msg: Bool) -> None:
        self._task_is_done = msg.data

    def joy_callback(self, msg: Joy) -> None:
        self._buttons = msg.buttons
        # A button
        self._prev_a = copy.copy(self._a)
        self._a = self._buttons[0]
        self._a_was_pressed = self._prev_a == 0 and self._a == 1
        if self._a_was_pressed:
            self._permanent_a_press = True
        self._a_was_released = self._prev_a == 1 and self._a == 0
        if self._a_was_released:
            self._permanent_a_release = True
        # B button
        self._prev_b = copy.copy(self._b)
        self._b = self._buttons[1]
        self._b_was_pressed = self._prev_b == 0 and self._b == 1
        if self._b_was_pressed:
            self._permanent_b_press = True
        self._b_was_released = self._prev_b == 1 and self._b == 0
        if self._b_was_released:
            self._permanent_b_release = True

    def run(self):
        self.get_logger().info("Press 'A' to start!")
        self.wait_for_A_button()
        self.get_logger().info("Starting the goal publisher node...")
        while rclpy.ok():
            self.run_goal_loop()
            if self.goal_formater.goal is None:
                self._goals_exhausted = True
            if self._goals_exhausted:
                self.get_logger().info("Exhausted all goals. Press 'B' to exit, press 'A' to restart.")
                repeat = self.wait_for_A_or_B_button()
                if repeat:
                    self.goal_formater.reset()
                    self._goals_exhausted = False
                else:
                    break
            else:
                self.get_logger().info("Press the 'A' key to send the next goal!")
                self.wait_for_A_button()

    def run_goal_loop(self) -> None:
        rate = self.create_rate(0.5)

        goal_was_sent = False
        while rclpy.ok():
            if self.goal_pub.get_subscription_count() == 0:
                self.get_logger().info("Waiting for subscriber...")
            if self.goal_pub.get_subscription_count() != 0 and goal_was_sent == False:
                self.goal_pub.publish(self.auto_project(self.goal_formater.goal))
                self.get_logger().info(self.goal_formater.log_publish())
                goal_was_sent = True

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
