from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
import torch
import copy


class BaseStatePreProcessor:
    """A class to process state information from a robot.

    The state processor maintains a buffer of the robot's position, quaternion, and time. The state processor can
    compute the heading, rotation matrix, linear velocities in the world frame, and linear velocities in the body
    frame. The state processor is primed after a certain number of steps, after which the state can be accessed
    through the state variables. All the operations are done on the GPU to ensure mimum CPU-GPU data transfer.

    IMPORTANT:
    The base state preprocessor expects the following:
    - The state processor is updated with a ROS message.
    - The ROS message is providing the position and quaternion of the robot in a World frame.
    - The ROS message is providing the linear and angular velocities of the robot in a World frame.

    If the ROS message is not providing the linear and angular velocities in the World frame, the child class should
    overload the update functions / properties associated with the linear and angular velocities. Examples of such
    functions can be seen in the `OdometryStatePreprocessor`.
    """

    def __init__(self, buffer_size: int = 30, device: str = "auto", **kwargs) -> None:
        """Initialize the state processor with a buffer size and device.

        Args:
            buffer_size (int, optional): The size of the state buffer. Defaults to 30.
            device (str, optional): The device to perform computations on. Defaults to "auto".
        """
        # General parameters
        self._device = device
        self._buffer_size = buffer_size
        self.ROS_TYPE = None
        self.ROS_CALLBACK = self.update_state_ROS
        self.QOS_PROFILE = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1
        )

        # Device selection
        if device == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # State variables
        self._time: torch.Tensor = torch.zeros((1, 1), device=self._device)
        self._position: torch.Tensor = torch.zeros((1, 3), device=self._device)
        self._heading: torch.Tensor = torch.zeros((1, 1), device=self._device)
        self._quaternion: torch.Tensor = torch.zeros((1, 4), device=self._device)
        self._rotation_matrix: torch.Tensor = torch.zeros((1, 2, 2), device=self._device)
        self._inverse_rotation_matrix: torch.Tensor = torch.zeros((1, 2, 2), device=self._device)
        self._tranformation_matrix: torch.Tensor = torch.zeros((1, 3, 3), device=self._device)
        self._linear_velocities_world: torch.Tensor = torch.zeros((1, 3), device=self._device)
        self._angular_velocities_world: torch.Tensor = torch.zeros((1, 3), device=self._device)
        self._linear_velocities_body: torch.Tensor = torch.zeros((1, 3), device=self._device)
        self._angular_velocities_body: torch.Tensor = torch.zeros((1, 3), device=self._device)

        # Lazy updates of state variables
        self._step_heading: int = 0
        self._step_rotation_matrix: int = 0
        self._step_inverse_rotation_matrix: int = 0
        self._step_linear_velocity_body: int = 0
        self._step_angular_velocity_body: int = 0
        self._step_logs: int = 0

        # State priming (need to wait for a few steps before a reliable state is available)
        self._is_primed = False
        self._step = 0
        self._start_time = -1

        # Log hook
        self.build_logs()

    def build_logs(self) -> None:
        """Build the logs for the state processor.

        In this case, we log the position, quaternion, heading, linear velocities, angular velocities, ROS time,
        and elapsed time. Also build the log specs for the state variables. Logs specs are used to define the units
        of the state variables.
        """
        # State preprocessor logs
        self._logs = {}
        self._logs["position_world"] = torch.zeros((1, 3), device=self._device)
        self._logs["quaternion_world"] = torch.zeros((1, 4), device=self._device)
        self._logs["heading_world"] = torch.zeros((1, 1), device=self._device)
        self._logs["linear_velocities_world"] = torch.zeros((1, 3), device=self._device)
        self._logs["angular_velocities_world"] = torch.zeros((1, 3), device=self._device)
        self._logs["linear_velocities_body"] = torch.zeros((1, 3), device=self._device)
        self._logs["angular_velocities_body"] = torch.zeros((1, 3), device=self._device)
        self._logs["ros_time"] = torch.zeros((1, 1), device=self._device)
        self._logs["elapsed_time"] = torch.zeros((1, 1), device=self._device)

        # Log specifications, this is used to provide a user friendly way to interpret the logs.
        self._logs_specs = {}
        self._logs_specs["position_world"] = [".x.m", ".y.m", ".z.m"]
        self._logs_specs["quaternion_world"] = [".w.quat", ".x.quat", ".y.quat", ".z.quat"]
        self._logs_specs["heading_world"] = [".rad"]
        self._logs_specs["linear_velocities_world"] = ["x.m/s", "y.m/s", "z.m/s"]
        self._logs_specs["angular_velocities_world"] = ["x.rad/s", "y.rad/s", "z.rad/s"]
        self._logs_specs["linear_velocities_body"] = ["x.m/s", "y.m/s", "z.m/s"]
        self._logs_specs["angular_velocities_body"] = ["x.rad/s", "y.rad/s", "z.rad/s"]
        self._logs_specs["ros_time"] = [".s"]
        self._logs_specs["elapsed_time"] = [".s"]

    def update_logs(self) -> None:
        """Update the logs for the state processor."""
        self._logs["position_world"] = self.position
        self._logs["quaternion_world"] = self.quaternion
        self._logs["heading_world"] = self.heading
        self._logs["linear_velocities_world"] = self.linear_velocities_world
        self._logs["angular_velocities_world"] = self.angular_velocities_world
        self._logs["linear_velocities_body"] = self.linear_velocities_body
        self._logs["angular_velocities_body"] = self.angular_velocities_body
        self._logs["ros_time"] = torch.tensor([[self._time]], device=self._device)
        if self._start_time == -1:
            self._start_time = copy.copy(self._time)
        self._logs["elapsed_time"] = torch.tensor([[self._time - self._start_time]], device=self._device)

    @property
    def logs(self) -> dict["str", torch.Tensor]:
        """Logs for the state processor.

        The logs are updated using a lazy update mechanism to avoid unnecessary computations. The logs are updated
        every time they are accessed.
        """
        if self._step_logs != self._step:
            self._step_logs = copy.copy(self._step)
            self.update_logs()
        return self._logs

    @property
    def logs_names(self) -> list[str]:
        """Logs names for the state processor."""
        return self._logs.keys()

    @property
    def logs_specs(self) -> dict[str, list[str]]:
        """The logs specifications for the state processor."""
        return self._logs_specs

    @property
    def is_primed(self) -> bool:
        """Whether the state processor is primed and ready to provide state information."""
        return self._is_primed

    @property
    def step(self) -> int:
        """The current step count."""
        return self._step

    @property
    def position(self) -> torch.Tensor | None:
        """The current position (x, y, z). If the position is not available, output None."""
        return self._position

    @property
    def heading(self) -> torch.Tensor | None:
        """The current heading (yaw angle in radians).

        If the heading is not available, return None. The update is lazy, so the heading is only computed when
        it is requested. Also, the heading is only updated when the quaternion changes.
        """
        if (self.quaternion is not None) and (self._step_heading != self._step):
            self._heading[0] = self.get_heading_from_quat(self.quaternion)
            # Update the step count for lazy updates
            self._step_heading = copy.copy(self._step)
        return self._heading

    @property
    def quaternion(self) -> torch.Tensor | None:
        """The current quaternion.

        Quaternion is in the form (w, x, y, z). If the quaternion is not available, output None.
        """
        return self._quaternion

    @property
    def rotation_matrix(self) -> torch.Tensor | None:
        """The current rotation matrix. The format is a 2x2 matrix.

        If the rotation matrix is not available, return None. The update is lazy, so the rotation matrix is only
        computed when it is requested. Also, the rotation matrix is only updated when the heading changes.
        """
        if (self.quaternion is not None) and (self._step_rotation_matrix != self._step):
            self._rotation_matrix[0] = self.get_rotation_matrix_from_heading(self.heading)
            # Update the step count for lazy updates
            self._step_rotation_matrix = copy.copy(self._step)
        return self._rotation_matrix
    
    @property
    def inverse_rotation_matrix(self) -> torch.Tensor | None:
        """The current inverse rotation matrix. The format is a 2x2 matrix.
        
        If the inverse rotation matrix is not available, return None.
        """
        if (self.rotation_matrix is not None) and (self._step_inverse_rotation_matrix != self._step):
            self._inverse_rotation_matrix[0] = self.rotation_matrix[0].T
            self._step_inverse_rotation_matrix = copy.copy(self._step)
        return self._inverse_rotation_matrix

    @property
    def linear_velocities_world(self) -> torch.Tensor | None:
        """Return the linear velocities in the world frame. If the velocities are not available, return None."""
        return self._linear_velocities_world

    @property
    def angular_velocities_world(self) -> torch.Tensor | None:
        """Return the angular velocities in the world frame. If the velocities are not available, return None."""
        return self._angular_velocities_world

    @property
    def linear_velocities_body(self) -> torch.Tensor | None:
        """Return the linear velocities in the body frame. If the velocities are not available, return None."""
        if (self.linear_velocities_world is not None) and (self._step_linear_velocity_body != self._step):
            self.get_linear_velocities_body()
            # Update the step count for lazy updates
            self._step_linear_velocity_body = copy.copy(self._step)
        return self._linear_velocities_body

    @property
    def angular_velocities_body(self) -> torch.Tensor | None:
        """Return the angular velocities in the body frame.

        If the velocities are not available, return None.
        Note: The angular velocities in the body frame are the same as the angular velocities in the world frame.
        """
        if (self.angular_velocities_world is not None) and (self._step_angular_velocity_body != self._step):
            self._angular_velocities_body = self._angular_velocities_world
            # Update the step count for lazy updates
            self._step_angular_velocity_body = copy.copy(self._step)
        return self._angular_velocities_body

    def get_logs(self):
        """Hook used by the logger to get the logs for the state processor."""
        return self.logs

    @staticmethod
    def get_heading_from_quat(quaternion: torch.Tensor) -> None:
        """Convert a quaternion to a heading (yaw angle in radians).

        Args:
            quaternion (torch.Tensor): The quaternion in the form (w, x, y, z). Tensor shape: [N, 4]
        """
        return torch.arctan2(
            2.0 * (quaternion[:, 1] * quaternion[:, 2] + quaternion[:, 3] * quaternion[:, 0]),
            1.0 - 2.0 * (quaternion[:, 2] * quaternion[:, 2] + quaternion[:, 3] * quaternion[:, 3]),
        )

    @staticmethod
    def get_rotation_matrix_from_heading(heading: torch.Tensor) -> torch.Tensor:
        """Generate a 2D rotation matrix for a given heading angle.

        Args:
            heading (torch.Tensor): The heading angle in radians. Tensor shape: [N, 1]
        """
        return torch.tensor(
            [[torch.cos(heading), torch.sin(heading)], [-torch.sin(heading), torch.cos(heading)]],
            device=heading.device,
        )

    @staticmethod
    def get_quat_from_heading(heading: torch.Tensor) -> torch.Tensor:
        """Convert a heading to a quaternion.

        Args:
            heading (torch.Tensor): The heading in radians. Tensor shape: [N, 1]
        """
        return torch.cat(
            (torch.cos(heading / 2), torch.zeros_like(heading), torch.zeros_like(heading), torch.sin(heading / 2)),
            dim=1,
        )

    @staticmethod
    def get_transfrom_matrix(position: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Generate the 2D tranformation matrix associated with the world to body frame transformation.

        Args:
            position (torch.Tensor): The position tensor. Tensor shape: [N, 3]
            rotation_matrix (torch.Tensor): The rotation matrix tensor. Tensor shape: [N, 2, 2]
        """
        transfrom_matrix = torch.ones((position.shape[0], 3, 3), device=position.device)
        transfrom_matrix[:, :2, :2] = rotation_matrix[:]
        transfrom_matrix[:, :2, 2] = position[:, :2]
        return transfrom_matrix

    def get_linear_velocities_body(self):
        """Compute linear velocities in the body frame."""
        self._linear_velocities_body[0, :2] = self.rotation_matrix[0] @ self.linear_velocities_world[0, :2]

    def get_pose_in_local_frame(self, pose: torch.Tensor) -> torch.Tensor:
        """Transform a pose from the world frame to the local frame.

        While the tensor are provided in a 6DoF format, the function solves for the 3DoF pose in the local frame,
        assuming x, y translation and yaw (z) rotation.

        Args:
            pose (torch.Tensor): The pose in the world frame. Tensor shape: [N, 7]
        """
        if len(pose.shape) != 2:
            pose = pose.unsqueeze(0)

        # Get the position and quaternion
        position = pose[:, :3]
        quaternion = pose[:, 3:]
        # Get the heading, rotation matrix and transformation matrix
        heading = self.get_heading_from_quat(quaternion)
        rotation_matrix = self.get_rotation_matrix_from_heading(heading)
        transformation_matrix = self.get_transfrom_matrix(position, rotation_matrix)
        # Use the transformation matrix to get the pose in the local frame
        position_transformed = torch.bmm(transformation_matrix, position)
        heading_transformed = heading - self.heading
        quaternion_transformed = self.get_quat_from_heading(heading_transformed)
        return torch.cat((position_transformed, quaternion_transformed), dim=1)

    def get_pose_in_local_frame_ROS(self, pose: PoseStamped) -> torch.Tensor:
        """Transform a pose from the world frame to the local frame.

        Args:
            pose (PoseStamped): The pose in the world frame.
        """
        # Get the position and quaternion
        position = torch.tensor(
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], device=self._device
        ).unsqueeze(0)
        quaternion = torch.tensor(
            [pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z],
            device=self._device,
        ).unsqueeze(0)
        # Get the heading, rotation matrix and transformation matrix
        heading = self.get_heading_from_quat(quaternion)
        rotation_matrix = self.get_rotation_matrix_from_heading(heading)
        transformation_matrix = self.get_transfrom_matrix(position, rotation_matrix)
        # Use the transformation matrix to get the pose in the local frame
        position_transformed = torch.bmm(transformation_matrix, position)
        heading_transformed = heading - self.heading
        quaternion_transformed = self.get_quat_from_heading(heading_transformed)
        return torch.cat((position_transformed, quaternion_transformed), dim=1)

    def update_state_ROS(self, *args, **kwargs) -> None:
        """Update the state processor with a new ROS message.

        If the state processor is primed, update the state. When primed, the state can be accessed through the state
        variables. In a typical use case, this is the only method that needs to be overloaded by the child class.
        """
        raise NotImplementedError("Update state ROS method not implemented")

    def reset(self) -> None:
        """Reset the state processor to its initial state."""
        self.build_logs()
        self._last_time = None
        self.is_primed = False
        self._step = 0
        self._start_time = -1

        # State variables
        self._time: torch.Tensor = torch.zeros((1, 1), device=self._device)
        self._position: torch.Tensor = torch.zeros((1, 3), device=self._device)
        self._heading: torch.Tensor = torch.zeros((1, 1), device=self._device)
        self._quaternion: torch.Tensor = torch.zeros((1, 4), device=self._device)
        self._rotation_matrix: torch.Tensor = torch.zeros((1, 2, 2), device=self._device)
        self._inverse_rotation_matrix: torch.Tensor = torch.zeros((1, 2, 2), device=self._device)
        self._tranformation_matrix: torch.Tensor = torch.zeros((1, 3, 3), device=self._device)
        self._linear_velocities_world: torch.Tensor = torch.zeros((1, 3), device=self._device)
        self._angular_velocities_world: torch.Tensor = torch.zeros((1, 3), device=self._device)
        self._linear_velocities_body: torch.Tensor = torch.zeros((1, 3), device=self._device)
        self._angular_velocities_body: torch.Tensor = torch.zeros((1, 3), device=self._device)

        # Reset lazy updates
        self._step_heading = 0
        self._step_rotation_matrix = 0
        self._step_inverse_rotation_matrix = 0
        self._step_linear_velocity_body = 0
        self._step_angular_velocity_body = 0
        self._step_logs = 0
