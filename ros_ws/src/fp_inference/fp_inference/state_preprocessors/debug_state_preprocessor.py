from . import Registerable
from . import BaseStatePreProcessor
import torch
from geometry_msgs.msg import PoseStamped


class DebugStatePreProcessor(Registerable, BaseStatePreProcessor):
    """A class to process state information from a robot.

    The state processor maintains a buffer of the robot's position, quaternion, and time. The state processor can
    compute the heading, rotation matrix, linear velocities in the world frame, and linear velocities in the body
    frame. The state processor is primed after a certain number of steps, after which the state can be accessed
    through the state variables. All the operations are done on the GPU to ensure mimum CPU-GPU data transfer.

    The DebugStatePreProcessor is just a dummy class to test the rest of the code. It does not process any data and
    is primed after 10 steps. The state variables are always zero.

    We add a priming step to this preprocessor as some may need to aggregate data over a few steps before being able
    to derive velocities or other quantities. This is the case for the OptitrackStatePreProcessor, which needs to
    aggregate position and quaternion data over a few steps to derive velocities.
    """

    def __init__(self, buffer_size: int = 30, device: str = "auto", **kwargs) -> None:
        """Initialize the state processor with a buffer size and device.

        Args:
            buffer_size (int, optional): The size of the state buffer. Defaults to 30.
            device (str, optional): The device to perform computations on. Defaults to "auto".
        """
        super().__init__(buffer_size=buffer_size, device=device, **kwargs)
        self.ROS_TYPE = PoseStamped
        self.ROS_CALLBACK = self.update_state_ROS
        self.ROS_QUEUE_SIZE = 1

    def update_state_ROS(self, data: PoseStamped, **kwargs) -> None:
        """Update the state processor with a new ROS message.

        If the state processor is primed, update the state. When primed, the state can be accessed through the
        state variables.
        """
        self._position = torch.zeros((1, 3), device=self._device)
        self._quaternion = torch.zeros((1, 4), device=self._device)
        self._time = torch.zeros((1), device=self._device)
        self._linear_velocities_world = torch.zeros((1, 3), device=self._device)
        self._angular_velocities_world = torch.zeros((1, 3), device=self._device)
        self._step += 1

        if self._step > 10:
            self._is_primed = True

    def reset(self) -> None:
        """Reset the state processor. The state variables are set to zero and the state processor is unprimed."""
        super().reset()
        self._is_primed = False
