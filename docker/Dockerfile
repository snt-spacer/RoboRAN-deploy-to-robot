# Use ROS2 Humble base image
FROM ros:humble

# Set the working directory to the main folder of your repository
WORKDIR /RANS_DeployToRobot

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-rclpy \
    ros-humble-geometry-msgs \
    ros-humble-joy \
    ros-humble-ackermann-msgs \
    ffmpeg \
    vim \
    git \
    byobu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /RANS_DeployToRobot
COPY ./requirements.txt .
# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/elharirymatteo/skrl.git skrl
WORKDIR /RANS_DeployToRobot/skrl
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -e .[torch]

# Source ROS2 setup
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

WORKDIR /RANS_DeployToRobot/ros_ws
RUN colcon build --symlink-install

# Source ROS2 workspace setup
RUN echo "source /RANS_DeployToRobot/ros_ws/install/setup.bash" >> ~/.bashrc

WORKDIR /RANS_DeployToRobot/ros_ws
# Default command
COPY ./ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
