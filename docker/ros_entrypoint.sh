#!/bin/bash
set -e

# setup ros environment
colcon build --symlink-install
source "/RANS_DeployToRobot/ros_ws/install/setup.bash"
export ROS_DOMAIN_ID=0

exec "$@"
