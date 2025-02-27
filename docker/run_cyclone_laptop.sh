#!/bin/bash
xhost +
docker run --name rans-ros-deploy-container -it --privileged -e "ACCEPT_EULA=Y" --rm --network host --ipc host \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v ${PWD}/ros_ws/src:/RANS_DeployToRobot/ros_ws/src \
    -v ${PWD}/models:/RANS_DeployToRobot/models \
    -v ${PWD}/ros_experiments_logs:/RANS_DeployToRobot/ros_experiments_logs \
    rans-ros-deploy-cyclone-laptop:latest
