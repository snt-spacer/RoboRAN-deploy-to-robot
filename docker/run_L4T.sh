#!/bin/bash
xhost +
docker run --name rans-ros-deploy-container -it --runtime=nvidia -e "ACCEPT_EULA=Y" --rm --network host --ipc host \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v ${PWD}/ros_ws/src:/RANS_DeployToRobot/ros_ws/src \
    -e DISPLAY \
    -e "PRIVACY_CONSENT=Y" \
    -v ${PWD}/models:/RANS_DeployToRobot/models \
    rans-ros-deploy-l4t:latest
