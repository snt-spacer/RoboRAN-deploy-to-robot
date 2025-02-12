#!/bin/bash
xhost +
docker run --name rans-ros-deploy-container -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host --ipc=host \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -e DISPLAY \
    -e "PRIVACY_CONSENT=Y" \
    -v ${PWD}:/workspace \
    rans-ros-deploy:latest
