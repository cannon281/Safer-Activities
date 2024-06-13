#!/bin/bash

NAME=safer_activities_dataset_docker
MAIN_DIR=$(realpath "`dirname $0`/../")
FWD_DISPLAY="-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"
ENV_PLATFORM="DESKTOP"

cd "`dirname $0`"

docker run -it --rm -e PLATFORM=$ENV_PLATFORM --name $NAME \
    --gpus all \
    $FWD_DISPLAY \
    -v $MAIN_DIR:/home/work \
    $NAME:latest \
    bash
