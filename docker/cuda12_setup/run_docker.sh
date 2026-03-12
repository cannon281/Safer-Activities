#!/bin/bash

# Name of the image built by build_docker_cuda12.sh
NAME=safer_activities_cuda12
# Define the main project directory relative to the script's location
# Assumes this script is in Safer-Activities/docker/cuda12_setup
MAIN_DIR=$(realpath "$(dirname "$0")/../../")
# X11 Forwarding flags
FWD_DISPLAY="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
# Environment variable (optional, kept from original script)
ENV_PLATFORM="DESKTOP"

echo "Running container from image: $NAME"
echo "Mounting project directory: $MAIN_DIR to /home/work"
echo "Ensure X11 forwarding is enabled on host (run 'xhost +local:docker')"

# Navigate to the script's directory before running docker
# This isn't strictly necessary for docker run, but good practice
cd "$(dirname "$0")"

docker run -it --rm \
    -e PLATFORM=$ENV_PLATFORM \
    --name ${NAME}_instance \
    --gpus all \
    $FWD_DISPLAY \
    -v "$MAIN_DIR":/home/work \
    $NAME:latest \
    bash -c "cd /home/work/inference/pyskl && pip install --no-deps -e . > /dev/null 2>&1 && cd /home/work && exec bash"

echo "Container exited."
