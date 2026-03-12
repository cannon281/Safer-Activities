#!/bin/bash

# Define container name and directory variables
NAME=safer_activities_cuda12
MAIN_DIR=$(realpath "$(dirname "$0")/../../")

# Check if nvidia-smi is working normally
echo "Checking NVIDIA driver status..."
nvidia-smi_start_time=$(date +%s)
nvidia-smi
nvidia-smi_end_time=$(date +%s)
nvidia-smi_duration=$((nvidia-smi_end_time - nvidia-smi_start_time))

# Build the Docker image
echo "Building Docker image for $NAME with CUDA 12.8 support..."
docker build -t $NAME -f Dockerfile .

# Set up X11 forwarding
xhost +local:docker

# Run the container with display forwarding and CUDA capabilities
echo "Running container with X11 forwarding and CUDA 12.8 support..."
docker run -it --rm \
    --name $NAME \
    --gpus all \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $MAIN_DIR:/home/work \
    $NAME:latest \
    bash

# Clean up X11 permissions after exit
xhost -local:docker
