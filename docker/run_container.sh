#!/bin/bash

NAME=safer_activities_dataset_docker
MAIN_DIR=$(realpath "`dirname $0`/../")

cd "`dirname $0`"

docker build . -t $NAME

docker run -d -it --name $NAME \
    --gpus all \
    -v $MAIN_DIR:/home/work \
    $NAME:latest \
    bash
