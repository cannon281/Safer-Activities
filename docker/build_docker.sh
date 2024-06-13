#!/bin/bash

NAME=safer_activities_dataset_docker
docker build docker/. -t $NAME
