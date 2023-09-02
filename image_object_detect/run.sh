#!/bin/bash
# Make sure current directory is image_object_detect, if its not then exit
if [ ${PWD##*/} != "image_object_detect" ]; then
    echo "Please run this script from the image_object_detect directory"
    exit 1
fi

go run main.go
