#! /bin/bash

gst-launch-1.0 -e qtiqmmfsrc name=qmmf camera=0 ! video/x-h264, framerate=30/1 ! h264parse config-interval=1 ! queue ! rtph264pay ! udpsink host=192.168.50.74 port=11024
