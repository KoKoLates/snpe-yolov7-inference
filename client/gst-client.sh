#! /bin/bash

gst-launch-1.0 udpsrc port=11024 ! application/x-rtp, encoding-name=H264, payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink