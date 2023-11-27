# Makefile for GStreamer Example
# Author: Po-Ting Ko
# Date: 2023-11-22

# Compiler
CC=gcc

# Compiler flags using pkg-config for GStreamer
FLAGS=`pkg-config --cflags --libs gstreamer-1.0`

# Source files
SOURCES=main.c

# Targets (executable names derived from source file names)
TARGETS=$(foreach n, $(SOURCES), $(basename $(n)))

# Default target: build all executables
all: ${TARGETS}

# Phony targets to avoid conflicts with file names
.PHONY: all clean

# Rule to build each target from its corresponding source file
${TARGETS}: %:%.c
	$(CC) -Wall $< -o $@ $(FLAGS)

# Clean up compiled executables
clean:
	rm -f ${TARGETS}
