cmake_minimum_required(VERSION 3.5)
project(snpe_task)

## Compile as C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -march=armv8-a")

## Setup the environment, include directories and libraries 
## for the Qualcomm SNPE SDK to access the system.
set(SNPE_ROOT /data/snpe-sdk)
set(SNPE_INCLUDE_DIR ${SNPE_ROOT}/include/zdl)
set(SNPE_LIBRARY_DIR ${SNPE_ROOT}/lib/aarch64-ubuntu-gcc7.5)

message(STATUS "SNPE root path: ${SNPE_ROOT}")
message(STATUS "SNPE include: ${SNPE_INCLUDE_DIR}")
message(STATUS "SNPE library: ${SNPE_LIBRARY_DIR}")

## System dependencies are found with CMake's conventions
## the main package might be used is OpenCV and Gstreamer
# find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV 4.8.0 REQUIRED)

## Add the SNPE pipeline and detection process as the static library 
## for the inference process to use
add_library(pipeline STATIC 
    src/VideoPipeline.cpp 
    src/SNPEPipeline.cpp 
    src/ObjectDetector.cpp 
)

## Add include directories of the library
target_include_directories(pipeline PUBLIC 
    include
    ${SNPE_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(pipeline PRIVATE 
    pthread
    ${OpenCV_LIBS}
    ${SNPE_ROOT}/lib/aarch64-ubuntu-gcc7.5/libSNPE.so
)

# Specify additional locations of header files
# Your package locations should be listed before other locations
include_directories(
    include
    ${JSONCPP_INCLUDE_DIRS} 
)

## Declare a C++ executable
## With all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
set(SOURCES main.cpp)
add_executable(snpe_task ${SOURCES})
target_link_libraries(snpe_task PRIVATE
    pipeline
    jsoncpp
    ${JSON_LIBRARIES}
)