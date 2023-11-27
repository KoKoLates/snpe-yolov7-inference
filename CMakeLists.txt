cmake_minimum_required(VERSION 3.1.0)
project(snpe-yolov7-inference)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -march=armv8-a")

set(SNPE_ROOT /data/snpe-sdk)
set(SNPE_INCLUDE_DIR ${SNPE_ROOT}/include/zdl)
set(SNPE_LIBRARY_DIR ${SNPE_ROOT}/lib/aarch64-ubuntu-gcc7.5)

message(STATUS "SNPE root path: ${SNPE_ROOT}")
message(STATUS "SNPE include: ${SNPE_INCLUDE_DIR}")
message(STATUS "SNPE library: ${SNPE_LIBRARY_DIR}")

find_package(OpenCV 4.8.0 REQUIRED)

include_directories{
    ${SNPE_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
}

set(SOURCES main.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${SNPE_ROOT}/lib/aarch64-ubuntu-gcc7.5/libSNPE.so 
    pthreads
)