cmake_minimum_required(VERSION 3.5)
project(snpe_test)

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

add_library(pipeline STATIC 
    src/VideoPipeline.cpp 
    src/SNPEPipeline.cpp 
    src/ObjectDetector.cpp 
)

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

include_directories(
    include
    ${JSONCPP_INCLUDE_DIRS} 
)

set(SOURCES main.cpp)
add_executable(snpe_test ${SOURCES})
target_link_libraries(snpe_test PRIVATE
    pipeline
    jsoncpp
    ${JSON_LIBRARIES}
)