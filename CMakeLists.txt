# Tell CMake exactly where your OpenCV installation is
set(OpenCV_DIR "C:/Users/olade/opencv/build")

cmake_minimum_required(VERSION 3.10)
project(OpticalFlowInterpolation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(optical_flow_interpolation
    src/main.cpp
    src/frame_interpolation.cpp
    src/occlusion.cpp
    src/spatialRegularization.cpp
    src/warpUtils.cpp
)

target_link_libraries(optical_flow_interpolation
    ${OpenCV_LIBS}
)