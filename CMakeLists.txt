cmake_minimum_required(VERSION 3.10)
project(OpticalFlowInterpolation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

add_executable(optical_flow_interpolation
    src/main.cpp
    src/computeSymmetricFlow.cpp
    src/spatialRegularization.cpp
    src/warpUtils.cpp
    src/occlusionHandling.cpp
)

target_link_libraries(optical_flow_interpolation
    ${OpenCV_LIBS}
)
