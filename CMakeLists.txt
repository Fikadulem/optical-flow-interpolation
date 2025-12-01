cmake_minimum_required(VERSION 3.10)
project(optical-flow-interpolation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)

# Collect only project sources, exclude build/ directory
file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${CMAKE_SOURCE_DIR}/include/*.h ${CMAKE_SOURCE_DIR}/include/*.hpp)

# Add executable
# add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
add_executable(optical-flow-interpolation
    src/main.cpp
)
target_link_libraries(optical-flow-interpolation ${OpenCV_LIBS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} include)
# Add this line to install the binary
install(TARGETS optical-flow-interpolation DESTINATION /usr/local/bin)



# cmake_minimum_required(VERSION 3.10)
# project(optical-flow-interpolation)

# set(CMAKE_CXX_STANDARD 17)

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# add_executable(optical-flow-interpolation
#     src/main.cpp
# )

# target_link_libraries(optical-flow-interpolation ${OpenCV_LIBS})


# ,,,,,