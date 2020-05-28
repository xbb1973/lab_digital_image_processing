cmake_minimum_required(VERSION 3.13)
project(lab_digital_image)

set(CMAKE_CXX_STANDARD 17)

#find_library(OpenCV)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(graphics_opencv graphics.cpp)

target_link_libraries(graphics_opencv ${OpenCV_LIBS})