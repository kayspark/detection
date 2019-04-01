cmake_minimum_required(VERSION 3.8)
project(detection)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

add_executable(detection main.cpp MotionCapture.cpp MotionCapture.hpp Frame.hpp )

#add_executable(detection Main.cpp)


find_package(OpenCV REQUIRED)
target_link_libraries(detection ${OpenCV_LIBS} )
