cmake_minimum_required(VERSION 2.8.4)
project(detection)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(SOURCE Main.cpp Frame.cpp MotionCapture.cpp)
set(HEADER Frame.h MotionCapture.h)
add_executable(detection ${SOURCE})

#add_executable(detection Main.cpp)


find_package(OpenCV 4 REQUIRED)
target_link_libraries(detection ${OpenCV_LIBS} )
