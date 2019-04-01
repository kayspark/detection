cmake_minimum_required(VERSION 3.8)
project(detection)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
add_executable(detection main.cpp MotionCapture.cpp MotionCapture.hpp Frame.hpp )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")  
  target_link_libraries(detection ${OpenCV_LIBS} stdc++fs)
else()
  target_link_libraries(detection ${OpenCV_LIBS})
endif()
