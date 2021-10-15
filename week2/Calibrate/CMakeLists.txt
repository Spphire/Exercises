cmake_minimum_required(VERSION 3.10)
project(Calibrate)


set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Calibrate calibrate0.cpp)
target_link_libraries(  Calibrate ${OpenCV_LIBS}  )
