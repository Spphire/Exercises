cmake_minimum_required(VERSION 3.10)
project(test2)


set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test2 main.cpp)
target_link_libraries(  test2 ${OpenCV_LIBS}  )
