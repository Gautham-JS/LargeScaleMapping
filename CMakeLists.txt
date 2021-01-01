cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project(LargeScaleMapping)

SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(
	VisualOdometry ${PROJECT_SOURCE_DIR}/visualOdometry/VisualOdometry.cpp
)

target_link_libraries(
	VisualOdometry ${OpenCV_LIBS}
)


