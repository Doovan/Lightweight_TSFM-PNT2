cmake_minimum_required(VERSION 3.0)
project(PointCloudProcessing)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(PointCloudProcessing main.cpp)
target_link_libraries(PointCloudProcessing ${PCL_LIBRARIES})
