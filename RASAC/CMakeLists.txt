cmake_minimum_required(VERSION 3.5)
project(vbtest)


#查找 PCL 库
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
# 查找 Eigen3 库
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(/usr/include/pcl-1.10/pcl)
include_directories(/usr/include/pcl-1.10/pcl/sample_consensus/)

add_executable(test test.cpp)

# 链接 PCL、Eigen3 和 VTK 库
target_link_libraries(test ${PCL_LIBRARIES} ${EIGEN3_LIBRARIES})
