cmake_minimum_required(VERSION 2.8.3)
project(charuco_calibration)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
set(catkin_RUN_PACKAGES
    cv_bridge
    image_transport
    roscpp
    )
# set catkin packages that are needed when building the code
set(catkin_BUILD_PACKAGES
    ${catkin_RUN_PACKAGES}
    cmake_modules)

find_package(catkin REQUIRED
  COMPONENTS
  ${catkin_BUILD_PACKAGES}
)
find_package(OpenCV REQUIRED NO_MODULE PATHS $ENV{OpenCV_PATH} NO_DEFAULT_PATH)

message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

catkin_package(
)

include_directories(
  common/include
  ${catkin_INCLUDE_DIRS}
)


file(GLOB_RECURSE CPP_ALL_HEADER_FILES "${CMAKE_SOURCE_DIR}/common/include/*.hpp")
file(GLOB_RECURSE CPP_ALL_SOURCE_FILES "${CMAKE_SOURCE_DIR}/common/src/*.cpp")

add_library(${PROJECT_NAME}
    ${CPP_ALL_HEADER_FILES}
    ${CPP_ALL_SOURCE_FILES}
    )

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS}
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)


add_executable(${PROJECT_NAME}_node 
  src/calibrate_camera_ros.cpp
)

target_link_libraries(${PROJECT_NAME}_node PRIVATE
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

add_executable(calibrate_camera_from_files
  src/calibrate_camera_from_files.cpp
)

target_link_libraries(calibrate_camera_from_files PRIVATE
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)


add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(calibrate_camera_from_files ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


#######

#add_executable(camera_pose_node
#  src/camera_pose_node.cpp
#)

#target_link_libraries(camera_pose_node
#  ${PROJECT_NAME}
#  ${catkin_LIBRARIES}
#)


#add_dependencies(camera_pose_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

