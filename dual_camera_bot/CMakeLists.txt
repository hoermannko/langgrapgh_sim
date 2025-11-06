cmake_minimum_required(VERSION 3.8)
project(dual_camera_bot)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  description
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
