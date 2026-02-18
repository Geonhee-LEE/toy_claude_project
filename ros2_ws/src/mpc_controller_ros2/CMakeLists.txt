cmake_minimum_required(VERSION 3.8)
project(mpc_controller_ros2)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# C++ dependencies for MPPI plugin
find_package(rclcpp REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# C++ library: MPPI controller utilities
add_library(mppi_utils SHARED
  src/utils.cpp
)

target_include_directories(mppi_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(mppi_utils
  rclcpp
  geometry_msgs
  tf2
  tf2_geometry_msgs
)

# C++ library: MPPI controller plugin (M2 features included)
add_library(mppi_controller_plugin SHARED
  src/mppi_controller_plugin.cpp
  src/log_mppi_controller_plugin.cpp
  src/tsallis_mppi_controller_plugin.cpp
  src/risk_aware_mppi_controller_plugin.cpp
  src/svmpc_controller_plugin.cpp
  src/weight_computation.cpp
  src/batch_dynamics_wrapper.cpp
  src/cost_functions.cpp
  src/sampling.cpp
  src/adaptive_temperature.cpp
  src/ancillary_controller.cpp
  src/tube_mppi.cpp
)

target_include_directories(mppi_controller_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(mppi_controller_plugin
  mppi_utils
)

ament_target_dependencies(mppi_controller_plugin
  rclcpp
  nav2_core
  nav2_costmap_2d
  nav2_util
  tf2
  tf2_ros
  tf2_geometry_msgs
  pluginlib
  geometry_msgs
  nav_msgs
)

# OpenMP support (optional)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mppi_controller_plugin OpenMP::OpenMP_CXX)
endif()

# Export plugin description
pluginlib_export_plugin_description_file(nav2_core plugins/mppi_controller_plugin.xml)

# Install C++ libraries
install(TARGETS mppi_utils mppi_controller_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install C++ headers
install(DIRECTORY include/
  DESTINATION include/
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/mpc_controller_node
  scripts/send_goal.py
  scripts/send_nav_goal.py
  scripts/test_robot_motion.py
  scripts/verify_robot_motion.py
  scripts/odom_to_tf.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install Gazebo worlds
install(DIRECTORY
  worlds
  DESTINATION share/${PROJECT_NAME}/
)

# Install robot URDF files
install(DIRECTORY
  urdf
  DESTINATION share/${PROJECT_NAME}/
)

# Install Gazebo models
install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}/
)

# Install maps
install(DIRECTORY
  maps
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  # the following line skips the linter which checks for copyrights
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  # Add pytest tests
  ament_add_pytest_test(test_mpc_controller_node test/test_mpc_controller_node.py)

  # Add C++ unit tests
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_batch_dynamics test/unit/test_batch_dynamics.cpp)
  if(TARGET test_batch_dynamics)
    target_link_libraries(test_batch_dynamics mppi_controller_plugin)
  endif()

  ament_add_gtest(test_cost_functions test/unit/test_cost_functions.cpp)
  if(TARGET test_cost_functions)
    target_link_libraries(test_cost_functions mppi_controller_plugin)
  endif()

  ament_add_gtest(test_sampling test/unit/test_sampling.cpp)
  if(TARGET test_sampling)
    target_link_libraries(test_sampling mppi_controller_plugin)
  endif()

  ament_add_gtest(test_mppi_algorithm test/unit/test_mppi_algorithm.cpp)
  if(TARGET test_mppi_algorithm)
    target_link_libraries(test_mppi_algorithm mppi_controller_plugin)
  endif()

  ament_add_gtest(test_adaptive_temperature test/unit/test_adaptive_temperature.cpp)
  if(TARGET test_adaptive_temperature)
    target_link_libraries(test_adaptive_temperature mppi_controller_plugin)
  endif()

  ament_add_gtest(test_tube_mppi test/unit/test_tube_mppi.cpp)
  if(TARGET test_tube_mppi)
    target_link_libraries(test_tube_mppi mppi_controller_plugin)
  endif()

  ament_add_gtest(test_weight_computation test/unit/test_weight_computation.cpp)
  if(TARGET test_weight_computation)
    target_link_libraries(test_weight_computation mppi_controller_plugin)
  endif()

  ament_add_gtest(test_svmpc test/unit/test_svmpc.cpp)
  if(TARGET test_svmpc)
    target_link_libraries(test_svmpc mppi_controller_plugin)
  endif()
endif()

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(mppi_utils mppi_controller_plugin)
ament_export_dependencies(
  rclcpp
  nav2_core
  nav2_costmap_2d
  nav2_util
  tf2
  tf2_ros
  tf2_geometry_msgs
  pluginlib
  geometry_msgs
  nav_msgs
)

ament_package()
