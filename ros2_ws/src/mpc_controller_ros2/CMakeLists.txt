cmake_minimum_required(VERSION 3.8)
project(mpc_controller_ros2)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -march=native for Release/RelWithDebInfo (SIMD 자동 벡터화)
if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
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
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# Try pip-installed pybind11 first (numpy 2.x compatible), fall back to vendor
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_PIP_FOUND
)
if(PYBIND11_PIP_FOUND EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
  find_package(pybind11 REQUIRED CONFIG)
else()
  find_package(pybind11_vendor REQUIRED)
  find_package(pybind11 REQUIRED)
endif()

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
  src/smooth_mppi_controller_plugin.cpp
  src/spline_mppi_controller_plugin.cpp
  src/svg_mppi_controller_plugin.cpp
  src/biased_mppi_controller_plugin.cpp
  src/dial_mppi_controller_plugin.cpp
  src/weight_computation.cpp
  src/batch_dynamics_wrapper.cpp
  src/cost_functions.cpp
  src/sampling.cpp
  src/adaptive_temperature.cpp
  src/ancillary_controller.cpp
  src/tube_mppi.cpp
  src/motion_model.cpp
  src/diff_drive_model.cpp
  src/swerve_drive_model.cpp
  src/non_coaxial_swerve_model.cpp
  src/ackermann_model.cpp
  src/motion_model_factory.cpp
  src/barrier_function.cpp
  src/cbf_safety_filter.cpp
  src/savitzky_golay_filter.cpp
  src/eigen_mlp.cpp
  src/residual_dynamics_model.cpp
  src/conformal_predictor.cpp
  src/shield_mppi_controller_plugin.cpp
  src/ilqr_solver.cpp
  src/ilqr_mppi_controller_plugin.cpp
  src/cs_mppi_controller_plugin.cpp
  src/pi_mppi_controller_plugin.cpp
  src/wheel_level_4d_model.cpp
  src/hybrid_swerve_mppi_controller_plugin.cpp
  src/ensemble_dynamics_model.cpp
  src/online_data_buffer.cpp
  src/c3bf_barrier.cpp
  src/adaptive_shield_mppi_controller_plugin.cpp
  src/clf_function.cpp
  src/clf_cbf_qp_solver.cpp
  src/clf_cbf_mppi_controller_plugin.cpp
  src/online_data_collector.cpp
  src/model_reloader.cpp
  src/predictive_safety_filter.cpp
  src/predictive_safety_mppi_controller_plugin.cpp
  src/tube_mppi_controller_plugin.cpp
  src/dynamic_obstacle_tracker.cpp
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

# pybind11 Python bindings
pybind11_add_module(mppi_py src/bindings/mppi_bindings.cpp)
target_link_libraries(mppi_py PRIVATE mppi_controller_plugin)
target_include_directories(mppi_py PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
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
  scripts/swerve_kinematics_node.py
  scripts/swerve_e2e_test.py
  scripts/stress_test.py
  scripts/nav2_lifecycle_bringup.py
  scripts/train_residual_model.py
  scripts/pi_mppi_e2e_test.py
  scripts/controller_benchmark.py
  scripts/benchmark_report.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install pybind11 module into Python package
install(TARGETS mppi_py
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}/mppi"
)
install(FILES mpc_controller_ros2/mppi/__init__.py
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}/mppi"
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
  ament_add_pytest_test(test_pybind_core test/python/test_pybind_core.py)
  ament_add_pytest_test(test_pybind_costs test/python/test_pybind_costs.py)
  ament_add_pytest_test(test_pybind_advanced test/python/test_pybind_advanced.py)

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

  ament_add_gtest(test_m35_plugins test/unit/test_m35_plugins.cpp)
  if(TARGET test_m35_plugins)
    target_link_libraries(test_m35_plugins mppi_controller_plugin)
  endif()

  ament_add_gtest(test_motion_model test/unit/test_motion_model.cpp)
  if(TARGET test_motion_model)
    target_link_libraries(test_motion_model mppi_controller_plugin)
  endif()

  ament_add_gtest(test_cbf test/unit/test_cbf.cpp)
  if(TARGET test_cbf)
    target_link_libraries(test_cbf mppi_controller_plugin)
  endif()

  ament_add_gtest(test_trajectory_stability test/unit/test_trajectory_stability.cpp)
  if(TARGET test_trajectory_stability)
    target_link_libraries(test_trajectory_stability mppi_controller_plugin)
  endif()

  ament_add_gtest(test_biased_mppi test/unit/test_biased_mppi.cpp)
  if(TARGET test_biased_mppi)
    target_link_libraries(test_biased_mppi mppi_controller_plugin)
  endif()

  ament_add_gtest(test_dial_mppi test/unit/test_dial_mppi.cpp)
  if(TARGET test_dial_mppi)
    target_link_libraries(test_dial_mppi mppi_controller_plugin)
  endif()

  ament_add_gtest(test_residual_dynamics test/unit/test_residual_dynamics.cpp)
  if(TARGET test_residual_dynamics)
    target_link_libraries(test_residual_dynamics mppi_controller_plugin)
  endif()

  ament_add_gtest(test_safety_enhancement test/unit/test_safety_enhancement.cpp)
  if(TARGET test_safety_enhancement)
    target_link_libraries(test_safety_enhancement mppi_controller_plugin)
  endif()

  ament_add_gtest(test_ilqr_solver test/unit/test_ilqr_solver.cpp)
  if(TARGET test_ilqr_solver)
    target_link_libraries(test_ilqr_solver mppi_controller_plugin)
  endif()

  ament_add_gtest(test_ilqr_mppi test/unit/test_ilqr_mppi.cpp)
  if(TARGET test_ilqr_mppi)
    target_link_libraries(test_ilqr_mppi mppi_controller_plugin)
  endif()

  ament_add_gtest(test_cs_mppi test/unit/test_cs_mppi.cpp)
  if(TARGET test_cs_mppi)
    target_link_libraries(test_cs_mppi mppi_controller_plugin)
  endif()

  ament_add_gtest(test_pi_mppi test/unit/test_pi_mppi.cpp)
  if(TARGET test_pi_mppi)
    target_link_libraries(test_pi_mppi mppi_controller_plugin)
  endif()

  ament_add_gtest(test_hybrid_swerve_mppi test/unit/test_hybrid_swerve_mppi.cpp)
  if(TARGET test_hybrid_swerve_mppi)
    target_link_libraries(test_hybrid_swerve_mppi mppi_controller_plugin)
  endif()

  ament_add_gtest(test_ensemble_dynamics test/unit/test_ensemble_dynamics.cpp)
  if(TARGET test_ensemble_dynamics)
    target_link_libraries(test_ensemble_dynamics mppi_controller_plugin)
  endif()

  ament_add_gtest(test_advanced_cbf test/unit/test_advanced_cbf.cpp)
  if(TARGET test_advanced_cbf)
    target_link_libraries(test_advanced_cbf mppi_controller_plugin)
  endif()

  ament_add_gtest(test_clf_cbf_qp test/unit/test_clf_cbf_qp.cpp)
  if(TARGET test_clf_cbf_qp)
    target_link_libraries(test_clf_cbf_qp mppi_controller_plugin)
  endif()

  ament_add_gtest(test_cbf_composition test/unit/test_cbf_composition.cpp)
  if(TARGET test_cbf_composition)
    target_link_libraries(test_cbf_composition mppi_controller_plugin)
  endif()

  ament_add_gtest(test_online_learning test/unit/test_online_learning.cpp)
  if(TARGET test_online_learning)
    target_link_libraries(test_online_learning mppi_controller_plugin)
  endif()

  ament_add_gtest(test_predictive_safety test/unit/test_predictive_safety.cpp)
  if(TARGET test_predictive_safety)
    target_link_libraries(test_predictive_safety mppi_controller_plugin)
  endif()

  ament_add_gtest(test_tube_mppi_plugin test/unit/test_tube_mppi_plugin.cpp)
  if(TARGET test_tube_mppi_plugin)
    target_link_libraries(test_tube_mppi_plugin mppi_controller_plugin)
  endif()

  ament_add_gtest(test_dynamic_obstacle_tracker test/unit/test_dynamic_obstacle_tracker.cpp)
  if(TARGET test_dynamic_obstacle_tracker)
    target_link_libraries(test_dynamic_obstacle_tracker mppi_controller_plugin)
  endif()
endif()

# Benchmark executable (항상 빌드 가능 — ROS2 의존성 없음)
add_executable(bench_mppi_pipeline test/benchmark/bench_mppi_pipeline.cpp)
target_link_libraries(bench_mppi_pipeline mppi_controller_plugin)
target_include_directories(bench_mppi_pipeline PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(bench_mppi_pipeline OpenMP::OpenMP_CXX)
endif()
# Release 최적화 강제 적용
target_compile_options(bench_mppi_pipeline PRIVATE -O2)
install(TARGETS bench_mppi_pipeline RUNTIME DESTINATION lib/${PROJECT_NAME})

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
