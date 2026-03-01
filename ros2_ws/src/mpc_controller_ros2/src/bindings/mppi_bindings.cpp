/**
 * @file mppi_bindings.cpp
 * @brief pybind11 Python bindings for C++ MPPI controller core
 *
 * Exposes the complete MPPI pipeline to Python:
 *   from mpc_controller_ros2.mppi import MPPIParams, DiffDriveModel, ...
 *
 * Design decisions:
 *   - Eigen <-> NumPy: automatic via pybind11/eigen.h
 *   - vector<MatrixXd> -> list[np.ndarray] via pybind11/stl.h
 *   - ROS2 messages excluded (controlToTwist, twistToControl, quaternionToYaw)
 *   - unique_ptr from factory: py::return_value_policy::move
 *   - "lambda" -> "lambda_" (Python reserved word)
 *   - CompositeMPPICost: convenience add_*() wrappers instead of raw addCost(unique_ptr)
 *   - CBFSafetyFilter: BarrierFunctionSet managed via shared_ptr for lifetime safety
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <memory>
#include <string>
#include <vector>

#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/motion_model.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/swerve_drive_model.hpp"
#include "mpc_controller_ros2/non_coaxial_swerve_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/adaptive_temperature.hpp"
#include "mpc_controller_ros2/tube_mppi.hpp"
#include "mpc_controller_ros2/ancillary_controller.hpp"
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/cbf_safety_filter.hpp"
#include "mpc_controller_ros2/savitzky_golay_filter.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace py = pybind11;
using namespace mpc_controller_ros2;

// ============================================================================
// MotionModel trampoline for polymorphic dispatch
// ============================================================================
class PyMotionModel : public MotionModel
{
public:
  using MotionModel::MotionModel;

  int stateDim() const override {
    PYBIND11_OVERRIDE_PURE(int, MotionModel, stateDim);
  }
  int controlDim() const override {
    PYBIND11_OVERRIDE_PURE(int, MotionModel, controlDim);
  }
  bool isHolonomic() const override {
    PYBIND11_OVERRIDE_PURE(bool, MotionModel, isHolonomic);
  }
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, MotionModel, name);
  }
  Eigen::MatrixXd dynamicsBatch(
    const Eigen::MatrixXd& states, const Eigen::MatrixXd& controls) const override
  {
    PYBIND11_OVERRIDE_PURE(Eigen::MatrixXd, MotionModel, dynamicsBatch, states, controls);
  }
  Eigen::MatrixXd clipControls(const Eigen::MatrixXd& controls) const override {
    PYBIND11_OVERRIDE_PURE(Eigen::MatrixXd, MotionModel, clipControls, controls);
  }
  void normalizeStates(Eigen::MatrixXd& states) const override {
    PYBIND11_OVERRIDE_PURE(void, MotionModel, normalizeStates, states);
  }
  geometry_msgs::msg::Twist controlToTwist(const Eigen::VectorXd& control) const override {
    // Not bound to Python — throw if called
    throw std::runtime_error("controlToTwist is not available in Python bindings");
  }
  Eigen::VectorXd twistToControl(const geometry_msgs::msg::Twist& twist) const override {
    throw std::runtime_error("twistToControl is not available in Python bindings");
  }
  std::vector<int> angleIndices() const override {
    PYBIND11_OVERRIDE_PURE(std::vector<int>, MotionModel, angleIndices);
  }
  Eigen::MatrixXd propagateBatch(
    const Eigen::MatrixXd& states, const Eigen::MatrixXd& controls, double dt) const override
  {
    PYBIND11_OVERRIDE(Eigen::MatrixXd, MotionModel, propagateBatch, states, controls, dt);
  }
  std::vector<Eigen::MatrixXd> rolloutBatch(
    const Eigen::VectorXd& x0,
    const std::vector<Eigen::MatrixXd>& control_sequences,
    double dt) const override
  {
    PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, MotionModel, rolloutBatch,
                      x0, control_sequences, dt);
  }
};

// ============================================================================
// Module definition
// ============================================================================
PYBIND11_MODULE(mppi_py, m)
{
  m.doc() = "C++ MPPI controller core — Python bindings via pybind11";

  // --------------------------------------------------------------------------
  // MPPIParams
  // --------------------------------------------------------------------------
  py::class_<MPPIParams>(m, "MPPIParams")
    .def(py::init<>())
    // Basic MPPI
    .def_readwrite("N", &MPPIParams::N)
    .def_readwrite("dt", &MPPIParams::dt)
    .def_readwrite("K", &MPPIParams::K)
    .def_readwrite("lambda_", &MPPIParams::lambda)
    .def_property("noise_sigma",
      [](const MPPIParams& p) -> Eigen::VectorXd { return p.noise_sigma; },
      [](MPPIParams& p, py::array_t<double> arr) {
        auto buf = arr.request();
        p.noise_sigma = Eigen::Map<const Eigen::VectorXd>(
          static_cast<const double*>(buf.ptr), buf.shape[0]);
      })
    .def_readwrite("exploration_ratio", &MPPIParams::exploration_ratio)
    .def_property("Q",
      [](const MPPIParams& p) -> Eigen::MatrixXd { return p.Q; },
      [](MPPIParams& p, py::array_t<double, py::array::c_style> arr) {
        auto buf = arr.request();
        p.Q = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          static_cast<const double*>(buf.ptr), buf.shape[0], buf.shape[1]);
      })
    .def_property("Qf",
      [](const MPPIParams& p) -> Eigen::MatrixXd { return p.Qf; },
      [](MPPIParams& p, py::array_t<double, py::array::c_style> arr) {
        auto buf = arr.request();
        p.Qf = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          static_cast<const double*>(buf.ptr), buf.shape[0], buf.shape[1]);
      })
    .def_property("R",
      [](const MPPIParams& p) -> Eigen::MatrixXd { return p.R; },
      [](MPPIParams& p, py::array_t<double, py::array::c_style> arr) {
        auto buf = arr.request();
        p.R = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          static_cast<const double*>(buf.ptr), buf.shape[0], buf.shape[1]);
      })
    .def_property("R_rate",
      [](const MPPIParams& p) -> Eigen::MatrixXd { return p.R_rate; },
      [](MPPIParams& p, py::array_t<double, py::array::c_style> arr) {
        auto buf = arr.request();
        p.R_rate = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          static_cast<const double*>(buf.ptr), buf.shape[0], buf.shape[1]);
      })
    // Control limits
    .def_readwrite("v_max", &MPPIParams::v_max)
    .def_readwrite("v_min", &MPPIParams::v_min)
    .def_readwrite("omega_max", &MPPIParams::omega_max)
    .def_readwrite("omega_min", &MPPIParams::omega_min)
    .def_readwrite("vy_max", &MPPIParams::vy_max)
    // Obstacle
    .def_readwrite("obstacle_weight", &MPPIParams::obstacle_weight)
    .def_readwrite("safety_distance", &MPPIParams::safety_distance)
    // Forward preference
    .def_readwrite("prefer_forward_weight", &MPPIParams::prefer_forward_weight)
    .def_readwrite("prefer_forward_linear_ratio", &MPPIParams::prefer_forward_linear_ratio)
    .def_readwrite("prefer_forward_velocity_incentive", &MPPIParams::prefer_forward_velocity_incentive)
    // Smoothing
    .def_readwrite("control_smoothing_alpha", &MPPIParams::control_smoothing_alpha)
    .def_readwrite("sg_filter_enabled", &MPPIParams::sg_filter_enabled)
    .def_readwrite("sg_half_window", &MPPIParams::sg_half_window)
    .def_readwrite("sg_poly_order", &MPPIParams::sg_poly_order)
    .def_readwrite("it_alpha", &MPPIParams::it_alpha)
    // Colored noise
    .def_readwrite("colored_noise", &MPPIParams::colored_noise)
    .def_readwrite("noise_beta", &MPPIParams::noise_beta)
    // Adaptive temperature
    .def_readwrite("adaptive_temperature", &MPPIParams::adaptive_temperature)
    .def_readwrite("target_ess_ratio", &MPPIParams::target_ess_ratio)
    .def_readwrite("adaptation_rate", &MPPIParams::adaptation_rate)
    .def_readwrite("lambda_min", &MPPIParams::lambda_min)
    .def_readwrite("lambda_max", &MPPIParams::lambda_max)
    // Tube-MPPI
    .def_readwrite("tube_enabled", &MPPIParams::tube_enabled)
    .def_readwrite("tube_width", &MPPIParams::tube_width)
    .def_readwrite("k_forward", &MPPIParams::k_forward)
    .def_readwrite("k_lateral", &MPPIParams::k_lateral)
    .def_readwrite("k_angle", &MPPIParams::k_angle)
    // SOTA variants
    .def_readwrite("tsallis_q", &MPPIParams::tsallis_q)
    .def_readwrite("cvar_alpha", &MPPIParams::cvar_alpha)
    .def_readwrite("svgd_num_iterations", &MPPIParams::svgd_num_iterations)
    .def_readwrite("svgd_step_size", &MPPIParams::svgd_step_size)
    .def_readwrite("svgd_bandwidth", &MPPIParams::svgd_bandwidth)
    // M3.5
    .def_readwrite("smooth_R_jerk_v", &MPPIParams::smooth_R_jerk_v)
    .def_readwrite("smooth_R_jerk_omega", &MPPIParams::smooth_R_jerk_omega)
    .def_readwrite("smooth_action_cost_weight", &MPPIParams::smooth_action_cost_weight)
    .def_readwrite("spline_num_knots", &MPPIParams::spline_num_knots)
    .def_readwrite("spline_degree", &MPPIParams::spline_degree)
    .def_readwrite("spline_auto_knot_sigma", &MPPIParams::spline_auto_knot_sigma)
    .def_readwrite("svg_num_guide_particles", &MPPIParams::svg_num_guide_particles)
    .def_readwrite("svg_guide_iterations", &MPPIParams::svg_guide_iterations)
    .def_readwrite("svg_guide_step_size", &MPPIParams::svg_guide_step_size)
    .def_readwrite("svg_resample_std", &MPPIParams::svg_resample_std)
    // NonCoaxial
    .def_readwrite("max_steering_rate", &MPPIParams::max_steering_rate)
    .def_readwrite("max_steering_angle", &MPPIParams::max_steering_angle)
    // CBF
    .def_readwrite("cbf_enabled", &MPPIParams::cbf_enabled)
    .def_readwrite("cbf_gamma", &MPPIParams::cbf_gamma)
    .def_readwrite("cbf_safety_margin", &MPPIParams::cbf_safety_margin)
    .def_readwrite("cbf_robot_radius", &MPPIParams::cbf_robot_radius)
    .def_readwrite("cbf_activation_distance", &MPPIParams::cbf_activation_distance)
    .def_readwrite("cbf_cost_weight", &MPPIParams::cbf_cost_weight)
    .def_readwrite("cbf_use_safety_filter", &MPPIParams::cbf_use_safety_filter)
    // Motion model
    .def_readwrite("motion_model", &MPPIParams::motion_model)
    // Costmap
    .def_readwrite("use_costmap_cost", &MPPIParams::use_costmap_cost)
    .def_readwrite("costmap_lethal_cost", &MPPIParams::costmap_lethal_cost)
    .def_readwrite("costmap_critical_cost", &MPPIParams::costmap_critical_cost)
    .def_readwrite("lookahead_dist", &MPPIParams::lookahead_dist)
    .def_readwrite("min_lookahead", &MPPIParams::min_lookahead)
    .def_readwrite("goal_slowdown_dist", &MPPIParams::goal_slowdown_dist)
    .def_readwrite("ref_theta_smooth_window", &MPPIParams::ref_theta_smooth_window)
    // Velocity tracking
    .def_readwrite("velocity_tracking_weight", &MPPIParams::velocity_tracking_weight)
    .def_readwrite("reference_velocity", &MPPIParams::reference_velocity)
    // Debug/Viz
    .def_readwrite("debug_collision_viz", &MPPIParams::debug_collision_viz)
    .def_readwrite("debug_cost_breakdown", &MPPIParams::debug_cost_breakdown)
    .def_readwrite("debug_collision_points", &MPPIParams::debug_collision_points)
    .def_readwrite("debug_safety_footprint", &MPPIParams::debug_safety_footprint)
    .def_readwrite("debug_cost_heatmap", &MPPIParams::debug_cost_heatmap)
    .def_readwrite("debug_footprint_radius", &MPPIParams::debug_footprint_radius)
    .def_readwrite("debug_heatmap_stride", &MPPIParams::debug_heatmap_stride)
    .def_readwrite("visualize_samples", &MPPIParams::visualize_samples)
    .def_readwrite("visualize_best", &MPPIParams::visualize_best)
    .def_readwrite("visualize_weighted_avg", &MPPIParams::visualize_weighted_avg)
    .def_readwrite("visualize_reference", &MPPIParams::visualize_reference)
    .def_readwrite("visualize_text_info", &MPPIParams::visualize_text_info)
    .def_readwrite("visualize_control_sequence", &MPPIParams::visualize_control_sequence)
    .def_readwrite("visualize_tube", &MPPIParams::visualize_tube)
    .def_readwrite("visualize_cbf", &MPPIParams::visualize_cbf)
    .def_readwrite("max_visualized_samples", &MPPIParams::max_visualized_samples)
    // Methods
    .def("getFeedbackGainMatrix", &MPPIParams::getFeedbackGainMatrix)
    .def("__repr__", [](const MPPIParams& p) {
      return "<MPPIParams N=" + std::to_string(p.N)
           + " K=" + std::to_string(p.K)
           + " dt=" + std::to_string(p.dt)
           + " lambda=" + std::to_string(p.lambda)
           + " model='" + p.motion_model + "'>";
    });

  // --------------------------------------------------------------------------
  // MotionModel (abstract base)
  // --------------------------------------------------------------------------
  py::class_<MotionModel, PyMotionModel, std::shared_ptr<MotionModel>>(m, "MotionModel")
    .def(py::init<>())
    .def("stateDim", &MotionModel::stateDim)
    .def("controlDim", &MotionModel::controlDim)
    .def("isHolonomic", &MotionModel::isHolonomic)
    .def("name", &MotionModel::name)
    .def("dynamicsBatch", &MotionModel::dynamicsBatch,
         py::arg("states"), py::arg("controls"))
    .def("clipControls", &MotionModel::clipControls,
         py::arg("controls"))
    .def("angleIndices", &MotionModel::angleIndices)
    .def("propagateBatch", &MotionModel::propagateBatch,
         py::arg("states"), py::arg("controls"), py::arg("dt"))
    .def("rolloutBatch", &MotionModel::rolloutBatch,
         py::arg("x0"), py::arg("control_sequences"), py::arg("dt"));

  // --------------------------------------------------------------------------
  // DiffDriveModel
  // --------------------------------------------------------------------------
  py::class_<DiffDriveModel, MotionModel, std::shared_ptr<DiffDriveModel>>(m, "DiffDriveModel")
    .def(py::init<double, double, double, double>(),
         py::arg("v_min"), py::arg("v_max"),
         py::arg("omega_min"), py::arg("omega_max"));

  // --------------------------------------------------------------------------
  // SwerveDriveModel
  // --------------------------------------------------------------------------
  py::class_<SwerveDriveModel, MotionModel, std::shared_ptr<SwerveDriveModel>>(m, "SwerveDriveModel")
    .def(py::init<double, double, double, double>(),
         py::arg("vx_min"), py::arg("vx_max"),
         py::arg("vy_max"), py::arg("omega_max"));

  // --------------------------------------------------------------------------
  // NonCoaxialSwerveModel
  // --------------------------------------------------------------------------
  py::class_<NonCoaxialSwerveModel, MotionModel, std::shared_ptr<NonCoaxialSwerveModel>>(
    m, "NonCoaxialSwerveModel")
    .def(py::init<double, double, double, double, double>(),
         py::arg("v_min"), py::arg("v_max"), py::arg("omega_max"),
         py::arg("max_steering_rate"), py::arg("max_steering_angle") = M_PI / 2.0)
    .def("setLastDelta", &NonCoaxialSwerveModel::setLastDelta, py::arg("delta"))
    .def("getLastDelta", &NonCoaxialSwerveModel::getLastDelta);

  // --------------------------------------------------------------------------
  // create_motion_model (factory)
  // --------------------------------------------------------------------------
  m.def("create_motion_model",
    [](const std::string& model_type, const MPPIParams& params) -> std::shared_ptr<MotionModel> {
      return MotionModelFactory::create(model_type, params);
    },
    py::arg("model_type"), py::arg("params"),
    "Create a MotionModel from type string ('diff_drive', 'swerve', 'non_coaxial_swerve')");

  // --------------------------------------------------------------------------
  // BatchDynamicsWrapper
  // --------------------------------------------------------------------------
  py::class_<BatchDynamicsWrapper>(m, "BatchDynamicsWrapper")
    .def(py::init<const MPPIParams&>(), py::arg("params"))
    .def(py::init<const MPPIParams&, std::shared_ptr<MotionModel>>(),
         py::arg("params"), py::arg("model"))
    .def("dynamicsBatch", &BatchDynamicsWrapper::dynamicsBatch,
         py::arg("states"), py::arg("controls"))
    .def("propagateBatch", &BatchDynamicsWrapper::propagateBatch,
         py::arg("states"), py::arg("controls"), py::arg("dt"))
    .def("rolloutBatch", &BatchDynamicsWrapper::rolloutBatch,
         py::arg("x0"), py::arg("control_sequences"), py::arg("dt"))
    .def("clipControls", &BatchDynamicsWrapper::clipControls, py::arg("controls"))
    .def("model", static_cast<const MotionModel& (BatchDynamicsWrapper::*)() const>(
         &BatchDynamicsWrapper::model),
         py::return_value_policy::reference_internal);

  // --------------------------------------------------------------------------
  // Samplers
  // --------------------------------------------------------------------------
  py::class_<GaussianSampler>(m, "GaussianSampler")
    .def(py::init<const Eigen::VectorXd&, unsigned int>(),
         py::arg("sigma"), py::arg("seed") = 42)
    .def("sample", &GaussianSampler::sample,
         py::arg("K"), py::arg("N"), py::arg("nu"))
    .def("resetSeed", &GaussianSampler::resetSeed, py::arg("seed"));

  py::class_<ColoredNoiseSampler>(m, "ColoredNoiseSampler")
    .def(py::init<const Eigen::VectorXd&, double, unsigned int>(),
         py::arg("sigma"), py::arg("beta") = 2.0, py::arg("seed") = 42)
    .def("sample", &ColoredNoiseSampler::sample,
         py::arg("K"), py::arg("N"), py::arg("nu"))
    .def("resetSeed", &ColoredNoiseSampler::resetSeed, py::arg("seed"));

  // --------------------------------------------------------------------------
  // CostBreakdown
  // --------------------------------------------------------------------------
  py::class_<CostBreakdown>(m, "CostBreakdown")
    .def(py::init<>())
    .def_readwrite("total_costs", &CostBreakdown::total_costs)
    .def_readwrite("component_costs", &CostBreakdown::component_costs);

  // --------------------------------------------------------------------------
  // Cost functions — individual concrete types for direct construction
  // --------------------------------------------------------------------------
  py::class_<StateTrackingCost>(m, "StateTrackingCost")
    .def(py::init<const Eigen::MatrixXd&>(), py::arg("Q"))
    .def("name", &StateTrackingCost::name)
    .def("compute", &StateTrackingCost::compute,
         py::arg("trajectories"), py::arg("controls"), py::arg("reference"));

  py::class_<TerminalCost>(m, "TerminalCost")
    .def(py::init<const Eigen::MatrixXd&>(), py::arg("Qf"))
    .def("name", &TerminalCost::name)
    .def("compute", &TerminalCost::compute,
         py::arg("trajectories"), py::arg("controls"), py::arg("reference"));

  py::class_<ControlEffortCost>(m, "ControlEffortCost")
    .def(py::init<const Eigen::MatrixXd&>(), py::arg("R"))
    .def("name", &ControlEffortCost::name)
    .def("compute", &ControlEffortCost::compute,
         py::arg("trajectories"), py::arg("controls"), py::arg("reference"));

  py::class_<ControlRateCost>(m, "ControlRateCost")
    .def(py::init<const Eigen::MatrixXd&>(), py::arg("R_rate"))
    .def("name", &ControlRateCost::name)
    .def("compute", &ControlRateCost::compute,
         py::arg("trajectories"), py::arg("controls"), py::arg("reference"));

  py::class_<PreferForwardCost>(m, "PreferForwardCost")
    .def(py::init<double, double, double>(),
         py::arg("weight"), py::arg("linear_ratio") = 0.0,
         py::arg("velocity_incentive") = 0.0)
    .def("name", &PreferForwardCost::name)
    .def("compute", &PreferForwardCost::compute,
         py::arg("trajectories"), py::arg("controls"), py::arg("reference"));

  py::class_<ObstacleCost>(m, "ObstacleCost")
    .def(py::init<double, double>(), py::arg("weight"), py::arg("safety_distance"))
    .def("name", &ObstacleCost::name)
    .def("setObstacles", &ObstacleCost::setObstacles, py::arg("obstacles"))
    .def("compute", &ObstacleCost::compute,
         py::arg("trajectories"), py::arg("controls"), py::arg("reference"));

  py::class_<VelocityTrackingCost>(m, "VelocityTrackingCost")
    .def(py::init<double, double, double>(),
         py::arg("weight"), py::arg("reference_velocity"), py::arg("dt"))
    .def("name", &VelocityTrackingCost::name)
    .def("compute", &VelocityTrackingCost::compute,
         py::arg("trajectories"), py::arg("controls"), py::arg("reference"));

  // --------------------------------------------------------------------------
  // CompositeMPPICost — convenience wrappers for addCost(unique_ptr)
  // --------------------------------------------------------------------------
  py::class_<CompositeMPPICost>(m, "CompositeMPPICost")
    .def(py::init<>())
    .def("add_state_tracking", [](CompositeMPPICost& self, const Eigen::MatrixXd& Q) {
      self.addCost(std::make_unique<StateTrackingCost>(Q));
    }, py::arg("Q"))
    .def("add_terminal", [](CompositeMPPICost& self, const Eigen::MatrixXd& Qf) {
      self.addCost(std::make_unique<TerminalCost>(Qf));
    }, py::arg("Qf"))
    .def("add_control_effort", [](CompositeMPPICost& self, const Eigen::MatrixXd& R) {
      self.addCost(std::make_unique<ControlEffortCost>(R));
    }, py::arg("R"))
    .def("add_control_rate", [](CompositeMPPICost& self, const Eigen::MatrixXd& R_rate) {
      self.addCost(std::make_unique<ControlRateCost>(R_rate));
    }, py::arg("R_rate"))
    .def("add_prefer_forward", [](CompositeMPPICost& self, double weight,
                                   double linear_ratio, double velocity_incentive) {
      self.addCost(std::make_unique<PreferForwardCost>(weight, linear_ratio, velocity_incentive));
    }, py::arg("weight"), py::arg("linear_ratio") = 0.0, py::arg("velocity_incentive") = 0.0)
    .def("add_obstacle", [](CompositeMPPICost& self, double weight, double safety_distance) {
      self.addCost(std::make_unique<ObstacleCost>(weight, safety_distance));
    }, py::arg("weight"), py::arg("safety_distance"))
    .def("add_velocity_tracking", [](CompositeMPPICost& self, double weight,
                                      double reference_velocity, double dt) {
      self.addCost(std::make_unique<VelocityTrackingCost>(weight, reference_velocity, dt));
    }, py::arg("weight"), py::arg("reference_velocity"), py::arg("dt"))
    .def("clearCosts", &CompositeMPPICost::clearCosts)
    .def("compute", &CompositeMPPICost::compute,
         py::arg("trajectories"), py::arg("controls"), py::arg("reference"))
    .def("computeDetailed", &CompositeMPPICost::computeDetailed,
         py::arg("trajectories"), py::arg("controls"), py::arg("reference"));

  // --------------------------------------------------------------------------
  // WeightComputation strategies
  // --------------------------------------------------------------------------
  py::class_<WeightComputation>(m, "WeightComputation")
    .def("compute", &WeightComputation::compute, py::arg("costs"), py::arg("lambda_"))
    .def("name", &WeightComputation::name);

  py::class_<VanillaMPPIWeights, WeightComputation>(m, "VanillaMPPIWeights")
    .def(py::init<>());

  py::class_<LogMPPIWeights, WeightComputation>(m, "LogMPPIWeights")
    .def(py::init<>());

  py::class_<TsallisMPPIWeights, WeightComputation>(m, "TsallisMPPIWeights")
    .def(py::init<double>(), py::arg("q") = 1.5);

  py::class_<RiskAwareMPPIWeights, WeightComputation>(m, "RiskAwareMPPIWeights")
    .def(py::init<double>(), py::arg("alpha") = 0.5);

  // --------------------------------------------------------------------------
  // AdaptiveTemperature
  // --------------------------------------------------------------------------
  py::class_<AdaptiveTemperature::AdaptiveInfo>(m, "AdaptiveInfo")
    .def_readonly("lambda_", &AdaptiveTemperature::AdaptiveInfo::lambda)
    .def_readonly("log_lambda", &AdaptiveTemperature::AdaptiveInfo::log_lambda)
    .def_readonly("ess_ratio", &AdaptiveTemperature::AdaptiveInfo::ess_ratio)
    .def_readonly("target_ratio", &AdaptiveTemperature::AdaptiveInfo::target_ratio)
    .def_readonly("delta", &AdaptiveTemperature::AdaptiveInfo::delta);

  py::class_<AdaptiveTemperature>(m, "AdaptiveTemperature")
    .def(py::init<double, double, double, double, double>(),
         py::arg("initial_lambda") = 10.0,
         py::arg("target_ess_ratio") = 0.5,
         py::arg("adaptation_rate") = 0.1,
         py::arg("lambda_min") = 0.1,
         py::arg("lambda_max") = 100.0)
    .def("update", &AdaptiveTemperature::update, py::arg("ess"), py::arg("K"))
    .def("getLambda", &AdaptiveTemperature::getLambda)
    .def("setLambda", &AdaptiveTemperature::setLambda, py::arg("lambda_"))
    .def("setParameters", &AdaptiveTemperature::setParameters,
         py::arg("target_ess_ratio"), py::arg("adaptation_rate"),
         py::arg("lambda_min"), py::arg("lambda_max"))
    .def("reset", &AdaptiveTemperature::reset, py::arg("initial_lambda"))
    .def("getInfo", &AdaptiveTemperature::getInfo);

  // --------------------------------------------------------------------------
  // AncillaryController
  // --------------------------------------------------------------------------
  py::class_<AncillaryController>(m, "AncillaryController")
    .def(py::init<const Eigen::MatrixXd&>(), py::arg("K_fb"))
    .def(py::init<double, double, double>(),
         py::arg("k_forward") = 0.8,
         py::arg("k_lateral") = 0.5,
         py::arg("k_angle") = 1.0)
    .def("computeBodyFrameError", &AncillaryController::computeBodyFrameError,
         py::arg("nominal_state"), py::arg("actual_state"))
    .def("computeCorrectedControl", &AncillaryController::computeCorrectedControl,
         py::arg("nominal_control"), py::arg("nominal_state"), py::arg("actual_state"))
    .def("setGains", static_cast<void (AncillaryController::*)(const Eigen::MatrixXd&)>(
         &AncillaryController::setGains), py::arg("K_fb"))
    .def("setGainsScalar", static_cast<void (AncillaryController::*)(double, double, double)>(
         &AncillaryController::setGains),
         py::arg("k_forward"), py::arg("k_lateral"), py::arg("k_angle"))
    .def("getGains", &AncillaryController::getGains)
    .def("computeFeedbackCorrection", &AncillaryController::computeFeedbackCorrection,
         py::arg("body_error"));

  // --------------------------------------------------------------------------
  // TubeMPPIInfo
  // --------------------------------------------------------------------------
  py::class_<TubeMPPIInfo>(m, "TubeMPPIInfo")
    .def(py::init<>())
    .def_readwrite("nominal_state", &TubeMPPIInfo::nominal_state)
    .def_readwrite("nominal_control", &TubeMPPIInfo::nominal_control)
    .def_readwrite("body_error", &TubeMPPIInfo::body_error)
    .def_readwrite("feedback_correction", &TubeMPPIInfo::feedback_correction)
    .def_readwrite("applied_control", &TubeMPPIInfo::applied_control)
    .def_readwrite("tube_width", &TubeMPPIInfo::tube_width)
    .def_readwrite("tube_boundary", &TubeMPPIInfo::tube_boundary);

  // --------------------------------------------------------------------------
  // TubeMPPI
  // --------------------------------------------------------------------------
  py::class_<TubeMPPI>(m, "TubeMPPI")
    .def(py::init<const MPPIParams&>(), py::arg("params"))
    .def("computeCorrectedControl", &TubeMPPI::computeCorrectedControl,
         py::arg("nominal_control"), py::arg("nominal_trajectory"),
         py::arg("actual_state"))
    .def("updateTubeWidth", &TubeMPPI::updateTubeWidth, py::arg("tracking_error"))
    .def("isInsideTube", &TubeMPPI::isInsideTube,
         py::arg("nominal_state"), py::arg("actual_state"))
    .def("setFeedbackGains", &TubeMPPI::setFeedbackGains,
         py::arg("k_forward"), py::arg("k_lateral"), py::arg("k_angle"))
    .def("setTubeWidth", &TubeMPPI::setTubeWidth, py::arg("width"))
    .def("getTubeWidth", &TubeMPPI::getTubeWidth)
    .def("updateParams", &TubeMPPI::updateParams, py::arg("params"));

  // --------------------------------------------------------------------------
  // CircleBarrier
  // --------------------------------------------------------------------------
  py::class_<CircleBarrier>(m, "CircleBarrier")
    .def(py::init<double, double, double, double, double>(),
         py::arg("obs_x"), py::arg("obs_y"), py::arg("obs_radius"),
         py::arg("robot_radius"), py::arg("safety_margin"))
    .def("evaluate", &CircleBarrier::evaluate, py::arg("state"))
    .def("evaluateBatch", &CircleBarrier::evaluateBatch, py::arg("states"))
    .def("gradient", &CircleBarrier::gradient, py::arg("state"))
    .def("obsX", &CircleBarrier::obsX)
    .def("obsY", &CircleBarrier::obsY)
    .def("safeDistance", &CircleBarrier::safeDistance);

  // --------------------------------------------------------------------------
  // BarrierFunctionSet
  // --------------------------------------------------------------------------
  py::class_<BarrierFunctionSet, std::shared_ptr<BarrierFunctionSet>>(m, "BarrierFunctionSet")
    .def(py::init<double, double, double>(),
         py::arg("robot_radius") = 0.2,
         py::arg("safety_margin") = 0.3,
         py::arg("activation_distance") = 3.0)
    .def("setObstacles", &BarrierFunctionSet::setObstacles, py::arg("obstacles"))
    .def("getActiveBarriers", [](const BarrierFunctionSet& self, const Eigen::VectorXd& state) {
      auto ptrs = self.getActiveBarriers(state);
      return static_cast<int>(ptrs.size());
    }, py::arg("state"), "Returns number of active barriers at given state")
    .def("evaluateAll", &BarrierFunctionSet::evaluateAll, py::arg("state"))
    .def("size", &BarrierFunctionSet::size)
    .def("empty", &BarrierFunctionSet::empty);

  // --------------------------------------------------------------------------
  // CBFFilterInfo
  // --------------------------------------------------------------------------
  py::class_<CBFFilterInfo>(m, "CBFFilterInfo")
    .def(py::init<>())
    .def_readwrite("num_active_barriers", &CBFFilterInfo::num_active_barriers)
    .def_readwrite("filter_applied", &CBFFilterInfo::filter_applied)
    .def_readwrite("qp_success", &CBFFilterInfo::qp_success)
    .def_readwrite("barrier_values", &CBFFilterInfo::barrier_values)
    .def_readwrite("constraint_margins", &CBFFilterInfo::constraint_margins)
    .def_readwrite("u_mppi", &CBFFilterInfo::u_mppi)
    .def_readwrite("u_safe", &CBFFilterInfo::u_safe);

  // --------------------------------------------------------------------------
  // CBFSafetyFilter
  // --------------------------------------------------------------------------
  py::class_<CBFSafetyFilter>(m, "CBFSafetyFilter")
    .def(py::init([](std::shared_ptr<BarrierFunctionSet> barrier_set,
                     double gamma, double dt,
                     const Eigen::VectorXd& u_min,
                     const Eigen::VectorXd& u_max) {
      return std::make_unique<CBFSafetyFilter>(
        barrier_set.get(), gamma, dt, u_min, u_max);
    }),
    py::arg("barrier_set"), py::arg("gamma"), py::arg("dt"),
    py::arg("u_min"), py::arg("u_max"),
    py::keep_alive<1, 2>())  // barrier_set kept alive by filter
    .def("filter", &CBFSafetyFilter::filter,
         py::arg("state"), py::arg("u_mppi"), py::arg("dynamics"));

  // --------------------------------------------------------------------------
  // SavitzkyGolayFilter
  // --------------------------------------------------------------------------
  py::class_<SavitzkyGolayFilter>(m, "SavitzkyGolayFilter")
    .def(py::init<int, int, int>(),
         py::arg("half_window"), py::arg("poly_order"), py::arg("nu"))
    .def("apply", &SavitzkyGolayFilter::apply,
         py::arg("control_sequence"), py::arg("current_step") = 0)
    .def("pushHistory", &SavitzkyGolayFilter::pushHistory, py::arg("control"))
    .def("reset", &SavitzkyGolayFilter::reset)
    .def("coefficients", &SavitzkyGolayFilter::coefficients)
    .def("windowSize", &SavitzkyGolayFilter::windowSize)
    .def("halfWindow", &SavitzkyGolayFilter::halfWindow);

  // --------------------------------------------------------------------------
  // Utility functions
  // --------------------------------------------------------------------------
  m.def("normalize_angle", &normalizeAngle, py::arg("angle"));
  m.def("normalize_angle_batch", &normalizeAngleBatch, py::arg("angles"));
  m.def("softmax_weights", &softmaxWeights, py::arg("costs"), py::arg("lambda_"));
  m.def("log_sum_exp", &logSumExp, py::arg("values"));
  m.def("compute_ess", &computeESS, py::arg("weights"));
  m.def("euclidean_distance_2d", &euclideanDistance2D,
        py::arg("points1"), py::arg("points2"));
  m.def("rowwise_min", &rowwiseMin, py::arg("matrix"));
  m.def("colwise_min", &colwiseMin, py::arg("matrix"));
  m.def("q_exponential", &qExponential, py::arg("x"), py::arg("q"));
}
