#ifndef MPC_CONTROLLER_ROS2__PI_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__PI_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace mpc_controller_ros2
{

/**
 * @brief ADMM QP Projector for pi-MPPI
 *
 * Solves per-dimension projection QP via ADMM:
 *   min_{v_tilde} (1/2)||v_tilde - v_raw||^2
 *   s.t. u_min <= v_tilde <= u_max           (control bounds)
 *        |D1 * v_tilde| <= rate_max           (rate bounds)
 *        |D2 * v_tilde| <= accel_max          (accel bounds, optional)
 *
 * Matrices D1, D2 are finite difference operators precomputed once.
 * KKT system P = I + rho * A^T * A is LLT-factorized once at configure.
 *
 * Reference: Andrejev et al. (2025) "pi-MPPI: A Projection-based MPPI
 *            Scheme for Smooth Optimal Control" (RA-L 2025, arXiv 2504.10962)
 */
class ADMMProjector
{
public:
  /**
   * @brief Construct projector with precomputed matrices
   * @param N      Horizon length (number of control steps)
   * @param dt     Time step (seconds)
   * @param rho    ADMM penalty parameter
   * @param max_iter  Maximum ADMM iterations
   * @param derivative_order  1 = rate only, 2 = rate + accel
   */
  ADMMProjector(int N, double dt, double rho, int max_iter, int derivative_order);

  /**
   * @brief Project a single control dimension sequence
   * @param v_raw    Input sequence (N,)
   * @param v_out    Output projected sequence (N,)
   * @param u_min    Lower control bound (scalar, this dimension)
   * @param u_max    Upper control bound (scalar, this dimension)
   * @param rate_max Rate bound (scalar, this dimension)
   * @param accel_max Accel bound (scalar, this dimension), ignored if order==1
   */
  void projectDimension(
    const Eigen::VectorXd& v_raw,
    Eigen::VectorXd& v_out,
    double u_min, double u_max,
    double rate_max, double accel_max) const;

  /**
   * @brief Project full control sequence (N x nu)
   * @param in       Input control sequence (N x nu)
   * @param out      Output projected sequence (N x nu)
   * @param u_min    Per-dimension lower bounds (nu,)
   * @param u_max    Per-dimension upper bounds (nu,)
   * @param rate_max Per-dimension rate bounds (nu,)
   * @param accel_max Per-dimension accel bounds (nu,)
   */
  void projectSequence(
    const Eigen::MatrixXd& in,
    Eigen::MatrixXd& out,
    const Eigen::VectorXd& u_min,
    const Eigen::VectorXd& u_max,
    const Eigen::VectorXd& rate_max,
    const Eigen::VectorXd& accel_max) const;

  // Accessors for testing
  int horizon() const { return N_; }
  int derivativeOrder() const { return derivative_order_; }
  const Eigen::MatrixXd& D1() const { return D1_; }
  const Eigen::MatrixXd& D2() const { return D2_; }
  const Eigen::MatrixXd& A() const { return A_; }

private:
  int N_;
  double dt_;
  double rho_;
  int max_iter_;
  int derivative_order_;

  Eigen::MatrixXd D1_;  // (N-1) x N  first-order finite difference
  Eigen::MatrixXd D2_;  // (N-2) x N  second-order finite difference
  Eigen::MatrixXd A_;   // stacked constraint matrix [I; D1; D2]
  int m_;               // total rows of A_

  Eigen::LLT<Eigen::MatrixXd> kkt_llt_;  // precomputed LLT of (I + rho * A^T * A)
};

/**
 * @brief pi-MPPI (Projection MPPI) nav2 Controller Plugin
 *
 * Reference: Andrejev et al. (2025) "pi-MPPI: A Projection-based MPPI
 *            Scheme for Smooth Optimal Control of Fixed-Wing Aerial Vehicles"
 *            (IEEE RA-L 2025, arXiv 2504.10962)
 *
 * Standard MPPI generates non-smooth controls via weighted averaging.
 * pi-MPPI adds an ADMM QP projection step that enforces hard bounds on:
 *   - Control magnitude: u_min <= u <= u_max
 *   - Control rate (1st derivative): |du/dt| <= rate_max
 *   - Control acceleration (2nd derivative): |d²u/dt²| <= accel_max
 *
 * The projection is applied both:
 *   1. Before rollout: each sampled sequence is projected (feasible sampling)
 *   2. After update: final control sequence is projected (feasible output)
 *
 * This guarantees smooth, constraint-satisfying controls without post-hoc
 * filtering, unlike SG filters or Smooth-MPPI.
 *
 * Performance: K=512, N=30, nu=2, 10 ADMM iter -> < 0.5ms overhead
 */
class PiMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  PiMPPIControllerPlugin() = default;
  ~PiMPPIControllerPlugin() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;

protected:
  std::pair<Eigen::VectorXd, MPPIInfo> computeControl(
    const Eigen::VectorXd& current_state,
    const Eigen::MatrixXd& reference_trajectory
  ) override;

  /**
   * @brief Project all K samples in perturbed_buffer_ via ADMM
   */
  void projectAllSamples();

  // ADMM projector
  std::unique_ptr<ADMMProjector> projector_;

  // Per-dimension bounds vectors (nu,)
  Eigen::VectorXd pi_u_min_;
  Eigen::VectorXd pi_u_max_;
  Eigen::VectorXd pi_rate_max_;
  Eigen::VectorXd pi_accel_max_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__PI_MPPI_CONTROLLER_PLUGIN_HPP_
