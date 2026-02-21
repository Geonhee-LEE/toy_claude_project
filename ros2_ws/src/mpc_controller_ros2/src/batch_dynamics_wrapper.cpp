#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

BatchDynamicsWrapper::BatchDynamicsWrapper(const MPPIParams& params)
: params_(params),
  model_(std::make_shared<DiffDriveModel>(
    params.v_min, params.v_max, params.omega_min, params.omega_max))
{
}

BatchDynamicsWrapper::BatchDynamicsWrapper(
  const MPPIParams& params, std::shared_ptr<MotionModel> model)
: params_(params), model_(std::move(model))
{
}

Eigen::MatrixXd BatchDynamicsWrapper::dynamicsBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls
) const
{
  return model_->dynamicsBatch(states, controls);
}

Eigen::MatrixXd BatchDynamicsWrapper::propagateBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls,
  double dt
) const
{
  return model_->propagateBatch(states, controls, dt);
}

std::vector<Eigen::MatrixXd> BatchDynamicsWrapper::rolloutBatch(
  const Eigen::VectorXd& x0,
  const std::vector<Eigen::MatrixXd>& control_sequences,
  double dt
) const
{
  return model_->rolloutBatch(x0, control_sequences, dt);
}

Eigen::MatrixXd BatchDynamicsWrapper::clipControls(const Eigen::MatrixXd& controls) const
{
  return model_->clipControls(controls);
}

}  // namespace mpc_controller_ros2
