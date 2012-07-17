#include <pr2_gripper_accelerometer/pr2_gripper_accelerometer.h>
#include <pluginlib/class_list_macros.h>
#include <pr2_hardware_interface/hardware_interface.h>

using namespace pr2_gripper_accelerometer; 

bool PR2GripperAccelerometer::init(pr2_mechanism_model::RobotState *robot, ros::NodeHandle &n)
{
  // get a handle to the hardware interface 
  pr2_hardware_interface::HardwareInterface* hardwareInterface = robot->model_->hw_;
  if(!hardwareInterface)
  {
    ROS_ERROR("Perhaps Something wrong with the hardware interface pointer!!!!");
  }

  // get a handle to our accelerometer 
  std::string accelerometer_name;
  if(!n.getParam("accelerometer_name", accelerometer_name))
  {
    ROS_ERROR("No accelerometer given in namespace: '%s')", n.getNamespace().c_str());
    return false;
  }
  accelerometerHandle = hardwareInterface->getAccelerometer(accelerometer_name);

  if(!accelerometerHandle)
  {
     ROS_ERROR("PR2GripperSensorController could not find sensor named '%s'", accelerometer_name.c_str());
     return false;
  }

  // Setup accelerometer values
  accelerometerHandle->command_.bandwidth_= 6;    // 1.5kz bandwidth
  accelerometerHandle->command_.range_= 2;        // +/- 8g range

  // get a handle to our desired joint 
  std::string joint_name;
  if (!n.getParam("joint_name", joint_name))
  {
    ROS_ERROR("No joint given in namespace: '%s')", n.getNamespace().c_str());
    return false;
  }
  joint_state_ = robot->getJointState(joint_name);
  if (!joint_state_)
  {
    ROS_ERROR("PR2GripperSensorController could not find joint named '%s'", joint_name.c_str());
    return false;
  }

  // Setup publishers for accelerometer and gripper
  gripper_accelerometer_publisher_.reset(new realtime_tools::RealtimePublisher<pr2_gripper_accelerometer::PR2GripperAccelerometerData>(n, "data", 1));

  return true;
}

void PR2GripperAccelerometer::starting()
{
  ROS_INFO("Starting controller to publish values");
  double init_pos_ = joint_state_ ->position_;
  ROS_INFO("Initial position is [%f]", init_pos_);
}

void PR2GripperAccelerometer::update()
{
  if (gripper_accelerometer_publisher_->trylock())
  {
    current_pos_ = joint_state_ -> position_;
    std::vector<geometry_msgs::Vector3> threeAccs = accelerometerHandle->state_.samples_;
    uint numReadings = threeAccs.size();
   
    gripper_accelerometer_publisher_->msg_.acc_x_raw = threeAccs[numReadings].x; 
    gripper_accelerometer_publisher_->msg_.acc_y_raw = threeAccs[numReadings].y;
    gripper_accelerometer_publisher_->msg_.acc_z_raw = threeAccs[numReadings].z;

    gripper_accelerometer_publisher_->msg_.aperture_position = joint_state_->position_;

    gripper_accelerometer_publisher_->unlockAndPublish();
  } 
}

void PR2GripperAccelerometer::stopping()
{}

// Register controller to pluginlib
PLUGINLIB_DECLARE_CLASS(pr2_gripper_accelerometer, PR2GripperAccelerometer,
                        pr2_gripper_accelerometer::PR2GripperAccelerometer, 
                        pr2_controller_interface::Controller)
