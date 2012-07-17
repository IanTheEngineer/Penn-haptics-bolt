#include <pr2_controller_interface/controller.h>
#include <pr2_mechanism_model/joint.h>
#include <boost/scoped_ptr.hpp>
#include <realtime_tools/realtime_publisher.h>
#include <pr2_gripper_accelerometer/PR2GripperAccelerometerData.h>

namespace pr2_gripper_accelerometer
{
  class PR2GripperAccelerometer: public pr2_controller_interface::Controller
  {
    private:
      pr2_mechanism_model::JointState* joint_state_;
      double current_pos_;
      // raw data real-time publisher
      boost::scoped_ptr<realtime_tools::RealtimePublisher< pr2_gripper_accelerometer::PR2GripperAccelerometerData> > gripper_accelerometer_publisher_;
      // Pointer to accelerometer
      pr2_hardware_interface::Accelerometer* accelerometerHandle;

    public:
      virtual bool init(pr2_mechanism_model::RobotState *robot, 
      ros::NodeHandle &n);
      virtual void starting();
      virtual void update();
      virtual void stopping();
  };
}
