#include <ros/ros.h>
#include <biotac_hand_class.h>
#include <biotac_sensors/BioTacHand.h>

using namespace std;
using namespace biotac;


int main(int argc, char** argv)
{
  ros::init(argc, argv, "biotac_pub", ros::init_options::AnonymousName);
  ros::NodeHandle n;
  ros::Rate loop_rate(100);
  ros::Publisher biotac_pub = n.advertise<biotac_sensors::BioTacHand>("biotac_pub", 1000);
  BioTacHandClass left_hand("left_hand");
  left_hand.initBioTacSensors();
  biotac_sensors::BioTacHand bt_hand_msg;

  while(ros::ok())
  {
    bt_hand_msg = left_hand.collectBatch();
    biotac_pub.publish(bt_hand_msg);
    loop_rate.sleep();
  }
  return 0;
}


