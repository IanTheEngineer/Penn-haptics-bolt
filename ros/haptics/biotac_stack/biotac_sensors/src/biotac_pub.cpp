/*********************************************************************
*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2012, University of Pennsylvania
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of University of Pennsylvania nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
* Author: Ian McMahon (imcmahon@grasp.upenn.edu)
*********************************************************************/

#include <ros/ros.h>
#include <biotac_sensors/biotac_hand_class.h>
#include <biotac_sensors/BioTacHand.h>

using namespace std;
using namespace biotac;


int main(int argc, char** argv)
{
  //Initialize ROS node
  ros::init(argc, argv, "biotac_sensors_pub", ros::init_options::AnonymousName);
  ros::NodeHandle n;
  //Loop at 100Hz
  ros::Rate loop_rate(100);
  //Advertise on "biotac_pub" topic
  ros::Publisher biotac_pub = n.advertise<biotac_sensors::BioTacHand>("biotac_pub", 1000);
  //Create a blank message to publish
  biotac_sensors::BioTacHand bt_hand_msg;
  //Check if a null_message parameter was set & publish zero data if so
  if(n.hasParam(ros::this_node::getName() + "/null_msg"))
  {
    biotac_sensors::BioTacData biotac_finger;
    bt_hand_msg.bt_data.push_back(biotac_finger);
    bt_hand_msg.bt_data[0].bt_serial = "Foo";
    bt_hand_msg.bt_data.push_back(biotac_finger);
    bt_hand_msg.bt_data[1].bt_serial = "Bar";
    bt_hand_msg.hand_id = "Null_Data";
    while(ros::ok())
    {
      biotac_pub.publish(bt_hand_msg);
      loop_rate.sleep();
    }
  }
  else //Start the node normally
  {
    //Name the BioTac hand that will be published
    BioTacHandClass left_hand("left_hand");
    //Connect to and configure the sensors
    left_hand.initBioTacSensors();

    while(ros::ok())
    { //Collect a batch of data
      bt_hand_msg = left_hand.collectBatch();
      biotac_pub.publish(bt_hand_msg);
      loop_rate.sleep();
    }
  }
  return 0;
}
