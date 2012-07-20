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
* Author: Vivian Chu (chuv@grasp.upenn.edu)
*********************************************************************/

//#include <biotac_simple_gripper/biotac_observer.h>
#include <biotac_simple_gripper/biotac_simple_gripper.h>

//================================================================
// BioTacSimpleGripper class constructor
// Initializes gripper action client
//================================================================
biotacSimpleGripper::biotacSimpleGripper()
{
  // Initialize the left gripper client controller
  gripper_client_ = new GripperClient("l_gripper_controller/gripper_action", true);

  // Wait for the gripper action server to come up
  while(!gripper_client_->waitForServer(ros::Duration(5.0)) && ros::ok())
  {
    ROS_INFO("Waiting for the l_gripper_controller/gripper_action action server to come up"); 
  }
}

//================================================================
// Destructor
//================================================================
biotacSimpleGripper::~biotacSimpleGripper()
{
  delete gripper_client_;
}

//================================================================
// Open the gripper to the position specified in meters
//================================================================
void biotacSimpleGripper::open2Position(double open_pos)
{
  pr2_controllers_msgs::Pr2GripperCommandGoal open;

  // Don't allow opening larger than max
  double position = open_pos;
  if (position > GripperMaxOpenPosition) position = GripperMaxOpenPosition;

  open.command.position = position;     
  open.command.max_effort = GripperForceMax;  // Do not limit effort (negative)

  ROS_INFO("Sending open goal of: [%f]", position);
  gripper_client_->sendGoal(open);
  gripper_client_->waitForResult();

  // Check gripper state and update position
  if(gripper_client_->getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_INFO("The gripper opened to [%f]", open_pos);
    last_position_ = open_pos;
  }
  else
  {
    ROS_INFO("The gripper failed to open to [%f]", open_pos);
  }
}

//================================================================
// Open the gripper by a set amount in meters
// This allows for slow or fast movements depending on how 
// large of a distance away from the last position the gripper 
// was controlled to move
//================================================================
void biotacSimpleGripper::openByAmount(double move_gripper_distance)
{
  pr2_controllers_msgs::Pr2GripperCommandGoal open;
  
  ROS_INFO("Last position is %f", last_position_);
  double position = last_position_+ move_gripper_distance;  // Move gripper by specified distance
 
  // Check if gripper position valid 
  if (position > GripperMaxOpenPosition) position = GripperMaxOpenPosition;
  open.command.position = position;
  open.command.max_effort = GripperForceMax;  // Do not limit effort (negative)

  ROS_INFO("Sending open goal of [%f]", position);
  gripper_client_->sendGoal(open);
  gripper_client_->waitForResult();

  // Check for success and then update position
  if(gripper_client_->getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
  { 
    ROS_INFO("The gripper by [%f]", move_gripper_distance);
    last_position_ = position;
  }
  else
  {
    ROS_INFO("The gripper failed to open by [%f]", move_gripper_distance);
  }
}

//================================================================
// Close the gripper by a set amount in meters
//================================================================
void biotacSimpleGripper::closeByAmount(double move_gripper_distance)
{
  pr2_controllers_msgs::Pr2GripperCommandGoal close;

  // Grab lock to prevent closing multiple times 
  //boost::shared_lock<boost::shared_mutex> lock(biotacObserver::biotac_mutex_);
  ROS_INFO("Last position is: %f", last_position_); 
  double position = last_position_- move_gripper_distance;  // Move gripper by specified distance 
  
  // CHeck if gripper position valid
  if (position < 0) position = 0.0;        
  close.command.position = position;
  close.command.max_effort = GripperForceGentle;       // Close gently 

  ROS_INFO("Sending close goal of: [%f]", position);
  gripper_client_ -> sendGoal(close);
  gripper_client_ -> waitForResult();

  // Check for success and then update position
  if(gripper_client_->getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_INFO("The gripper closed by [%f]", move_gripper_distance);
    last_position_ = position;          
  }
  else
  {
    ROS_INFO("The gripper failed to close by [%f]", move_gripper_distance);
  }
}

//================================================================
// Returns last known gripper position
// Gives access to the private variable that stored last position
//================================================================
double biotacSimpleGripper::getGripperLastPosition()
{
  return last_position_;
}

