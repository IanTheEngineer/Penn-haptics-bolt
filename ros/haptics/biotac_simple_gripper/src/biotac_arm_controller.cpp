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
*         ee_cart_imped_tutorial - http://www.ros.org/wiki/ee_cart_imped/Tutorials/
*********************************************************************/

#include <biotac_simple_gripper/biotac_arm_controller.h>


//======================================================================
// BioTacArmController class constructor to startup impedance controller
//======================================================================
biotacArmController::biotacArmController()
{
  arm_controller = new EECartImpedArm("l_arm_cart_imped_controller");
  tf_listener = new tf::TransformListener();
  store_transform = new tf::StampedTransform();
}

//======================================================================
// Move arm downwards by the specified distance from the specified
// x,y,z positions.  All in meters
//======================================================================
void biotacArmController::slide_down(double x, double y, double z, double distance)
{
  ee_cart_imped_msgs::EECartImpedGoal traj;
  
  /**
  *addTrajectoryPoint is a static function in the EECartImpedArm class that 
  *adds a trajectory point to the end of the first argument.  It simply 
  *assigns each value in the goal structure for us to prevent having to 
  *write it all out.
  *
  *This is a point in the center of the robot's body.  This simply moves the 
  *arm to that point with maximum stiffness.  
  */
  EECartImpedArm::addTrajectoryPoint(traj, x, y, z, 0, 0, 0, 1,
                                    1000, 1000, 1000, 30, 30, 30,
                                    false, false, false, false, false,
                                    false, 4, "/torso_lift_link");
  /**
  *This point is farther in front of the robot, but it is only allowed to 
  *use a very small stiffness in the x direction
  */
  EECartImpedArm::addTrajectoryPoint(traj, x, y, z-distance, 0, 0, 0, 1,
                                     500, 500, 500, 30, 30, 30,
                                     false, false, false, false, false,
                                     false, 6, "/torso_lift_link");
 /**
 *This is the line that actually sends the trajectory to the action server
 *and starts the arm moving.  The server will block until the arm completes 
 *the trajectory or it is aborted.
 */
  arm_controller->startTrajectory(traj);
}

//======================================================================
// Move arm to the starting point to get object
// This is specified in meters and from the torso_link_lift
//======================================================================
void biotacArmController::moveArmToStart()
{
  ee_cart_imped_msgs::EECartImpedGoal traj;
 
  // Move arm to starting point 
  EECartImpedArm::addTrajectoryPoint(traj, 1.0, 0, 0.04, 0, 0, 0, 1,
                                    1000, 1000, 1000, 30, 30, 30,
                                    false, false, false, false, false,
                                    false, 4, "/torso_lift_link");
  // Send path
  arm_controller->startTrajectory(traj);
}

//======================================================================
// Get current transform of the arm - from torso_lift_link
// to the gripper tool frame using tf_listener
//======================================================================
void biotacArmController::getArmTransform()
{
  bool noTransform = true;
  ros::Rate rate(1);

  while (noTransform && ros::ok())
  {
    try 
    {
      tf_listener->lookupTransform("/torso_lift_link", "/l_gripper_tool_frame", ros::Time(0), *store_transform);
      noTransform = false;
    }
    catch (tf::TransformException ex)
    {
      ROS_INFO("No valid transform. Trying again");
    }
    rate.sleep();
  }  
}

//======================================================================
// Simple getter that grabs the transforms for the specified channel
//======================================================================
double biotacArmController::getTransform(char channel)
{
  switch (channel)
  {
    case 'x':
      return store_transform->getOrigin().x();
    case 'y':
      return store_transform->getOrigin().y();
    case 'z':
      return store_transform->getOrigin().z();
  }
  return -1;
}


