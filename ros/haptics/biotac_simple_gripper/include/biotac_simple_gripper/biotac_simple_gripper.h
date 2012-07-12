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

#include <ros/ros.h>
#include <pr2_controllers_msgs/Pr2GripperCommandAction.h>
#include <actionlib/client/simple_action_client.h>

#ifndef _BIOTAC_SIMPLE_GRIPPER
#define _BIOTAC_SIMPLE_GRIPPER

typedef actionlib::SimpleActionClient<pr2_controllers_msgs::Pr2GripperCommandAction> GripperClient;

//================================================================
// BioTacSimpleGripper Class Definition
//================================================================
class biotacSimpleGripper
{
  private:
    GripperClient* gripper_client_;                         // Action client gripper
   
    // Constants
    static const double GripperMaxOpenPosition = 0.08;      // Distance in meters (8cm)
    static const double GripperForceGentle = 15.0;          // In "Newtons"
    static const double GripperForceMax = -1.0;             // No effort limit
    double last_position_;                                  // Store last known position of gripper

  public:
    biotacSimpleGripper();              // Constructor
    ~biotacSimpleGripper();             // Destructor
    void open2Position(double);         // Open the gripper to specified distance
    void openByAmount(double);          // Opens the gripper by specified amount 
    void closeByAmount(double);         // Closes gripper by specified amount 
    double getGripperLastPosition();      // Returns last known gripper position

};

//================================================================
// Simple Template for getting max of two values
//================================================================
template <class T>
const T& max(const T& a, const T& b)
{
  return (a > b ? a:b);
};


#endif  // _BIOTAC_SIMPLE_GRIPPER

