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


#ifndef _BIOTAC_ARM_CONTROLLER
#define _BIOTAC_ARM_CONTROLLER

#include <ee_cart_imped_action/ee_cart_imped_arm.hh>
#include <ee_cart_imped_msgs/EECartImpedGoal.h>
#include <tf/transform_listener.h>
#include <message_filters/subscriber.h>

//================================================================
// BioTacArm Controller Class
//================================================================
class biotacArmController
{
  private:
    EECartImpedArm::EECartImpedArm *arm_controller;       // Arm controller

  public:

    //================================================================
    // Functions
    //================================================================
    biotacArmController();                                // Constructor
    ~biotacArmController();                               // Destructor
    void moveArmToStart();                                // Move arm to start position
    void slide_down(double, double, double, double);      // Moves arm down
    void getArmTransform();                               // Polls for arm transform
    double getTransform(char);                            // Returns the specified value
   
    //================================================================ 
    // Variables
    //================================================================
    tf::TransformListener *tf_listener;                   // Transform Listener
    tf::StampedTransform *store_transform;                // Store transform
};

#endif // _BIOTAC_ARM_CONTROLLER
