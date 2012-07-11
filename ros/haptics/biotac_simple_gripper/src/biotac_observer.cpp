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
*         Ian McMahon (imcmahon@grasp.upenn.edu)
*********************************************************************/


#include <biotac_simple_gripper/biotac_observer.h>

//======================================================================
// BioTacObserver class constructor - setups flags and storage vects
//======================================================================
biotacObserver::biotacObserver()
{
  // Setup normalization flags and storage 
  init_complete_flag_ = false;
  init_pressure_vect_.resize(NumberFingers);
}

//======================================================================
// Callback function for BioTac Sensors
// Stores the pressure values to get ready to normalize.  When values
// reach NumberNormalizeValues, the mean is computed and used to 
// normalize the sensors
//======================================================================
void biotacObserver::bioTacCB(const biotac_sensors::BioTacHand::ConstPtr& msg)
{
  // Store most current pressure value 
  pressure_[Left] = msg->bt_data[Left].pdc_data;
  pressure_[Right] = msg->bt_data[Right].pdc_data;

  if (init_pressure_vect_[Left].size() < NumberNormalizeValues)
  {
    // Store raw pressure values into vector for normalizing
    init_pressure_vect_[Left].push_back(pressure_[Left]);
    init_pressure_vect_[Right].push_back(pressure_[Right]);
  } 
  else
  {
    normalize_pressure();
  }
}

//======================================================================
// Normalize Pressure From BioTacs
// Where the mean and actual normalization is done
//======================================================================
void biotacObserver::normalize_pressure()
{
  // If the mean has never been set, then set the mean
  if (!init_complete_flag_)
  {
    // Sum up values to take the mean
    int sum[NumberFingers] = {0};
    for (unsigned int i = 0; i < init_pressure_vect_[Left].size(); i++)
    {
      sum[Left] += init_pressure_vect_[Left][i];
      sum[Right] += init_pressure_vect_[Right][i];
    }

    pressure_mean_[Left] = sum[Left]/init_pressure_vect_[Left].size();
    pressure_mean_[Right] = sum[Right]/init_pressure_vect_[Right].size();
    
    init_complete_flag_ = true;
    ROS_INFO("Setting the left finger mean to: %d", pressure_mean_[Left]);
    ROS_INFO("Setting the right finger mean to: %d", pressure_mean_[Right]);
  }

  // Grab lock on pressure_normalized because it is accessed publically
  boost::upgrade_lock<boost::shared_mutex> lock(biotac_mutex_);
  // For writing
  boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);

  // Subtract the mean from newest pressure reading
  pressure_normalized_[Left] = pressure_[Left] - pressure_mean_[Left];
  pressure_normalized_[Right] = pressure_[Right] - pressure_mean_[Right];
}

//======================================================================
// Destructor - clear out vectors
//======================================================================
biotacObserver::~biotacObserver()
{
  init_pressure_vect_[Left].clear();
  init_pressure_vect_[Right].clear();
}

