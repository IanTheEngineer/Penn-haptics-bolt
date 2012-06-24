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

#ifndef BIOTACHANDCLASS_H_
#define BIOTACHANDCLASS_H_


#include <ros/ros.h>
#include <biotac_sensors/BioTacHand.h>
#include <vector>
#include <string>

using namespace std;

namespace biotac {

extern "C"{
#include <biotac_sensors/biotac.h>
}

struct finger_info {
  string finger_serial;
  int finger_position;
};

class BioTacHandClass
{
private:
  bt_info biotac_;
  Cheetah ch_handle_;
  int number_of_samples_;
  std::vector<finger_info> biotac_serial_no_info_;
  string hand_id_;
  ros::Time frame_start_time_;

  biotac_sensors::BioTacHand bt_hand_msg_;

  enum{ PAC = 0, PDC = 1, TAC = 2, \
        TDC = 3, ELEC_LO = 17, ELEC_HI = 35};

  BioTac  getBioTacProperties();
  BioTac  configureBatch();
  void    displayErrors(BioTac bt_err_code);
public:
  BioTacHandClass(string hand_id);
  ~BioTacHandClass();
  void initBioTacSensors();
  biotac_sensors::BioTacHand collectBatch();
};

}
#endif
