#ifndef BIOTACHANDCLASS_H_
#define BIOTACHANDCLASS_H_


#include <ros/ros.h>
#include <biotac_sensors/BioTacHand.h>
#include <vector>
#include <string>

using namespace std;

namespace biotac {

extern "C"{
#include <biotac.h>
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
