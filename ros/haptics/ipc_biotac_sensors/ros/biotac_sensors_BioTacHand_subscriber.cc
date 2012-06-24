#include <ros/ros.h>
#include <ipc_bridge/ipc_bridge.h>

#include <biotac_sensors/BioTacHand.h>
#include <ipc_bridge/msgs/biotac_sensors_BioTacHand.h>

#define NAMESPACE biotac_sensors
#define NAME BioTacHand

ros::Publisher pub;
NAMESPACE::NAME out_msg;

void callback(const ipc_bridge::NAMESPACE::NAME &msg)
{ 
  out_msg.header.seq = msg.header.seq;
  out_msg.header.stamp = ros::Time(msg.header.stamp);
  if (msg.header.frame_id != 0)
    out_msg.header.frame_id = std::string(msg.header.frame_id);
  else
    out_msg.header.frame_id = std::string("");

  if (msg.hand_id != 0)
    out_msg.hand_id = std::string(msg.hand_id);
  else
    out_msg.hand_id = std::string("");

  out_msg.bt_data.resize(msg.bt_data_length);
  for (unsigned int i = 0; i < msg.bt_data_length; i++)
    {
      if (msg.bt_data[i].bt_serial != 0)
        out_msg.bt_data[i].bt_serial = std::string(msg.bt_data[i].bt_serial);
      else
        out_msg.bt_data[i].bt_serial = std::string("");

      out_msg.bt_data[i].tdc_data = msg.bt_data[i].tdc_data;
      out_msg.bt_data[i].tac_data = msg.bt_data[i].tac_data;
      out_msg.bt_data[i].pdc_data = msg.bt_data[i].pdc_data;

      for (int j = 0; j < 22; j++)
        out_msg.bt_data[i].pac_data[j] = msg.bt_data[i].pac_data[j];

      for (int j = 0; j < 19; j++)
        out_msg.bt_data[i].electrode_data[j] = msg.bt_data[i].electrode_data[j];

    }

  out_msg.bt_time.frame_start_time = ros::Time(msg.bt_time.frame_start_time);
  out_msg.bt_time.frame_end_time = ros::Time(msg.bt_time.frame_end_time);

  out_msg.bt_time.tdc_ns_offset = msg.bt_time.tdc_ns_offset;
  out_msg.bt_time.tac_ns_offset = msg.bt_time.tac_ns_offset;
  out_msg.bt_time.pdc_ns_offset = msg.bt_time.pdc_ns_offset;
  for (int i = 0; i < 22; i++)
    out_msg.bt_time.pac_ns_offset[i] = msg.bt_time.pac_ns_offset[i];

  for (int i = 0; i < 19; i++)
    out_msg.bt_time.electrode_ns_offset[i] = msg.bt_time.electrode_ns_offset[i];

  pub.publish(out_msg);
}

#include "subscriber.h"
