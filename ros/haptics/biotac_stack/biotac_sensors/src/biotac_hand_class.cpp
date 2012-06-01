#include <biotac_hand_class.h>
#include <stdio.h>
#include <iostream>

//=========================================================================
// CONSTANTS
//=========================================================================

static const unsigned char parity_values[] = \
                { 0x01, 0x02, 0x04, 0x07, 0x08, 0x0B, 0x0D, 0x0E, \
                  0x10, 0x13, 0x15, 0x16, 0x19, 0x1A, 0x1C, 0x1F, \
                  0x20, 0x23, 0x25, 0x26, 0x29, 0x2A, 0x2C, 0x2F, \
                  0x31, 0x32, 0x34, 0x37, 0x38, 0x3B, 0x3D, 0x3E, \
                  0x40, 0x43, 0x45, 0x46, 0x49, 0x4A, 0x4C, 0x4F, \
                  0x51, 0x52, 0x54, 0x57, 0x58, 0x5B, 0x5D, 0x5E, \
                  0x61, 0x62, 0x64, 0x67, 0x68, 0x6B, 0x6D, 0x6E, \
                  0x70, 0x73, 0x75, 0x76, 0x79, 0x7A, 0x7C, 0x7F, \
                  0x80, 0x83, 0x85, 0x86, 0x89, 0x8A, 0x8C, 0x8F, \
                  0x91, 0x92, 0x94, 0x97, 0x98, 0x9B, 0x9D, 0x9E, \
                  0xA1, 0xA2, 0xA4, 0xA7, 0xA8, 0xAB, 0xAD, 0xAE, \
                  0xB0, 0xB3, 0xB5, 0xB6, 0xB9, 0xBA, 0xBC, 0xBF, \
                  0xC1, 0xC2, 0xC4, 0xC7, 0xC8, 0xCB, 0xCD, 0xCE, \
                  0xD0, 0xD3, 0xD5, 0xD6, 0xD9, 0xDA, 0xDC, 0xDF, \
                  0xE0, 0xE3, 0xE5, 0xE6, 0xE9, 0xEA, 0xEC, 0xEF, \
                  0xF1, 0xF2, 0xF4, 0xF7, 0xF8, 0xFB, 0xFD, 0xFE};

using namespace biotac;


BioTacHandClass::BioTacHandClass(string hand_id)
{
  hand_id_ = hand_id;
  biotac_.spi_clock_speed = BT_SPI_BITRATE_KHZ_DEFAULT;
  biotac_.number_of_biotacs = 0;
  biotac_.sample_rate_Hz = BT_SAMPLE_RATE_HZ_DEFAULT;
  //Using the Default frame type
  biotac_.frame.frame_type = 0;
  //Only sample 1 frame every batch
  biotac_.batch.batch_frame_count = 1;
  //Configure the Cheetah to sample every 10ms
  biotac_.batch.batch_ms = 10;

  // Set the duration of the run time
  double length_of_data_in_second = 0.01;
  number_of_samples_ = (int)(BT_SAMPLE_RATE_HZ_DEFAULT * length_of_data_in_second);

  // Check if any initial settings are wrong
  if (MAX_BIOTACS_PER_CHEETAH != 3 && MAX_BIOTACS_PER_CHEETAH != 5)
  {
    BioTac bt_err_code = BT_WRONG_MAX_BIOTAC_NUMBER;
    displayErrors(bt_err_code);
  }

}

//========================================================
//Clean up the Cheetah
//========================================================
BioTacHandClass::~BioTacHandClass()
{
  bt_cheetah_close(ch_handle_);
}

//========================================================
//Initialization of BioTac sensors & Cheetah
//This function will keep retrying to find BioTac/Cheetah
//devices, every second
//========================================================
void BioTacHandClass::initBioTacSensors()
{
  BioTac bt_err_code = BT_NO_CHEETAH_DETECTED;
  ros::Rate loop_rate_cheetah(1);
  //Initialize the Cheetah Device 
  while( bt_err_code < BT_OK && ros::ok())
  {
    bt_err_code = bt_cheetah_initialize(&biotac_, &ch_handle_);
    if(bt_err_code < BT_OK)
    {
      displayErrors(bt_err_code);
      ROS_INFO("Attempting to initialize the Cheetah again in 1 second...");
      loop_rate_cheetah.sleep();
    }
  }

  bt_err_code = BT_NO_BIOTAC_DETECTED;
  ros::Rate loop_rate_biotac_connect(1);
  //Grabbing properties of the BioTac sensors
  while( bt_err_code < BT_OK && ros::ok())
  { 
    bt_err_code = getBioTacProperties();
    if(bt_err_code < BT_OK)
    {
      displayErrors(bt_err_code);
      ROS_INFO("Attempting to connect to BioTac sensors again in 1 second...");
      loop_rate_biotac_connect.sleep();
    }
  }

  bt_err_code = BT_DATA_SIZE_TOO_SMALL;
  ros::Rate loop_rate_biotac_configure(1);
  //Configures the SPI batch to send to the BioTac sensors
  while( bt_err_code < BT_OK && ros::ok())
  { 
    bt_err_code = configureBatch();
    if(bt_err_code < BT_OK)
    {
      displayErrors(bt_err_code);
      ROS_INFO("Configuring the BioTac SPI batch again, this time using default parameters.");
      ROS_INFO("Attempting to configure the BioTac SPI batch again in 1 second...");
      //Default Parameters
      biotac_.spi_clock_speed = BT_SPI_BITRATE_KHZ_DEFAULT;
      biotac_.number_of_biotacs = 0;
      biotac_.sample_rate_Hz = BT_SAMPLE_RATE_HZ_DEFAULT;
      //Using the Default frame type
      biotac_.frame.frame_type = 0;
      //Only sample 1 frame every batch
      biotac_.batch.batch_frame_count = 1;
      //Configure the Cheetah to sample every 10ms
      biotac_.batch.batch_ms = 10;
      // Set the duration of the run time
      double length_of_data_in_second = 0.01;
      number_of_samples_ = (int)(BT_SAMPLE_RATE_HZ_DEFAULT * length_of_data_in_second);
      //Sleep for 1 second
      loop_rate_biotac_configure.sleep();
    }
  }

  //Construct the BioTacHand Message
  biotac_sensors::BioTacData one_finger;
  for(int i = 0; i < biotac_.number_of_biotacs; i++)
  {
    bt_hand_msg_.bt_data.push_back(one_finger);
  }
  if(bt_err_code == BT_OK)
  {
    ROS_INFO("Configuring of Cheetah SPI and BioTac sensors complete.");
    ROS_INFO("Reading BioTac data...");
  }
}

//========================================================
//Get and print properties of the BioTac(s)
//========================================================
BioTac BioTacHandClass::getBioTacProperties()
{
  biotac_.number_of_biotacs = 0;
  bt_property biotac_property[MAX_BIOTACS_PER_CHEETAH];
  BioTac bt_err_code;
  for (int i = 0; i < MAX_BIOTACS_PER_CHEETAH; i++)
  {
    bt_err_code = bt_cheetah_get_properties(ch_handle_, i+1, &(biotac_property[i]));
    if (bt_err_code)
    {//Check to see if the properties were successfully acquired
      return bt_err_code;
    }

    if (biotac_property[i].bt_connected == YES)
    {
      (biotac_.number_of_biotacs)++;
      //Store off BioTac serial number
      finger_info new_finger;
      char serial_char_array [25];
      sprintf(serial_char_array, "%c%c-%c%c-%c%c.%c.%c-%c-%c%c-%c-%c%c%c%c", \
                    biotac_property[i].serial_number[0], biotac_property[i].serial_number[1], \
                    biotac_property[i].serial_number[2], biotac_property[i].serial_number[3], \
                    biotac_property[i].serial_number[4], biotac_property[i].serial_number[5], \
                    biotac_property[i].serial_number[6], biotac_property[i].serial_number[7], \
                    biotac_property[i].serial_number[8], biotac_property[i].serial_number[9], \
                    biotac_property[i].serial_number[10], biotac_property[i].serial_number[11], \
                    biotac_property[i].serial_number[12], biotac_property[i].serial_number[13], \
                    biotac_property[i].serial_number[14], biotac_property[i].serial_number[15]);
      new_finger.finger_serial = string(serial_char_array);
      //Store off BioTac position on the Multi-BioTac Board
      new_finger.finger_position = i+1;
      biotac_serial_no_info_.push_back(new_finger);
      ROS_INFO("Detected a BioTac at board position %d with serial number %s", \
                new_finger.finger_position, serial_char_array+'\0');
    }

  }
  
  // Check if any BioTacs detected
  if (biotac_.number_of_biotacs == 0)
  {
    bt_err_code = BT_NO_BIOTAC_DETECTED;
    return bt_err_code;
  }
  //Report total number of Biotacs
  ROS_INFO("%d BioTac(s) detected.", biotac_.number_of_biotacs);
  
  bt_err_code = BT_OK;
  return bt_err_code;
}

//=================================
//Configure the SPI batch 
//=================================
BioTac BioTacHandClass::configureBatch()
{
  BioTac bt_err_code;

  bt_err_code = bt_cheetah_configure_batch(ch_handle_, &biotac_, number_of_samples_);

  if(bt_err_code == BT_OK)
  {
    ROS_INFO("Configured the Cheetah batch.");
    ros::Time frame_start_time_ = ros::Time::now();
  }

  return bt_err_code;
}
//=========================================================================
// Collect a single SPI Batch
//=========================================================================

biotac_sensors::BioTacHand BioTacHandClass::collectBatch()
{
  int i, j;
  int byte_shift = 2 + MAX_BIOTACS_PER_CHEETAH*2;			// 2 bytes of command + 2 bytes per BioTac data
  int spi_data_len;
  int number_of_samples_in_batch;
  static u08 *bt_raw_data;
  unsigned int channel_id;
  int finger_vector_pos;

  spi_data_len = ch_spi_batch_length(ch_handle_);
  bt_raw_data = (u08*) malloc(spi_data_len * sizeof *bt_raw_data);
  ch_spi_async_collect(ch_handle_, spi_data_len, bt_raw_data);
  ros::Time frame_end_time = ros::Time::now();
  ch_spi_async_submit(ch_handle_);

  number_of_samples_in_batch = spi_data_len/byte_shift;

  
  bt_hand_msg_.hand_id = hand_id_;
  bt_hand_msg_.header.stamp = frame_end_time;
  bt_hand_msg_.bt_time.frame_start_time = frame_start_time_;
  bt_hand_msg_.bt_time.frame_end_time = frame_end_time;
  double sample_total_time = (frame_end_time.sec + (double)frame_end_time.nsec/1000000) - \
                             (frame_start_time_.sec + (double)frame_start_time_.nsec/1000000);
  unsigned int time_step = (unsigned int)((sample_total_time / number_of_samples_in_batch) * 1000000.0);
  frame_start_time_ = frame_end_time;

  for(unsigned int k = 0; k < biotac_serial_no_info_.size(); k++)
  {
    bt_hand_msg_.bt_data[k].bt_position = biotac_serial_no_info_[k].finger_position;
    bt_hand_msg_.bt_data[k].bt_serial = biotac_serial_no_info_[k].finger_serial;
  }

  unsigned int ns_offset = 0;
  unsigned int pac_index = 0;
  unsigned int electrode_index = 0;
  unsigned int spi_data;
  for(i = 0; i < number_of_samples_in_batch; i++)
  {
    channel_id = (biotac_.frame.frame_structure[i%(biotac_.frame.frame_size)] & 0x7E) >> 1;
    finger_vector_pos = 0;
    for(j = 0; j < MAX_BIOTACS_PER_CHEETAH; j++)
    {
      spi_data = (unsigned int) (bt_raw_data[i*byte_shift + j*2 + 2] >> 1) * 32 + (bt_raw_data[i*byte_shift + j*2 + 3] >> 3);

      if((parity_values[bt_raw_data[i*byte_shift + j*2 + 2] >> 1] == bt_raw_data[i*byte_shift + j*2 + 2]) && \
          (parity_values[bt_raw_data[i*byte_shift + j*2 + 3] >> 1] == bt_raw_data[i*byte_shift + j*2 + 3]))
      {//data[index].bt_parity[j] = PARITY_GOOD;
        //bt_hand_msg_.bt_data[finger_vector_pos].bt_position = j+1;
        switch (channel_id)
        {
           case TDC:
              //Temperature DC
              bt_hand_msg_.bt_data[finger_vector_pos].tdc_data = spi_data;
              bt_hand_msg_.bt_time.tdc_ns_offset = ns_offset;
              break;
           case TAC:
              //Temperature AC
              bt_hand_msg_.bt_data[finger_vector_pos].tac_data = spi_data;
              bt_hand_msg_.bt_time.tac_ns_offset = ns_offset;
              break;
           case PDC:
              //Pressure DC
              bt_hand_msg_.bt_data[finger_vector_pos].pdc_data = spi_data;
              bt_hand_msg_.bt_time.pdc_ns_offset = ns_offset;
              break;
           case PAC:
              //Pressure AC
              bt_hand_msg_.bt_data[finger_vector_pos].pac_data[pac_index] = spi_data;
              bt_hand_msg_.bt_time.pac_ns_offset[pac_index] = ns_offset;
              break;
           default:
              //Electrodes
              if(channel_id >= ELEC_LO && channel_id <= ELEC_HI)
              {//Electrode
                electrode_index = channel_id - ELEC_LO;
                bt_hand_msg_.bt_data[finger_vector_pos].electrode_data[electrode_index] = spi_data;
                bt_hand_msg_.bt_time.electrode_ns_offset[electrode_index] = ns_offset;
              }
              else
              {//error
              }
        }  

        if( finger_vector_pos < (int)bt_hand_msg_.bt_data.size()-1 )
        {
          finger_vector_pos++;
        }
      }
      //else
      //{
      //  data[index].bt_parity[j] = PARITY_BAD;
      //}

     }

    //Increment PAC index
     if(channel_id == PAC)
     {
       pac_index++;
     }
     //Add seconds to offset
     ns_offset += time_step;

  }
  free(bt_raw_data);
  return bt_hand_msg_;
}


//=========================================================================
// PRINT ERROR CODES
//=========================================================================
void BioTacHandClass::displayErrors(BioTac bt_err_code)
{

  switch(bt_err_code)
  {
  case BT_OK:
    //Break, nothing is wrong
    break;
  case BT_WRONG_NUMBER_ASSIGNED:
    ROS_WARN("Wrong BioTac number assigned!");
    break;
  case BT_NO_BIOTAC_DETECTED:
    ROS_ERROR("No BioTacs are detected!");
    break;
  case BT_WRONG_MAX_BIOTAC_NUMBER:
    ROS_WARN("Wrong maximum number of BioTacs assigned (should be 3 or 5)!");
    break;
  case BT_DATA_SIZE_TOO_SMALL:
    ROS_WARN("The number of samples is too small! Using default sample size of ____.");
    break;
  case BT_NO_CHEETAH_DETECTED:
    ROS_ERROR("No Cheetah device detected!");
    break;
  case BT_UNABLE_TO_OPEN_CHEETAH:
    ROS_ERROR("Unable to open Cheetah device on current port.");
    break;
  case BT_UNABLE_TO_OPEN_FILE:
    ROS_ERROR("Cannot open output file.");
    break;
  default:
    ROS_ERROR("Unrecognized Biotac error encountered.");
    break;
  }

}
