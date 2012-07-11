#ifndef __IPC_BRIDGE_MATLAB_BIOTAC_SENSORS_BIOTACHAND__
#define __IPC_BRIDGE_MATLAB_BIOTAC_SENSORS_BIOTACHAND__
#include <ipc_bridge_matlab/ipc_bridge_matlab.h>
#include <ipc_bridge/msgs/biotac_sensors_BioTacHand.h>

#include <rosgraph_msgs_Header.h>
#include <biotac_sensors_BioTacData.h>
#include <biotac_sensors_BioTacTime.h>

namespace ipc_bridge_matlab
{
  namespace biotac_sensors
  {
    namespace BioTacHand
    {
      static mxArray* ProcessMessage(const ipc_bridge::biotac_sensors::BioTacHand &msg)
      {
        const char *fields[] = {"header", "hand_id", "bt_data", "bt_time"};
        const int nfields = sizeof(fields)/sizeof(*fields);
        mxArray *out = mxCreateStructMatrix(1, 1, nfields, fields);

      mxSetField(out, 0, "header",ipc_bridge_matlab::Header::ProcessMessage(msg.header));

      if (msg.hand_id == 0)
        {
          char buf[1] = "";
          mxSetField(out, 0, "hand_id", mxCreateString(buf));
        }
      else
        {
          char buf[strlen(msg.hand_id) + 1];
          strcpy(buf, msg.hand_id);
          mxSetField(out, 0, "hand_id", mxCreateString(buf));
        }

       const int length = msg.bt_data_length;   
       mxArray *bt_data = mxCreateCellArray(1, &length);
       for (unsigned int i = 0; i < length; i++)
          mxSetCell(bt_data, i,  
            ipc_bridge_matlab::biotac_sensors::BioTacData::ProcessMessage(msg.bt_data[i]));
       mxSetField(out, 0, "bt_data", bt_data);

       mxSetField(out, 0, "bt_time",ipc_bridge_matlab::biotac_sensors::BioTacTime::ProcessMessage(msg.bt_time));

       return out;
      }

      static int ProcessArray(const mxArray *a, ipc_bridge::biotac_sensors::BioTacHand &msg)
      {
        mxArray *field;

        field = mxGetField(a, 0, "header");
        ipc_bridge_matlab::Header::ProcessArray(field, msg.header);

        field = mxGetField(a, 0, "hand_id");
        int buflen = 128;
        char buf[buflen];
        mxGetString(field, buf, buflen);
        if (strlen(buf) > 0)
        {
          msg.hand_id = new char[strlen(buf) + 1];
          strcpy(msg.hand_id, buf);
        }

        field = mxGetField(a, 0, "bt_time");
        ipc_bridge_matlab::biotac_sensors::BioTacTime::ProcessArray(field, msg.bt_time);

        field = mxGetField(a, 0, "bt_data");
        int nrows = mxGetM(field);
        int ncols = mxGetN(field);

        unsigned int length = nrows;
        if (nrows < ncols)
          length = ncols;
        msg.bt_data_length = length;

        if ((ncols == 0) || (nrows == 0)) 
        {   
           msg.bt_data_length = 0;
           msg.bt_data = 0;
        }   

        if (msg.bt_data_length > 0)
        {   
          msg.bt_data = new biotac_sensors_BioTacData[msg.bt_data_length];
          for (unsigned int i = 0; i < msg.bt_data_length; i++)
          {
            mxArray *p = mxGetCell(field, i);
            ipc_bridge_matlab::biotac_sensors::BioTacData::ProcessArray(p, msg.bt_data[i]);
          }
        }



        return SUCCESS;
      }


      static void Cleanup(ipc_bridge::biotac_sensors::BioTacHand &msg)
      {
        ipc_bridge_matlab::Header::Cleanup(msg.header);
        if (msg.hand_id != 0)
         {
           delete[] msg.hand_id;
           msg.hand_id = 0;
         }
        ipc_bridge_matlab::biotac_sensors::BioTacTime::Cleanup(msg.bt_time);
        for (unsigned int i = 0; i < msg.bt_data_length; i++)
          ipc_bridge_matlab::biotac_sensors::BioTacData::Cleanup(msg.bt_data[i]);
        if (msg.bt_data != 0)
          {
            delete[] msg.bt_data;
            msg.bt_data_length = 0;
            msg.bt_data = 0;
          }
      }
    }
  }
}
#endif
