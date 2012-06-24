#ifndef __IPC_BRIDGE_MATLAB_BIOTAC_SENSORS_BIOTACDATA__
#define __IPC_BRIDGE_MATLAB_BIOTAC_SENSORS_BIOTACDATA__
#include <ipc_bridge_matlab/ipc_bridge_matlab.h>
#include <ipc_bridge/msgs/biotac_sensors_BioTacData.h>

namespace ipc_bridge_matlab
{
  namespace biotac_sensors
  {
    namespace BioTacData
    {
      static mxArray* ProcessMessage(const ipc_bridge::biotac_sensors::BioTacData &msg)
      {
        const char *fields[] = {"bt_serial", "bt_position", "tdc_data", "tac_data", "pdc_data", "pac_data", "electrode_data"};
        const int nfields = sizeof(fields)/sizeof(*fields);
        mxArray *out = mxCreateStructMatrix(1, 1, nfields, fields);

      if (msg.bt_serial == 0)
        {
          char buf[1] = "";
          mxSetField(out, 0, "bt_serial", mxCreateString(buf));
        }
      else
        {
          char buf[strlen(msg.bt_serial) + 1];
          strcpy(buf, msg.bt_serial);
          mxSetField(out, 0, "bt_serial", mxCreateString(buf));
        }


        mxSetField(out, 0, "bt_position", mxCreateDoubleScalar(msg.bt_position));
        mxSetField(out, 0, "tdc_data", mxCreateDoubleScalar(msg.tdc_data));
        mxSetField(out, 0, "tac_data", mxCreateDoubleScalar(msg.tac_data));
        mxSetField(out, 0, "pdc_data", mxCreateDoubleScalar(msg.pdc_data));

        mxArray *pac_data = mxCreateDoubleMatrix(1, 22, mxREAL);
        std::copy(msg.pac_data, msg.pac_data + 22, mxGetPr(pac_data));
        mxSetField(out, 0, "pac_data", pac_data);

        mxArray *electrode_data = mxCreateDoubleMatrix(1, 19, mxREAL);
        std::copy(msg.electrode_data, msg.electrode_data + 19, mxGetPr(electrode_data));
        mxSetField(out, 0, "electrode_data", electrode_data);

        return out;
      }

      static int ProcessArray(const mxArray *a, ipc_bridge::biotac_sensors::BioTacData &msg)
      {
        mxArray *field;

        field = mxGetField(a, 0, "bt_serial");
        int buflen = 128;
        char buf[buflen];
        mxGetString(field, buf, buflen);
        if (strlen(buf) > 0)
        {
          msg.bt_serial = new char[strlen(buf) + 1];
          strcpy(msg.bt_serial, buf);
        }

        field = mxGetField(a, 0, "bt_position");
        msg.bt_position = mxGetScalar(field);

        field = mxGetField(a, 0, "tdc_data");
        msg.tdc_data = mxGetScalar(field);

        field = mxGetField(a, 0, "tac_data");
        msg.tac_data = mxGetScalar(field);

        field = mxGetField(a, 0, "pdc_data");
        msg.pdc_data = mxGetScalar(field);

        field = mxGetField(a, 0, "pac_data");
        double tmp_pac_data[22];
        ipc_bridge_matlab::GetDoubleArray(field, 22, tmp_pac_data);
        for (int i = 0; i < 22; i++)
          msg.pac_data[i] = (int)tmp_pac_data[i];

        field = mxGetField(a, 0, "electrode_data");
        double tmp_electrode_data[19];
        ipc_bridge_matlab::GetDoubleArray(field, 19, tmp_electrode_data);
        for (int i = 0; i < 19; i++)
          msg.electrode_data[i] = (int)tmp_electrode_data[i];

        return SUCCESS;
      }


      static void Cleanup(ipc_bridge::biotac_sensors::BioTacData &msg)
      {
        if (msg.bt_serial != 0)
         {
           delete[] msg.bt_serial;
           msg.bt_serial = 0;
         }
      }
    }
  }
}
#endif
