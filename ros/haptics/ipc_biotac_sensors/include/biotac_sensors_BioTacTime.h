#ifndef __IPC_BRIDGE_MATLAB_BIOTAC_SENSORS_BIOTACTIME__
#define __IPC_BRIDGE_MATLAB_BIOTAC_SENSORS_BIOTACTIME__
#include <ipc_bridge_matlab/ipc_bridge_matlab.h>
#include <ipc_bridge/msgs/biotac_sensors_BioTacTime.h>

namespace ipc_bridge_matlab
{
  namespace biotac_sensors
  {
    namespace BioTacTime
    {
      static mxArray* ProcessMessage(const ipc_bridge::biotac_sensors::BioTacTime &msg)
      {
        const char *fields[] = {"frame_start_time", "frame_end_time", "tdc_ns_offset", "tac_ns_offset", "pdc_ns_offset", "pac_ns_offset", "electrode_ns_offset"};
        const int nfields = sizeof(fields)/sizeof(*fields);
        mxArray *out = mxCreateStructMatrix(1, 1, nfields, fields);


        mxSetField(out, 0, "frame_start_time", mxCreateDoubleScalar(msg.frame_start_time));
        mxSetField(out, 0, "frame_end_time", mxCreateDoubleScalar(msg.frame_end_time));
        mxSetField(out, 0, "tdc_ns_offset", mxCreateDoubleScalar(msg.tdc_ns_offset));
        mxSetField(out, 0, "tac_ns_offset", mxCreateDoubleScalar(msg.tac_ns_offset));
        mxSetField(out, 0, "pdc_ns_offset", mxCreateDoubleScalar(msg.pdc_ns_offset));

        mxArray *pac_ns_offset = mxCreateDoubleMatrix(1, 22, mxREAL);
        std::copy(msg.pac_ns_offset, msg.pac_ns_offset + 22, mxGetPr(pac_ns_offset));
        mxSetField(out, 0, "pac_ns_offset", pac_ns_offset);

        mxArray *electrode_ns_offset = mxCreateDoubleMatrix(1, 19, mxREAL);
        std::copy(msg.electrode_ns_offset, msg.electrode_ns_offset + 19, mxGetPr(electrode_ns_offset));
        mxSetField(out, 0, "electrode_ns_offset", electrode_ns_offset);

        return out;
      }

      static int ProcessArray(const mxArray *a, ipc_bridge::biotac_sensors::BioTacTime &msg)
      {
        mxArray *field;

        field = mxGetField(a, 0, "frame_start_time");
        msg.frame_start_time = mxGetScalar(field);

        field = mxGetField(a, 0, "frame_end_time");
        msg.frame_end_time = mxGetScalar(field);

        field = mxGetField(a, 0, "tdc_ns_offset");
        msg.tdc_ns_offset = mxGetScalar(field);

        field = mxGetField(a, 0, "tac_ns_offset");
        msg.tac_ns_offset = mxGetScalar(field);

        field = mxGetField(a, 0, "pdc_ns_offset");
        msg.pdc_ns_offset = mxGetScalar(field);

        field = mxGetField(a, 0, "pac_ns_offset");
        double tmp_pac_ns_offset[22];
        ipc_bridge_matlab::GetDoubleArray(field, 22, tmp_pac_ns_offset);
        for (int i = 0; i < 22; i++)
          msg.pac_ns_offset[i] = (int)tmp_pac_ns_offset[i];

        field = mxGetField(a, 0, "electrode_ns_offset");
        double tmp_electrode_ns_offset[19];
        ipc_bridge_matlab::GetDoubleArray(field, 19, tmp_electrode_ns_offset);
        for (int i = 0; i < 19; i++)
          msg.electrode_ns_offset[i] = (int)tmp_electrode_ns_offset[i];

        return SUCCESS;
      }


      static void Cleanup(ipc_bridge::biotac_sensors::BioTacTime &msg)
      {
      }
    }
  }
}
#endif
