#! /usr/bin/python
import roslib
roslib.load_manifest("bolt_data_parser")
import rospy
import rosbag
import numpy
import glob
import sys
import cPickle
import os
from optparse import OptionParser
import itertools
import tables
from biotac_sensors.msg import BioTacHand
from pr2_gripper_accelerometer.msg import PR2GripperAccelerometerData 

from collections import defaultdict



def main():
    if len(sys.argv) < 3:
        rospy.loginfo("Usage: %s [input_files] [output_files]", sys.argv[0])
        return
    
    input_filenames = sys.argv[1:-1]
    output_filename = sys.argv[-1]

    if not output_filename.endswith(".h5"):
        rospy.loginfo("Filename %s is not a h5 file" % output_filename)
        sys.exit()

    filters = tables.Filters(complevel=9)
    h5file = tables.openFile(output_filename, mode="w", title="Aggregated BOLT Data",
                             filters = filters)

    for filename in input_filenames:

        rospy.loginfo("Opening bag %s"% filename)        
        bag = rosbag.Bag(filename)

        # Store Biotac Data
        tdc_data = defaultdict(list)
        tac_data = defaultdict(list)
        pdc_data = defaultdict(list)
        pac_data = defaultdict(list)
        electrode_data = defaultdict(list)
        time_stamp = []

        # Store Gripper Accelerometer Data Msgs
        gripper_accelerometer_msg = PR2GripperAccelerometerData() 
        gripper_accelerometer_timestamp = []
        accelerometer_data = []
        gripper_position = []
        gripper_joint_effort = []
        gripper_velocity = []

        # Store gripper controller state
        current_state = 0 
        control_state = []
        current_detail_state = "DISABLED" 
        detail_state = []

        # Number of entries
        num_entries = 0
        num_biotac_entries = 0
        num_controller_state_entries = 0
        num_gripper_accelerometer_entries = 0

        for topic, msg, stamp in bag.read_messages(topics=["/biotac_pub", "/pr2_gripper_accelerometer/data", "/simple_gripper_controller_state", "/simple_gripper_controller_state_detailed"]):
            num_entries += 1
            # Currently downsampling the accelerometers by storing them off when they arrive
            # and then writing to the array at the same frequency at the biotacs
            if msg._type == 'pr2_gripper_accelerometer/PR2GripperAccelerometerData':
                num_gripper_accelerometer_entries += 1
                gripper_accelerometer_msg = msg
                gripper_accelerometer_timestamp.append(stamp.to_sec())

            if topic == '/simple_gripper_controller_state':
                num_controller_state_entries += 1
                current_state = msg.data

            if topic == '/simple_gripper_controller_state_detailed':
                current_detail_state = msg.data

            if msg._type == 'biotac_sensors/BioTacHand': 
                num_biotac_entries += 1          
                num_fingers = len(msg.bt_data)
                for finger_index in xrange(num_fingers):                
                
                    tdc_data[finger_index].append( msg.bt_data[finger_index].tdc_data)
                    tac_data[finger_index].append( msg.bt_data[finger_index].tac_data)
                    pdc_data[finger_index].append( msg.bt_data[finger_index].pdc_data)
                    pac_data[finger_index].append( msg.bt_data[finger_index].pac_data)
                    electrode_data[finger_index].append( msg.bt_data[finger_index].electrode_data)
                # Store accelerometer
                accel_store = [] 
                accel_store.append(gripper_accelerometer_msg.acc_x_raw)
                accel_store.append(gripper_accelerometer_msg.acc_y_raw)
                accel_store.append(gripper_accelerometer_msg.acc_z_raw)
                accelerometer_data.append(accel_store);

                # Store gripper
                gripper_position.append(gripper_accelerometer_msg.gripper_joint_position)
                gripper_velocity.append(gripper_accelerometer_msg.gripper_joint_velocity)
                gripper_joint_effort.append(gripper_accelerometer_msg.gripper_joint_effort)

                # Store control state
                control_state.append(current_state)

                # Store control detail state
                detail_state.append(current_detail_state)

            # Store time stamps for entire run 
            time_stamp.append( stamp.to_sec())

        #group_name = "trajectory_" + str(traj_num)        
        group_name = filename.partition(".")[0]
        group_name = os.path.split(group_name)[1] #remove path and trailing /
        
        #f[group_name + "/timestamps"] = time_stamp
        bag_group = h5file.createGroup("/", group_name)
        
        timestamps_carray = h5file.createCArray(bag_group, "timestamps", tables.Int64Atom(), (num_entries,)
                                              )
        timestamps_carray[:] = time_stamp

        # Create biotac group
        biotac_group = h5file.createGroup(bag_group, "biotacs")

        # Store information for biotacs        
        for finger_index in xrange(num_fingers):
            
            #electrode_dsc = dict( ("electrode_" + str(i), tables.Int64Col()) for i in xrange(19))
            #pac_dsc = dict( ("pac_" + str(i), tables.Int64Col()) for i in xrange(22))            
            #single_finger_dsc = {"tdc": tables.Int64Col(),
                                 #"tac": tables.Int64Col(),
                                 #"pdc": tables.Int64Col(),
                                 
                                 ##"pac": pac_dsc,
                                 ##"electrode" : electrode_dsc
                                 #}
            
            finger_group = h5file.createGroup(biotac_group, "finger_"+str(finger_index))
            
            electrode_carray = h5file.createCArray(finger_group, "electrodes", tables.Int64Atom(), (num_biotac_entries, 19))
            #import pdb; pdb.set_trace() 

            electrode_carray[:] = electrode_data[finger_index]
            
            pac_carray = h5file.createCArray(finger_group, "pac", tables.Int64Atom(), (num_biotac_entries, 22))
            pac_carray[:] = pac_data[finger_index]
            
            tdc_array = h5file.createCArray(finger_group, "tdc", tables.Int64Atom(), (num_biotac_entries,))
            tdc_array[:] = tdc_data[finger_index]
            
            tac_carray = h5file.createCArray(finger_group, "tac", tables.Int64Atom(), (num_biotac_entries,))
            tac_carray[:] = tac_data[finger_index]
            
            pdc_carray = h5file.createCArray(finger_group, "pdc", tables.Int64Atom(), (num_biotac_entries,))
            pdc_carray[:] = pdc_data[finger_index]
        
      # Store accelerometer information
        accel_carray = h5file.createCArray(bag_group, "accelerometer", tables.Float64Atom(), (num_biotac_entries, 3))

        accel_carray[:] = accelerometer_data

      # Store gripper information
        gripper_group = h5file.createGroup(bag_group, "gripper_aperture")

        gripper_position_carray = h5file.createCArray(gripper_group, "joint_position", tables.Float64Atom(), (num_biotac_entries,))
        gripper_position_carray[:] = gripper_position
        
        gripper_velocity_carray = h5file.createCArray(gripper_group, "joint_velocity", tables.Float64Atom(), (num_biotac_entries,))
        gripper_velocity_carray[:] = gripper_velocity

        gripper_joint_effort_carray = h5file.createCArray(gripper_group, "joint_effort", tables.Float64Atom(), (num_biotac_entries,))
        gripper_joint_effort_carray[:] = gripper_joint_effort
        #import pdb; pdb.set_trace()
      
        state_group = h5file.createGroup(bag_group, "state")

        #Store controller state
        control_state_carray = h5file.createCArray(state_group, "controller_state", tables.Int64Atom(), (num_biotac_entries,))
        control_state_carray[:] = control_state 
       
        # Store controller detailed state
        control_detail_carray = h5file.createCArray(state_group, "controller_detail_state", tables.StringAtom(itemsize=30), (num_biotac_entries,))
        control_detail_carray[:] = detail_state
        
    h5file.close()


if __name__ == "__main__":
    main()
