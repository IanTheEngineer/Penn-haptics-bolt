#! /usr/bin/python
import roslib
roslib.load_manifest("biotac_log_parser")
import rospy
import rosbag
import numpy
import glob
import sys
import cPickle
from optparse import OptionParser
import itertools
import tables
from biotac_sensors.msg import BioTacHand

from collections import defaultdict



def main():
    if len(sys.argv) < 3:
        rospy.loginfo("Usage: %s [input_files] [output_files]", sys.argv[0])
        return
    
    input_filenames = sys.argv[1:-1]
    output_filename = sys.argv[-1]

    filters = tables.Filters(complevel=9)
    h5file = tables.openFile(output_filename, mode="w", title="Biotach Log",
                             filters = filters)

    for filename in input_filenames:

        rospy.loginfo("Opening bag %s"% filename)        
        bag = rosbag.Bag(filename)

        tdc_data = defaultdict(list)
        tac_data = defaultdict(list)
        pdc_data = defaultdict(list)
        pac_data = defaultdict(list)
        electrode_data = defaultdict(list)
        time_stamp = []

        num_entries = 0
        for _, msg, stamp in bag.read_messages(topics="/biotac_pub"):
            num_entries += 1
            isinstance(msg, BioTacHand)            
            
            num_fingers = len(msg.bt_data)
            for finger_index in xrange(num_fingers):                
                
                tdc_data[finger_index].append( msg.bt_data[finger_index].tdc_data)
                tac_data[finger_index].append( msg.bt_data[finger_index].tac_data)
                pdc_data[finger_index].append( msg.bt_data[finger_index].pdc_data)
                pac_data[finger_index].append( msg.bt_data[finger_index].pac_data)
                electrode_data[finger_index].append( msg.bt_data[finger_index].electrode_data)
                
            time_stamp.append( stamp.to_sec())

        
        #group_name = "trajectory_" + str(traj_num)        
        group_name = filename.partition(".")[0]
        
        #f[group_name + "/timestamps"] = time_stamp
        bag_group = h5file.createGroup("/", group_name)
        
        timestamps_carray = h5file.createCArray(bag_group, "timestamps", tables.Int64Atom(), (num_entries,)
                                              )
        timestamps_carray[:] = time_stamp
                
        for finger_index in xrange(num_fingers):
            
            #electrode_dsc = dict( ("electrode_" + str(i), tables.Int64Col()) for i in xrange(19))
            #pac_dsc = dict( ("pac_" + str(i), tables.Int64Col()) for i in xrange(22))            
            #single_finger_dsc = {"tdc": tables.Int64Col(),
                                 #"tac": tables.Int64Col(),
                                 #"pdc": tables.Int64Col(),
                                 
                                 ##"pac": pac_dsc,
                                 ##"electrode" : electrode_dsc
                                 #}
            
            finger_group = h5file.createGroup(bag_group, "finger_"+str(finger_index))
            
            electrode_carray = h5file.createCArray(finger_group, "electrodes", tables.Int64Atom(), (num_entries, 19))
            electrode_carray[:] = electrode_data[finger_index]
            
            pac_carray = h5file.createCArray(finger_group, "pac", tables.Int64Atom(), (num_entries, 22))
            pac_carray[:] = pac_data[finger_index]
            
            tdc_array = h5file.createCArray(finger_group, "tdc", tables.Int64Atom(), (num_entries,))
            tdc_array[:] = tdc_data[finger_index]
            
            tac_carray = h5file.createCArray(finger_group, "tac", tables.Int64Atom(), (num_entries,))
            tac_carray[:] = tac_data[finger_index]
            
            pdc_carray = h5file.createCArray(finger_group, "pdc", tables.Int64Atom(), (num_entries,))
            pdc_carray[:] = pdc_data[finger_index]
        

    h5file.close()


if __name__ == "__main__":
    main()
