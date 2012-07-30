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
import h5py
from biotac_sensors.msg import BioTacHand

from collections import defaultdict

def main():
    """ Usage
    First log the data using rosbag:
    rosbag record /biotac_pub [other topics] -o file_prefix
    
    Many topics can be recorded, but this file only parses biotac msgs.
    The call rosrun rosrun biotac_log_parser parse_log_json.py -i bag_file -o output.hdf5
    
    Multiple bag files can be specified on the command line using wildcards, but they need to be
    enclosed in quotes:
    The call rosrun rosrun biotac_log_parser parse_log_json.py -i "*.bag" -o output.hdf5
    """    
    
    parser = OptionParser()
    parser.add_option("-i", "--input_file", dest="input_file",
                      help="input bag FILEs. Use wildcards for multiple files",
                      metavar="FILE", action="store")
    parser.add_option("-o", "--output_file", dest="output_file",
                      help="output FILE", metavar="FILE")

    (options, _) = parser.parse_args()

    if options.input_file is None:
        rospy.logerr("The input bag has to be specified")
        sys.exit()
    if options.output_file is None:
        rospy.logerr("The output file has to be specified")
        sys.exit()


    f = h5py.File(options.output_file, "w")

    gen = (glob.glob(f) for f in options.input_file.split())
    traj_num = 0

    for filename in itertools.chain.from_iterable(gen):

        rospy.loginfo("Opening bag %s"% filename)        
        bag = rosbag.Bag(filename)

        tdc_data = defaultdict(list)
        tac_data = defaultdict(list)
        pdc_data = defaultdict(list)
        pac_data = defaultdict(list)
        electrode_data = defaultdict(list)
        time_stamp = []

        num_fingers = 0
        for _, msg, _ in bag.read_messages(topics="/biotac_pub"):
            isinstance(msg, BioTacHand)            
            
            num_fingers = len(msg.bt_data)
            for finger_index in xrange(num_fingers):                
                
                tdc_data[finger_index].append( msg.bt_data[finger_index].tdc_data)
                tac_data[finger_index].append( msg.bt_data[finger_index].tac_data)
                pdc_data[finger_index].append( msg.bt_data[finger_index].pdc_data)
                pac_data[finger_index].append( msg.bt_data[finger_index].pac_data)
                electrode_data[finger_index].append( msg.bt_data[finger_index].electrode_data)
                
            time_stamp.append( msg.header.stamp.to_sec())

        
        #group_name = "trajectory_" + str(traj_num)        
        group_name = filename
        f[group_name + "/timestamps"] = time_stamp
        for finger_index in xrange(num_fingers):
            finger_name = "/finger_" + str(finger_index)
            f[group_name + finger_name + "/tdc_data"] = tdc_data[finger_index]
            f[group_name + finger_name + "/tac_data"] = tac_data[finger_index]
            f[group_name + finger_name + "/pdc_data"] = pdc_data[finger_index]
            f[group_name + finger_name + "/pac_data"] = pac_data[finger_index]
            f[group_name + finger_name + "/electrode_data"] = electrode_data[finger_index]

        traj_num += 1
    f.close()


if __name__ == "__main__":
    main()
