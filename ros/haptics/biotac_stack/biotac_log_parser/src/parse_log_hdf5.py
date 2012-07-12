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

        tdc_data = []
        tac_data = []
        pdc_data = []
        pac_data = []
        electrode_data = []
        time_stamp = []
        
        
        for _, msg, _ in bag.read_messages(topics="/biotac_pub"):
            isinstance(msg, BioTacHand)
            
            tdc_data.append( msg.bt_data[0].tdc_data)
            tac_data.append( msg.bt_data[0].tac_data)
            pdc_data.append( msg.bt_data[0].pdc_data)
            pac_data.append( msg.bt_data[0].pac_data)
            electrode_data.append( msg.bt_data[0].electrode_data)
            
            time_stamp.append( msg.header.stamp.to_sec())
            


        group_name = "trajectory_" + str(traj_num)
        f[group_name + "/tdc_data"] = tdc_data
        f[group_name + "/tac_data"] = tac_data
        f[group_name + "/pdc_data"] = pdc_data
        f[group_name + "/pac_data"] = pac_data
        f[group_name + "/electrode_data"] = electrode_data

        traj_num += 1
    f.close()


if __name__ == "__main__":
    main()
