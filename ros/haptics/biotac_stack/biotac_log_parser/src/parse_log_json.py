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
import rosjson_time
from biotac_sensors.msg import BioTacHand

def main():
    """ Usage
    First log the data using rosbag:
    rosbag record /biotac_pub [other topics] -o file_prefix
    
    Many topics can be recorded, but this file only parses biotac msgs.
    The call rosrun rosrun biotac_log_parser parse_log_json.py -i bag_file -o output.json
    
    Multiple bag files can be specified on the command line using wildcards, but they need to be
    enclosed in quotes:
    The call rosrun rosrun biotac_log_parser parse_log_json.py -i "*.bag" -o output.json
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


    output_file = open(options.output_file, "w")
    

    gen = (glob.glob(f) for f in options.input_file.split())
    frame_count = 0
    elements = []
    for filename in itertools.chain.from_iterable(gen):

        rospy.loginfo("Opening bag %s"% filename)
        bag = rosbag.Bag(filename)
        
        for _, msg, _ in bag.read_messages(topics="/biotac_pub"):
            isinstance(msg, BioTacHand)
            
            msg.header.frame_id = frame_count
            toWrite = rosjson_time.ros_message_to_json(msg) + '\n'
            elements.append(toWrite)
            
            frame_count +=1
    
    output_file.write("[\n")        
    output_file.write( ",".join(elements))
    output_file.write("]\n")
    output_file.close()


if __name__ == "__main__":
    main()
