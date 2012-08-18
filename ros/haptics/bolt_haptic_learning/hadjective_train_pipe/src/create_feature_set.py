#!/usr/bin/env python
import roslib; roslib.load_manifest("language_train_pipe")
import rospy
import sys
import numpy as np
import cPickle

from bolt_pr2_motion_obj import BoltPR2MotionObj

def main():

    # Load the pickle file
    import pdb; pdb.set_trace()

    if len(sys.argv) < 2:
        raise Exception("Usage: %s [input_file]", sys.argv[0])

    input_filename = sys.argv[1]

    if not intput_filename.endswith(".pkl"):
        raise Exception("Input file is %s \nPlease pass in a pickle data file" % input_filename)

    file_ptr = open(open_filename, "r")

    all_data = cPickle.load(file_ptr)

    #




if __name__== "__main__":
    main()
