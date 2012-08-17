#!/usr/bin/env python
import roslib; roslib.load_manifest("language_train_pipe")
import rospy
import numpy as np
import sys
import cPickle
import extract_features 
import matplotlib.pyplot as plt

def main():

    # Load the pickle file

    
    import pdb; pdb.set_trace()

    pickle_file_name = sys.argv[1]

    file_ptr = open(pickle_file_name, "r")

    all_data = cPickle.load(file_ptr)

    # For all of the motions in all_data
    for motion_name in all_data:
        motion_list = all_data.get(motion_name)
        
        # For all objects in each motion
        for motion in motion_list:
            print "object"


if __name__== "__main__":
    main()
