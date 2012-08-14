#!/usr/bin/env python
import roslib; roslib.load_manifest("language_train_pipe")
import rospy
import sys
import cPickle

def main():

    # Load the pickle file

    
    import pdb; pdb.set_trace()

    pickle_file_name = sys.argv[1]

    file_ptr = open(pickle_file_name, "r")

    all_data = cPickle.load(file_ptr)

if __name__== "__main__":
    main()
