#!/usr/bin/env python

# Script to start loading data from pytables with labels into the BoltPR2Motion Object pkl file 
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import sys 
import tables
import cPickle
from optparse import OptionParser

from bolt_pr2_motion_obj import BoltPR2MotionObj


def populate_BoltPR2MotionObj(all_data, out_file, adjective_file, save_to_file):
    
    # Checks the h5 table    
    if not adjective_file.endswith(".h5"):
        raise Exception("Adjective file is %s \nPlease pass in a hdf5 data file" % adjective_file)

    # Open the h5 file and gets the root and columns
    adjective_data = tables.openFile(adjective_file)
    adjective_root = adjective_data.root
    column_names = adjective_root.classes.colnames

    all_object_label_mapping = dict()
    # Loops through the adjective table and stores into a dictionary
    for row_num in xrange(adjective_root.classes.nrows):
        
        # Pulls out the row from the h5table 
        row = adjective_root.classes[row_num]
        object_mapping = dict()

        object_id = "default"
        # Store off the row into a dictionary
        for col_name in column_names:
            if col_name == "object_id":
                object_id = row[col_name]
            else:
                object_mapping[col_name] = row[col_name]

        # Store off the dictionary in the final dictionary
        all_object_label_mapping[object_id] = object_mapping


    # For all motions in all_data
    for motion_name in all_data:
        motion_list = all_data.get(motion_name)

        print motion_name
        # For all objects
        for motion in motion_list:
            
            # Pull object number from file
            object_name = motion.name
            object_name_split = object_name.split('_')
            object_id = object_name_split[-1]

            # Look up the object number in the h5 table
            motion.labels = all_object_label_mapping[object_id]
   
    # Stores the output in a pickle file
    if (save_to_file):
        file_ptr = open(out_file, "w")
        cPickle.dump(all_data, file_ptr, cPickle.HIGHEST_PROTOCOL)
        file_ptr.close()

def load_files(input_file):
    
    # Opens and loads pickle file with BoltPR2MotionObj
    if not input_file.endswith(".pkl"):
        raise Exception("Input BoltPR2Motion file %s is not a pkl file\nPlease pass in pkl data file" % input_file)

    object_file = open(input_file, "r") 
    all_data = cPickle.load(object_file)
    
    return all_data

def parse_arguments():
    """Parses the arguments provided at command line.
    
    Returns:
    (input_file, output_file, range)
    """
    parser = OptionParser()
    parser.add_option("-i", "--input_pickle", action="store", type="string", dest = "in_pickle_file")
    parser.add_option("-o", "--output", action="store", type="string", dest = "out_file", default = None)
    parser.add_option("-a", "--input_adjective", action="store", type="string", dest = "in_adjective_file")

    (options, args) = parser.parse_args()
    input_file = options.in_pickle_file #this is required
    
    if options.out_file is None:
        (_, name) = os.path.split(input_file)
        name = name.split(".")[0]
        out_file = name + ".pkl"
    else:        
        out_file = options.out_file
        if len(out_file.split(".")) == 1:
            out_file = out_file + ".pkl"
    
    adjective_file = options.in_adjective_file

    return input_file, out_file, adjective_file



if __name__ == "__main__":
    input_file, out_file, adjective_file = parse_arguments()
    bolt_obj_data = load_files(input_file)
    populate_BoltPR2MotionObj(bolt_obj_data, out_file, adjective_file, True)     

