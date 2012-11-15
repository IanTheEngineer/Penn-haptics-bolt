#!/usr/bin/env python
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import numpy as np
import sys 
import os
from optparse import OptionParser
import cPickle
import bolt_learning_utilities as utilities
import extract_features as extract_features
import matplotlib.pyplot as plt 

from bolt_feature_obj import BoltFeatureObj


# Loads the data from h5 table and adds labels
# Returns the dictionary of objects
def loadDataFromH5File(input_file, output_file, adjective_file):

    # Takes the input h5 file and converts into bolt object data
    all_bolt_data = utilities.convertH5ToBoltObjFile(input_file, None, False);

    # Inserts adjectives into the bolt_data  
    all_bolt_data_adj = utilities.insertAdjectiveLabels(all_bolt_data, output_file, adjective_file, False)

    # Load pickle file
    pca_dict = cPickle.load(open('pca.pkl', 'r'))

    all_feature_obj = BoltMotionObjToFeatureObj(all_bolt_data_adj, pca_dict) 

    cPickle.dump(all_feature_obj, open('aluminum_bar_701_01.pkl', 'w')) 
    import pdb; pdb.set_trace()
    return all_feature_obj

# Takes the bolt data and extracts features to run
def BoltMotionObjToFeatureObj(all_bolt_data, electrode_pca_dict):
    """ 

    For each object - pull out features and store in feature_obj
    with the same structure as all_bolt_data
   
        Dictionary - "tap", "slide", "slow_slide", 
                     "thermal_hold", "squeeze"

    """

    # Store in feature class object
    all_features_obj_dict = dict();

    for motion_name in all_bolt_data:
        trial_list = all_bolt_data.get(motion_name)
        print motion_name

        feature_list = list()
        # For all objects
        for trial in trial_list:

            bolt_feature_obj = extract_features.extract_features(trial, electrode_pca_dict[motion_name])

            feature_list.append(bolt_feature_obj)

        # Store all of the objects away
        all_features_obj_dict[motion_name] = feature_list

    return all_features_obj_dict


# Parse the command line arguments
def parse_arguments():
    """
    Parses the arguments provided at command line.
    
    Returns:
    (input_file, adjective_file, range)
    """
    parser = OptionParser()
    parser.add_option("-i", "--input_file", action="store", type="string", dest = "in_h5_file")
    parser.add_option("-o", "--output", action="store", type="string", dest = "out_file", default = None)
    parser.add_option("-a", "--input_adjective", action="store", type="string", dest = "in_adjective_file")

    (options, args) = parser.parse_args()
    input_file = options.in_h5_file #this is required
    adjective_file = options.in_adjective_file
    output_file = options.out_file

    return (input_file, output_file, adjective_file)

if __name__ == "__main__":
    input_file, output_file, adjective_file = parse_arguments()
    loadDataFromH5File(input_file, output_file, adjective_file)
