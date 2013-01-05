#! /usr/bin/python
import cPickle
import sklearn.grid_search
import sklearn.cross_validation
import hmm_chain
import os
import sys
import itertools
import utilities
from utilities import adjectives, phases, sensors, static_channels
import multiprocessing
import tables
import traceback
import numpy as np
from static_feature_obj import StaticFeatureObj
import upenn_features

def extract_static_features(data_dict, norm_dict):
   
    features = StaticFeatureObj()
    feature_array = []
 
    # PDC
    pdc_features = upenn_features.pdc_features(data_dict['pdc'], norm_dict['pdc'])
    features.pdc_area = pdc_features[0:2]
    features.pdc_max = pdc_features[2:4]
    features.pdc_rise_count = pdc_features[4:6]
    feature_array.append(pdc_features)

    # PAC
    pac_features = upenn_features.pac_features(data_dict['pac'], norm_dict['pac'])
    features.pac_energy = np.array((pac_features[0],pac_features[5]))
    features.pac_sc= np.array((pac_features[1],pac_features[6]))
    features.pac_sv = np.array((pac_features[2],pac_features[7]))
    features.pac_ss= np.array((pac_features[3],pac_features[8]))
    features.pac_sk = np.array((pac_features[4],pac_features[9]))
    feature_array.append(pac_features)
     
    # TAC
    tac_features = upenn_features.tac_features(data_dict['tac'], norm_dict['tac'])
    features.tac_area = tac_features
    feature_array.append(tac_features)

    # TDC 
    tdc_features = upenn_features.tdc_features(data_dict['tdc'], norm_dict['tdc'])
    features.tdc_exp_fit = tdc_features
 
    # electrodes
    electrode_features = upenn_features.electrodes_features(data_dict['electrodes'], norm_dict['electrodes'])
    polyfit = [] 
    polyfit.append(electrode_features[0:12])
    polyfit.append(electrode_features[12:24])
 
    features.electrode_polyfit = np.array(polyfit)
    feature_array.append(electrode_features)

    # Gripper
    gripper_features = upenn_features.gripper_features(data_dict['gripper_aperture'])
    features.gripper_min = gripper_features[0]
    features.gripper_mean = gripper_features[1]
 
    # Transform 
    transform_features = upenn_features.transform_features(data_dict['transforms'])
    features.transform_distance = transform_features

    return features, np.hstack((pdc_features, pac_features, tac_features,tdc_features, electrode_features, gripper_features, transform_features)) 

def create_feature_object_set(database, phase):
  
    feature_objs = []
    feature_vect = [] 
    norm_phase = "MOVE_ARM_START_POSITION"

    # For each object in the database, extract the phase and sensor
    # data for 
    #temp = [g for g in utilities.iterator_over_object_groups(database)] 
    for group in utilities.iterator_over_object_groups(database):
    #for group in temp[0:2]:   
        # Pull data from h5 database
        data_dict = utilities.dict_from_h5_group(group, [phase])
        norm_dict = utilities.dict_from_h5_group(group, [norm_phase])
        data = data_dict["data"][phase]
        norm_data = norm_dict["data"][norm_phase]
        object_name = data_dict["name"]
        name = object_name.split('_') 
    
        print "Loading object ", object_name
       
        # Extract features
        static_feature_phase, feats = extract_static_features(data, norm_data)
        # Store information about object
        static_feature_phase.labels = data_dict["adjectives"]
        static_feature_phase.name = object_name
        static_feature_phase.detailed_state = phase
        static_feature_phase.object_id = int(name[-2])
        static_feature_phase.run_num = int(name[-1])

        feature_objs.append(static_feature_phase)
        feature_vect.append(feats)

    return feature_objs, np.array(feature_vect)
 
def create_single_dataset(database, path, phase):
    """ 
    Creates a single dataset with the adjectives, phases, and sensor
    """
   
    # File name 
    dataset_file_name = "_".join(("static_feature", phase))+".pkl"
    newpath = os.path.join(path, "datasets")
    path_name = os.path.join(newpath, dataset_file_name)
    
    if os.path.exists(path_name):
        print "File %s already exists, skipping it." % path_name
        return

    print "Creating phase %s" % phase

    database = tables.openFile(database)
    feature_objs, feature_vect = create_feature_object_set(database, phase)
    dataset = (feature_objs, feature_vect)

    if len(dataset) is 0:
        print "Empty dataset???"
        return

    print "Saving dataset to file"

    # Save the results in the folder
    with open(path_name, "w") as f:
        print "Saving file: ", path_name
        print " " 
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
 

def main():
    if len(sys.argv) == 3:
        database, path = sys.argv[1:]
        print "Training all phases and sensor"
        for phase in phases:
            create_single_dataset(database, path, phase)
    else:
        print "Usage:"
        print "%s database path" % sys.argv[0]
        print "Files will be saved in path/datasets"

if __name__ == "__main__":
    main()
    print "done"

