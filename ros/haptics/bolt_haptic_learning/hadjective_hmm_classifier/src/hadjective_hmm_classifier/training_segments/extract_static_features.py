#! /usr/bin/python
import cPickle
import sklearn.grid_search
import sklearn.cross_validation
import hmm_chain
import os
import sys
import itertools
import utilities
from utilities import adjectives, phases, sensors
import multiprocessing
import tables
import traceback
import numpy as np
from static_feature_obj import StaticFeatureObj
import upenn_features

def load_dataset(database, adjective, phase, sensor):
    """Loads the data from a dataset corresponding to an adjective, phase and
    sensor."""

    if adjective not in adjectives:
        raise ValueError("%s is not a known adjective" % adjective)
    if phase not in phases:
        raise ValueError("%s is not a known phase" % phase)
    if sensor not in sensors:
        raise ValueError("%s is not a known sensor" % sensor)

    train_group = database.getNode("/train_test_sets", adjective).train
    train_set = [utilities.dict_from_h5_group(g, [phase], [sensor])["data"][phase][sensor]
                    for g in train_group._v_children.values()]
    test_group = database.getNode("/train_test_sets", adjective).test
    test_set = [utilities.dict_from_h5_group(g, [phase], [sensor])["data"][phase][sensor]
                    for g in test_group._v_children.values()]

    return train_set, test_set

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

    # TDC - currently not included

    # electrodes
    electrode_features = upenn_features.electrodes_features(data_dict['electrodes'], norm_dict['electrodes'])
    features.electrode_polyfit = electrode_features
    feature_array.append(electrode_features)

    return features, np.hstack((pdc_features, pac_features, tac_features, electrode_features)) 

def create_feature_object_set(database, phase):
  
    feature_objs = [] 
    norm_phase = "MOVE_ARM_START_POSITION"

    # For each object in the database, extract the phase and sensor
    # data for 
    for group in utilities.iterator_over_object_groups(database):
       
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

    return feature_objs
 
def create_feature_set(database, adjective, phase):
    """
    For each object in the database, run classifier.extract_features. All the
    features are then collected in a matrix.
    If the classifier's adjective is among the objects' then the feature
    is labeled with 1, otherwise 0. 

    Parameters:
    database: either a string or an open pytables file.
        
    Returns the features and the labels as two 2-dimensional matrices.
    """
    labels = []
    features = []
    data_store = []

    norm_phase = "MOVE_ARM_START_POSITION"

    # For each object in the database, extract the phase and sensor
    # data for 
    for group in utilities.iterator_over_object_groups(database):
       
        import pdb; pdb.set_trace() 
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
 
        import pdb; pdb.set_trace() 
                 
        # Extract features from the dictionary here 
        features.append(feats)

        # Store off the labels here  
        if adjective in data_dict["adjectives"]:
            labels.append(1)
        else:
            labels.append(0)

    features = np.array(features).squeeze()
    labels = np.array(labels).flatten()
    
    return features, labels



def create_single_dataset(database, path, adjective, phase):
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
    dataset = create_feature_object_set(database, phase)
    #dataset = create_feature_set(database, adjective, phase)
    #dataset = load_dataset(database, adjective, phase, sensor)

    if len(dataset) is 0:
        print "Empty dataset???"
        return

    print "Saving dataset to file"

    import pdb; pdb.set_trace()
    # Save the results in the folder
    with open(path_name, "w") as f:
        print "Saving file: ", path_name 
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
 

def main():
    """
    if len(sys.argv) == 6:
        database, path, adjective, phase, sensor = sys.argv[1:]
        train_single_dataset(database, path, adjective, phase, sensor)
    elif len(sys.argv) == 5:
        database, path, phase, sensor = sys.argv[1:]
        print "Training all the adjectives for phase %s and sensor %s" %(
            phase, sensor)
        for adjective in adjectives:
            train_single_dataset(database, path, adjective, phase, sensor)
    """
    if len(sys.argv) == 4:
        database, path, adjective = sys.argv[1:]
        print "Training all the phases and sensors for adjective %s" %(
                    adjective)
        for phase, sensor in itertools.product(phases):
            train_single_dataset(database, path, adjective, phase)
    elif len(sys.argv) == 3:
        database, path = sys.argv[1:]
        print "Training all combinations of adjectives, phases and sensor"
 #       for adjective, phase, sensor in itertools.product(adjectives,
        for adjective, phase in itertools.product(adjectives,
                                                  phases):

            create_single_dataset(database, path, adjective, phase)
    else:
        print "Usage:"
        print "%s database path adjective phase sensor" % sys.argv[0]
        print "%s database path phase sensor" % sys.argv[0]
        print "%s database path adjective" % sys.argv[0]
        print "%s database path" % sys.argv[0]
        print "Files will be saved in path/datasets"

if __name__ == "__main__":
    main()
    print "done"

