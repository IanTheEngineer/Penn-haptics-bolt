#! /usr/bin/python
import cPickle
import sklearn.grid_search
import sklearn.cross_validation
import hmm_chain
import os
import sys
import itertools
import utilities
from utilities import adjectives, phases, sensors, static_features
import multiprocessing
import tables
import traceback
import numpy as np
from static_feature_obj import StaticFeatureObj
import upenn_features
from collections import defaultdict
from sklearn.externals.joblib import Parallel, delayed


def get_train_test_objects(database, adjective):
    """
    Gets the objects in the test and train sets for the
    specified adjective 
    """

    if adjective not in adjectives:
        raise ValueError("%s is not a known adjective" % adjective)

    train_group = database.getNode("/train_test_sets", adjective).train
    test_group = database.getNode("/train_test_sets", adjective).test
    
    train_set_object_names = train_group._v_children
    test_set_object_names = test_group._v_children

    return train_set_object_names.keys(), test_set_object_names.keys()


def load_feature_objects(base_directory):
    """
    Given the base directory, will load all of the features associated with a 
    single phase and store it for selection based on object name
    """
    feat_obj = defaultdict(dict) # Store the objects by phase

    phase_dir = os.path.join(base_directory, "datasets")
    for f in os.listdir(phase_dir):
        # select pkl files associated with adjective
        if not f.endswith('.pkl'):
            continue
        
        # Load pickle file
        path_name = os.path.join(phase_dir, f)
        with open(path_name, "r") as file_path:
            features_objs = cPickle.load(file_path)

        chars = f.strip(".pkl").split("_")
        chars = chars[2:] # static_feature
        phase = "_".join(chars) # merge name together again
        feat_obj[phase] = features_objs[0] #only objects

    return feat_obj 


# Pull from the StaticFeatureObj and return a vector of features and labels
def createFeatureVector(static_feature_obj, feature_list):
    """ 
    Given a BoltFeatureObj and a list of features to pull, returns
    the features and labels in vector form for all adjectives

    feature_list: expects a list of strings containing the name of the
                  features to extract
                  
                  Ex. ["max_pdc", "centroid"]

    Returns: Vector of features (1 x feature length)
             Features - in order of feature_list

             Ex. max_pdc = 10, centroid = [14, 10]

             returned numpy array: [10 14 10]

    """
    # Store the features with their adjectives 
    all_feature_vector = list()
    
    # Pull out the features from the object
    for feature in feature_list:
        
        feature_vector = eval('static_feature_obj.'+ feature)
        # Make sure it is in list form 
        if isinstance(feature_vector, np.ndarray):
            feature_vector = feature_vector.tolist()

        if isinstance(feature_vector, float):
            feature_vector = [feature_vector]

        if feature == "electrode_polyfit":
            for idx in range(0,len(feature_vector[0])):
                all_feature_vector += [feature_vector[0][idx], feature_vector[1][idx]]
        else:
            all_feature_vector += feature_vector

    return np.array(all_feature_vector)


def create_feature_set(database, feature_dict, object_set, adjective):
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

    print "Building adjective %s" % adjective

    # For each object in the database, extract the phase and sensor
    # data for 
    for group in utilities.iterator_over_object_groups(database):
        # Pull data from h5 database
        data_dict = utilities.dict_from_h5_group(group)
        object_name = data_dict["name"]
        name = object_name.split('_') 

        # Skip over object if it is in the set
        # Training set will skip over test objects
        # and vice versa        
        if object_name in object_set:
            continue
 
#        print "Loading object ", object_name
       
        # Extract features
        feature_obj = feature_dict[object_name] 
        feature_vector = createFeatureVector(feature_obj, static_features)
        features.append(feature_vector)

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
    dataset_file_name = "_".join(("static_feature", adjective,phase))+".pkl"
    newpath = os.path.join(path, "adjective_phase_set")
    path_name = os.path.join(newpath, dataset_file_name)
    
    if os.path.exists(path_name):
        print "File %s already exists, skipping it." % path_name
        return

    print "Creating adjective %s and phase %s" % (adjective, phase)

    # Open database and get train/test split
    database = tables.openFile(database)
    train_objs, test_objs = get_train_test_objects(database, adjective)
   
    # Select the features from the feature objects 
    feature_set = load_feature_objects(path)

    # Store the train/test in a dataset
    # train set will skip over TEST objs and test set
    # skips over TRAIN objs
    dataset = defaultdict(dict)
    dataset['train'] = create_feature_set(database, feature_set[phase], test_objs, adjective)
    dataset['test'] = create_feature_set(database, feature_set[phase], train_objs, adjective)

    if len(dataset) is 0:
        print "Empty dataset???"
        return

    print "Saving dataset to file"

    #import pdb; pdb.set_trace()
    # Save the results in the folder
    with open(path_name, "w") as f:
        print "Saving file: ", path_name 
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
 

def main():
    """
    if len(sys.argv) == 6:
        database, path, adjective, phase, sensor = sys.argv[1:]
        train_single_dataset(database, path, adjective, phase, sensor)
    """ 
    if len(sys.argv) == 6:
        database, path, adjective, phase, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the adjectives %s and for phase %s" %(
            adjective, phase)
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(create_single_dataset)(database, path, adjective, phase))
   
    if len(sys.argv) == 5:
        database, path, adjective, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training all the phases for adjective %s" %(
                    adjective)
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(create_single_dataset)(database, path, adjective, phase) 
            for phase in itertools.product(phases))
            #    create_single_dataset(database, path, adjective, phase))

    elif len(sys.argv) == 4:
        database, path, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training all combinations of adjectives and phases"
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(create_single_dataset)(database, path, adjective, phase) 
        for adjective, phase in itertools.product(adjectives,
                                                  phases))
            #create_single_dataset(database, path, adjective, phase))
    else:
        print "Usage:"
        print "%s database path adjective phase n_jobs" % sys.argv[0]
        print "%s database path adjective n_jobs" % sys.argv[0]
        print "%s database path n_jobs" % sys.argv[0]
        print "Files will be saved in path/datasets"

if __name__ == "__main__":
    main()
    print "done"

