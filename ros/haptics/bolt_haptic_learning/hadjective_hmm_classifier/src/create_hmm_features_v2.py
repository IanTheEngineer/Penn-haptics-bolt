#! /usr/bin/python
import cPickle
import hmm_chain
import os
import sys
import itertools
import utilities
from utilities import adjectives, phases, sensors, static_features
import tables
from collections import defaultdict
from sklearn.externals.joblib import Parallel, delayed
from extract_static_features import get_train_test_objects
from adjective_classifier import AdjectiveClassifier
import numpy as np

class PhaseClass:
    def __init__(self):
        self.phase=''
        self.path_name=''
        self.build=False
        self.features = []
        self.labels = []
        self.object_names = []
        self.object_ids = [] 
    def wipe_data(self):
        self.labels = []
        self.object_names = []
        self.object_ids = []
        self.features = []

def create_hmm_feature_set(database, object_set, adj_obj, phase_list):
    """ 
    For each object in the database, run classifier.extract_features. All the
    features are then collected in a matrix.
    If the classifier's adjective is among the objects' then the feature
    is labeled with 1, otherwise 0. 

    Parameters:
    database: either a string or an open pytables file.
        
    Returns the features and the labels as two 2-dimensional matrices.
    """
    print "Building adjective %s" % adj_obj.adjective

    # For each object in the database, extract the phase and sensor
    # data for 
    for group in utilities.iterator_over_object_groups(database):
        # Pull data from h5 database
        data_dict = utilities.dict_from_h5_group(group)
        object_name = data_dict["name"]
        name = object_name.split('_')
        labels = []
        # Skip over object if it is in the set
        # Training set will skip over test objects
        # and vice versa        
        if object_name in object_set:

            # Extract features
            feature_data = data_dict["data"]
            
            for i, phase_obj in enumerate(phase_list):
                scores = []
                set_dict = defaultdict(dict)
                if phase_obj.build == False:         
                    continue

                for sensor, data in feature_data[phase_obj.phase].iteritems():
                    try:
                        chain = adj_obj.chains[phase_obj.phase][sensor]
                        scores.append(chain.score(data))
                    except KeyError:
                        pass
                #import pdb; pdb.set_trace()
                phase_obj.features.append(scores)              
                # Sort out the objec's label
                if adj_obj.adjective in data_dict["adjectives"]:
                    phase_obj.labels.append(1)
                else:
                    phase_obj.labels.append(0)
                phase_obj.object_names.append(object_name)
                phase_obj.object_ids.append(int(name[-2]))
        
    #Iterate over all phases, convert to dictionaries and sqeeze
    #place all phases in a list
    set_dict_list = []
    for phase_obj in phase_list:
        set_dict = defaultdict(dict)
        if phase_obj.build == True:   
            set_dict['features'] = np.array(phase_obj.features).squeeze()
            set_dict['labels'] = np.array(phase_obj.labels).flatten()
            set_dict['object_names'] = np.array(phase_obj.object_names).flatten()
            set_dict['object_ids'] = np.array(phase_obj.object_ids).flatten()
            phase_obj.wipe_data()
            #import pdb; pdb.set_trace()
        set_dict_list.append(set_dict)
    return set_dict_list

def create_single_dataset(database, path, adj_obj):
    """ 
    Creates a pickle file dataset for each motions for all objects with a particular adjective
    """
    adjective = adj_obj.adjective
    #import pdb; pdb.set_trace()
    # Test to see if any phase files already exist and only create the ones that don't, or skip all
    all_phase_list = phases

    phase_list = []
    nobuild = 0
    for phase in all_phase_list:
        phase_obj = PhaseClass()
        phase_obj.phase = phase;
        dataset_file_name = "_".join(("hmm_feature", adjective,phase))+".pkl"
        phase_obj.path_name = os.path.join(path, dataset_file_name)
        
        if os.path.exists(phase_obj.path_name):
            print "File %s already exists, skipping it." % phase_obj.path_name
            phase_obj.build = False
            nobuild = nobuild + 1
        else:
            phase_obj.build = True
            print "Creating adjective %s and phase %s" % (adjective, phase)
        phase_list.append(phase_obj)
    if nobuild == 4:
        print "All phases of adjective %s are already built. Moving on..." % adjective
        return

    # Open database and get train/test split
    database = tables.openFile(database)
    train_objs, test_objs = get_train_test_objects(database, adjective)
    # Select the features from the feature objects 
    
    feature_train_dict_list = create_hmm_feature_set(database, train_objs, adj_obj, phase_list)
    feature_test_dict_list = create_hmm_feature_set(database, test_objs, adj_obj, phase_list)

    for i, phase_object in enumerate(phase_list):
        if phase_object.build == True:
            # Store the train/test in a dataset
            #import pdb; pdb.set_trace()
            dataset = defaultdict(dict)
            dataset['train'] = feature_train_dict_list[i] 
            dataset['test'] = feature_test_dict_list[i]

            if len(dataset) is 0:
                print "Empty dataset for adj %s and phase %s" % (adjective, phase_object.phase)
                continue 

            print "Saving dataset to file"

            #import pdb; pdb.set_trace()
            # Save the results in the folder
            #Saves one file per motion. This needs to be a for loop
            with open(phase_object.path_name, "w") as f:
                print "Saving file: ", phase_object.path_name
                cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)

#Check if directory exits & create it
def check_dir(f): 
    if not os.path.exists(f):
        os.makedirs(f)
        return True
    return False

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

    elif len(sys.argv) == 3:
        database, path = sys.argv[1:]
        #n_jobs = int(n_jobs)
        print "Training all combinations of adjectives and phases"
        #p = Parallel(n_jobs=n_jobs,verbose=10)
        #p(delayed(create_single_dataset)(database, path, adjective, phase)
        #for adjective, phase in itertools.product(adjectives,
        #                                          phases))
        base_directory = path
        untrained_directory = os.path.join(base_directory, "untrained_adjectives")
        hmm_feature_directory = os.path.join(base_directory, "adjective_phase_set")
        check_dir(hmm_feature_directory)
        for adj_f in os.listdir(untrained_directory):
            full_adj_path = os.path.join(untrained_directory, adj_f)
            adj_obj = cPickle.load(open(full_adj_path))
            assert isinstance(adj_obj, AdjectiveClassifier)
            create_single_dataset(database, hmm_feature_directory, adj_obj)
        #    create_single_dataset(database, path, adjective, "some_phase")
    else:
        print "Usage:"
        print "%s database path adjective phase n_jobs" % sys.argv[0]
        print "%s database path adjective n_jobs" % sys.argv[0]
        print "%s database path" % sys.argv[0]
        print "Files will be saved in path/adjective_phase_set"

if __name__ == "__main__":
    main()
    print "done"

