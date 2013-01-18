#! /usr/bin/python
import cPickle
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


def load_adjective_phase(base_directory):

    adjective_dir = os.path.join(base_directory, "adjective_phase_set")
    all_features = defaultdict(dict)

    for f in os.listdir(adjective_dir):
        # select pkl files associated with adjective
        if not f.endswith('.pkl'):
            continue
        
        # Load pickle file
        path_name = os.path.join(adjective_dir, f)
        with open(path_name, "r") as file_path:
            features = cPickle.load(file_path)

        chars = f.strip(".pkl").split("_")
        chars = chars[2:] #static_feature
        adjective = chars[0] #adjective
        chars = chars[1:] #adjective
        phase = "_".join(chars) # merge together
        all_features[adjective][phase] = features

    return all_features

def train_adjective_phase_classifier(path, adjective, phase, all_features):
    """
    Example function on how to access all of the features
    stored in adjective_phase_set
    """

    # File name 
    dataset_file_name = "_".join(("trained", adjective, phase))+".pkl"
    newpath = os.path.join(path, "trained_adjective_phase")
    path_name = os.path.join(newpath, dataset_file_name)
    
    if os.path.exists(path_name):
        print "File %s already exists, skipping it." % path_name
        return

    print "Creating adjective %s and phase %s" % (adjective, phase)

    train_set = all_features[adjective][phase]['train']
    train_X = train_set['features']
    train_Y = train_set['labels']
    object_ids = train_set['object_ids']

    print "Training adjective %s and phase %s" %(adjective, phase)
    trained_clf = utilities.train_svm_gridsearch(train_X = train_X,
                         train_Y = train_Y,
                         verbose=True,
                         object_ids = object_ids,
                         n_jobs = 6 
                         )

    dataset = all_features[adjective][phase]
    dataset['classifier'] = trained_clf
   
    print "Saving trained_classifier" 

    # Save the results in the folder
    with open(path_name, "w") as f:
        print "Saving file: ", path_name
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)

def main():
    if len(sys.argv) == 5:
        path, adjective, phase, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the adjective %s for the phase %s" % (
                adjective, phase)

        loaded_features = load_adjective_phase(path)
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(train_adjective_phase_classifier)(path, adjective, phase, loaded_features))

    elif len(sys.argv) == 4:
        path, adjective, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the adjective %s" % adjective
        loaded_features = load_adjective_phase(path)
 
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(train_adjective_phase_classifier)(path, adjective, phase, loaded_features) 
            for phase in itertools.product(phases))
 
    elif len(sys.argv) == 3:
        path, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the all combinations of adjectives and phases"
        loaded_features = load_adjective_phase(path)
 
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(train_adjective_phase_classifier)(path, adjective, phase, loaded_features) 
            for adjective, phase in itertools.product(adjectives,
                                                      phases))
    else:
        print "Usage:"
        print "%s path adjective phase n_jobs" % sys.argv[0]
        print "%s path adjective n_jobs" % sys.argv[0]
        print "%s path n_jobs" % sys.argv[0]
        print "Path to the base directory"

if __name__=="__main__":
    main()
    print "done"        
