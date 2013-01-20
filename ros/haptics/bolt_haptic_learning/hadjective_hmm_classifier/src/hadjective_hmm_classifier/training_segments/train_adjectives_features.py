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
import sklearn
from sklearn.metrics import f1_score

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

def alt_train_adjective_phase_classifier(path, adjective, all_features):
    """
    Example function on how to access all of the features
    stored in adjective_phase_set
    """

    # File name 
    dataset_file_name = "_".join(("trained", adjective))+".pkl"
    newpath = os.path.join(path, "trained_adjectives")
    path_name = os.path.join(newpath, dataset_file_name)
    
    if os.path.exists(path_name):
        print "File %s already exists, skipping it." % path_name
        return

    train_X = []

    for phase in phases:
        train_set = all_features[adjective][phase]['train']
        train_X.append(train_set['features'])
        train_Y = train_set['labels']
        object_ids = train_set['object_ids']

    train_X = np.concatenate(train_X, axis=1)

    print "Training adjective %s" % adjective

   #magic training happening here!!!
    scaler = sklearn.preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)    
    parameters = {'C': np.linspace(0.001,1e6,100),              
                  'penalty': ['l2','l1'],
                  'dual': [False],
                  'class_weight' : ('auto',),
                  }
    clf = sklearn.svm.LinearSVC()
    grid = sklearn.grid_search.GridSearchCV(clf, parameters,
                                            n_jobs=6,
                                            score_func=f1_score,
                                            verbose=0)
    grid.fit(train_X, train_Y)
    trained_clf = grid.best_estimator_
    #end of magic training!!!
    
    dataset = all_features[adjective]
    dataset['adjective'] = adjective
    dataset['classifier'] = trained_clf
    dataset['scaler'] = scaler
   
    # Save the results in the folder
    with open(path_name, "w") as f:
        print "Saving file: ", path_name
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
        
    test_X = []

    for phase in phases:
        test_set = all_features[adjective][phase]['test']
        test_X.append(test_set['features'])
        test_Y = test_set['labels']
        object_ids = test_set['object_ids']

    test_X = scaler.transform(np.concatenate(test_X, axis=1))
    train_score =  grid.best_score_
    test_score = f1_score(test_Y, trained_clf.predict(test_X))
    print "The training score is: %.2f, test score is %.2f" % (train_score, test_score)    
    

def orig_train_adjective_phase_classifier(path, adjective, all_features):
    """
    Example function on how to access all of the features
    stored in adjective_phase_set
    """

    # File name 
    dataset_file_name = "_".join(("trained", adjective))+".pkl"
    newpath = os.path.join(path, "trained_adjectives")
    path_name = os.path.join(newpath, dataset_file_name)
    
    if os.path.exists(path_name):
        print "File %s already exists, skipping it." % path_name
        return

    print "Creating adjective %s" % adjective

    train_X = []

    for phase in phases:
        train_set = all_features[adjective][phase]['train']
        train_X.append(train_set['features'])
        train_Y = train_set['labels']
        object_ids = train_set['object_ids']

    train_X = np.concatenate(train_X, axis=1)

    print "Training adjective %s" % adjective

    if True:
        trained_clf,scaler = utilities.train_svm_gridsearch(train_X = train_X,
                             train_Y = train_Y,
                             verbose=True,
                             object_ids = object_ids,
                             n_jobs = 6,
                             scale = True 
                             )
    else: 
        trained_clf = utilities.train_gradient_boost(train_X = train_X,
                                train_Y = train_Y,
                                object_ids = object_ids,
                                )
    
    dataset = all_features[adjective]
    dataset['adjective'] = adjective
    dataset['classifier'] = trained_clf
    dataset['scaler'] = scaler
   
    print "Saving trained_classifier" 

    # Save the results in the folder
    with open(path_name, "w") as f:
        print "Saving file: ", path_name
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)

train_adjective_phase_classifier = alt_train_adjective_phase_classifier

def main():
    if len(sys.argv) == 4:
        path, adjective, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the adjective %s for the phase %s" % (
                adjective)

        loaded_features = load_adjective_phase(path)
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(train_adjective_phase_classifier)(path, adjective, loaded_features))

    elif len(sys.argv) == 3:
        path, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the all adjectives"
        loaded_features = load_adjective_phase(path)
 
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(train_adjective_phase_classifier)(path, adjective, loaded_features) 
            for adjective in adjectives)
                                                      
    else:
        print "Usage:"
        print "%s path adjective n_jobs" % sys.argv[0]
        print "%s path n_jobs" % sys.argv[0]
        print "Path to the base directory"

if __name__=="__main__":
    main()
    print "done"        

