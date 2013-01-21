#! /usr/bin/python
import cPickle
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
from collections import defaultdict
import sklearn
from sklearn.externals.joblib import Parallel, delayed
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

def collapse_all_phases(all_features, adjective, phases, train_test = 'train'):
    X = []
    
    for phase in phases:
        train_set = all_features[adjective][phase][train_test]
        X.append(train_set['features'])
        Y = train_set['labels']
        object_ids = train_set['object_ids']

    X = np.concatenate(X, axis=1)
    return X, Y

def train_combined_adjectives(path, adjective, static_features, dynamic_features,
                              n_jobs):
    """Combines all the features and then perform training
    """
    # File name 
    dataset_file_name = "_".join(("trained", adjective))+".pkl"
    newpath = os.path.join(path, "trained_adjectives")
    path_name = os.path.join(newpath, dataset_file_name)
    
    if os.path.exists(path_name):
        print "File %s already exists, skipping it." % path_name
        return
    
    static_X , train_Y = collapse_all_phases(static_features, adjective, phases)
    dyn_X, _=  collapse_all_phases(dynamic_features, adjective, phases)
    
    train_X = np.hstack((static_X, dyn_X))
    
    print "Training adjective %s" % adjective
    print "Size of training set: ", train_X.shape
    
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
                                            n_jobs=n_jobs,
                                            score_func=f1_score,
                                            verbose=0)
    grid.fit(train_X, train_Y)
    trained_clf = grid.best_estimator_
    #end of magic training!!!
    
    #dataset = all_features[adjective]
    dataset = {}
    dataset['adjective'] = adjective
    dataset['classifier'] = trained_clf
    dataset['scaler'] = scaler
   
    # Save the results in the folder
    with open(path_name, "w") as f:
        print "Saving file: ", path_name
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
        
    test_X = []
    
    static_X , test_Y = collapse_all_phases(static_features, adjective, phases, 'test')
    dyn_X, _=  collapse_all_phases(dynamic_features, adjective, phases, 'test')
       
    test_X = np.hstack((static_X, dyn_X))
    test_X = scaler.transform(test_X)
    train_score =  grid.best_score_
    test_score = f1_score(test_Y, trained_clf.predict(test_X))
    print "The training score is: %.2f, test score is %.2f" % (train_score, test_score)

def train_combined_adjectives_phases(path, adjective, phase, 
                                     static_features, dynamic_features,
                              n_jobs):
    """Combines all the features and then perform training
    """
    # File name 
    dataset_file_name = "_".join(("trained", adjective, phase))+".pkl"
    newpath = os.path.join(path, "trained_adjective_phase")
    path_name = os.path.join(newpath, dataset_file_name)
    
    if os.path.exists(path_name):
        print "File %s already exists, skipping it." % path_name
        return
    
    print "Training adjective %s and phase %s" %(adjective, phase)

    static_train_set = static_features[adjective][phase]['train']
    static_train_X = static_train_set['features']
    train_Y = static_train_set['labels']
    
    dyn_train_set = dynamic_features[adjective][phase]['train']
    dyn_train_X = dyn_train_set['features']
        
    train_X = np.hstack((static_train_X, dyn_train_X))
    
    print "Size of training set: ", train_X.shape
    
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
                                            n_jobs=n_jobs,
                                            score_func=f1_score,
                                            verbose=0)
    grid.fit(train_X, train_Y)
    trained_clf = grid.best_estimator_
    #end of magic training!!!
    
    #dataset = all_features[adjective]
    dataset = {}
    dataset['adjective'] = adjective
    dataset['classifier'] = trained_clf
    dataset['scaler'] = scaler
   
    # Save the results in the folder
    with open(path_name, "w") as f:
        print "Saving file: ", path_name
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
        
    
    static_test_set = static_features[adjective][phase]['test']
    static_test_X = static_test_set['features']
    test_Y = static_test_set['labels']
    
    dyn_test_set = dynamic_features[adjective][phase]['test']
    dyn_test_X = dyn_test_set['features']
        
    test_X = np.hstack((static_test_X, dyn_test_X))       
    test_X = scaler.transform(test_X)
    
    train_score =  grid.best_score_
    test_score = f1_score(test_Y, trained_clf.predict(test_X))
    print "The training score is: %.2f, test score is %.2f" % (train_score, test_score)
    
def train_adjectives_only():
    if len(sys.argv) == 5:
        static_path, dynamic_path, res_path, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the all adjectives"
        
        static_features = load_adjective_phase(static_path)
        dynamic_features = load_adjective_phase(dynamic_path)
        
        for adjective in adjectives:
            train_combined_adjectives(res_path,
                                      adjective, static_features, dynamic_features, n_jobs) 
        
                                                      
    else:
        print "Usage:"
        print "%s static_path dynamic_path res_path n_jobs" % sys.argv[0]
        
def train_adjectives_phases():
    if len(sys.argv) == 5:
        static_path, dynamic_path, res_path, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the all adjectives"
        
        static_features = load_adjective_phase(static_path)
        dynamic_features = load_adjective_phase(dynamic_path)
        
        for adjective, phase in itertools.product(adjectives,
                                                      phases):
            train_combined_adjectives_phases(res_path,
                                      adjective, phase,
                                      static_features, dynamic_features, n_jobs) 
        
                                                      
    else:
        print "Usage:"
        print "%s static_path dynamic_path res_path n_jobs" % sys.argv[0]

if __name__=="__main__":
    train_adjectives_phases()
    print "done"   