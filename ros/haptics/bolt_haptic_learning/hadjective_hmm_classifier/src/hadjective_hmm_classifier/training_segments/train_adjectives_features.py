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
from sklearn.pipeline import Pipeline
from safe_leave_p_out import SafeLeavePLabelOut

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

def load_adjective_collective(base_directory):
    
    all_features_phase = load_adjective_phase(base_directory)
    all_features = defaultdict(dict)
    
    for adjective in adjectives:
        train_X = []
        test_X = []
        for phase in phases:
            train_set = all_features_phase[adjective][phase]['train']
            train_X.append(train_set['features'])
            train_Y = train_set['labels']
            train_object_ids = train_set['object_ids']
            
            test_set = all_features_phase[adjective][phase]['test']
            test_X.append(test_set['features'])
            test_Y = test_set['labels']
            test_object_ids = test_set['object_ids']
        
        train_X = np.concatenate(train_X, axis=1)
        test_X = np.concatenate(test_X, axis=1)
        
        all_features[adjective]['adjective'] = adjective
        all_features[adjective]['train'] = {'X': train_X,
                                            'Y': train_Y,
                                            'ids':train_object_ids}
        
        all_features[adjective]['test'] = {'X': test_X,
                                            'Y': test_Y,
                                            'ids':test_object_ids}
    return all_features
        

def alt_train_adjective_phase_classifier(path, adjective, all_features, njobs):
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

    all_features =  all_features[adjective]

    train_X = all_features['train']['X']
    train_Y = all_features['train']['Y']
    train_ids = all_features['train']['ids']

    print "Training adjective %s" % adjective

    #magic training happening here!!!
    leav_out = 3
    clf = Pipeline([
        ('scaler', sklearn.preprocessing.StandardScaler()),
        ('svm', sklearn.svm.LinearSVC()), 
         ])   
    
    #cv = sklearn.cross_validation.LeavePLabelOut(train_ids, leav_out)
    cv = SafeLeavePLabelOut(train_ids, leav_out, 100, train_Y)
    parameters = {
        #'svm_C': np.linspace(0.001,1e6,100),
        #'svm__C': [10101.011090909091 ],
        'svm__C': np.linspace(1e2, 1e6, 100),                      
        #'svm__penalty': ['l2','l1'],
        'svm__penalty': ['l2'],
        'svm__dual': [False],
        'svm__class_weight' : ('auto',),
                  }
    
    verbose = 1
    grid = sklearn.grid_search.GridSearchCV(clf, parameters,
                                            n_jobs=njobs,
                                            cv=cv, 
                                            score_func=f1_score,
                                            verbose=verbose)
    grid.fit(train_X, train_Y)
    trained_clf = grid.best_estimator_
    #end of magic training!!!
    
    dataset = all_features
    dataset['adjective'] = adjective
    dataset['classifier'] = trained_clf
    dataset['scaler'] = False
   
    # Save the results in the folder
    with open(path_name, "w") as f:
        print "Saving file: ", path_name
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
        
    test_X = all_features['test']['X']
    test_Y = all_features['test']['Y']
    
    train_score =  grid.best_score_
    test_score = f1_score(test_Y, trained_clf.predict(test_X))
    print "The training score is: %.2f, test score is %.2f" % (train_score, test_score)
    print "Params are: ", grid.best_params_
    

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


def orig_main():
    if len(sys.argv) == 4:
        path, adjective, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the adjective %s for the phase %s" % (
                adjective)

        loaded_features = load_adjective_phase(path)
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(orig_train_adjective_phase_classifier)(path, adjective, loaded_features))

    elif len(sys.argv) == 3:
        path, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the all adjectives"
        loaded_features = load_adjective_phase(path)
 
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(orig_train_adjective_phase_classifier)(path, adjective, loaded_features) 
            for adjective in adjectives)
                                                      
    else:
        print "Usage:"
        print "%s path adjective n_jobs" % sys.argv[0]
        print "%s path n_jobs" % sys.argv[0]
        print "Path to the base directory"

def main():
    if len(sys.argv) == 4:
        path, adjective, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training only the adjective %s" % (adjective)
        
        all_features =  load_adjective_collective(path)
        alt_train_adjective_phase_classifier(path, adjective, all_features, n_jobs)

    elif len(sys.argv) == 3:
        path, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the all adjectives"
        
        all_features =  load_adjective_collective(path)

        #sleep tight while random race conditions happen
        failed_adjectives = adjectives[:]        
        while len(failed_adjectives) > 0:
            adjective = failed_adjectives.pop()
            try:
                alt_train_adjective_phase_classifier(path, adjective, all_features, n_jobs)
            except ValueError:
                print "adjective %s has problems, retrying..."
                failed_adjectives.append(adjective)
        
        
        #p = Parallel(n_jobs=n_jobs,verbose=10)
        #p(delayed(alt_train_adjective_phase_classifier)(path, adjective, all_features, 1) 
            #for adjective in adjectives)
                                                      
    else:
        print "Usage:"
        print "%s path adjective n_jobs" % sys.argv[0]
        print "%s path n_jobs" % sys.argv[0]
        print "Path to the base directory"

if __name__=="__main__":
    main()
    print "done"        

