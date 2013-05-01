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


def train_combined_adjectives(path, adjective, all_features,
                              n_jobs):
    
    # File name 
    dataset_file_name = "_".join(("trained", adjective))+".pkl"    
    newpath = os.path.join(path, "trained_adjectives")
    path_name = os.path.join(newpath, dataset_file_name)    
        
    
    train_X = all_features[adjective]['train']['X']
    train_Y = all_features[adjective]['train']['Y']
    train_ids = all_features[adjective]['train']['ids']
    print "Training adjective %s, size: %s" % (adjective, train_X.shape)
    
    leav_out = 3
    clf = Pipeline([
        ('scaler', sklearn.preprocessing.StandardScaler()),
        ('svm', sklearn.svm.LinearSVC()), 
         ])   
    
    cv = SafeLeavePLabelOut(train_ids, leav_out, 50, train_Y)
    parameters = {
        #'svm_C': np.linspace(0.001,1e6,100),
        #'svm__C': [10101.011090909091 ],
        'svm__C': np.linspace(1e2, 1e6, 50),                      
        #'svm__penalty': ['l2','l1'],
        'svm__penalty': ['l2'],
        'svm__dual': [False],
        'svm__class_weight' : ('auto',),
                  }
    
    verbose = 1
    grid = sklearn.grid_search.GridSearchCV(clf, parameters,
                                            n_jobs=n_jobs,
                                            cv=cv, 
                                            score_func=f1_score,
                                            verbose=verbose)
    grid.fit(train_X, train_Y)
    trained_clf = grid.best_estimator_
    #end of magic training!!!!
    
    #dataset = all_features[adjective]
    dataset = all_features[adjective]
    dataset['adjective'] = adjective
    dataset['classifier'] = trained_clf
    dataset['scaler'] = False
    
    test_X = all_features[adjective]['test']['X']
    test_Y = all_features[adjective]['test']['Y']
    
    train_score =  grid.best_score_
    test_score = f1_score(test_Y, trained_clf.predict(test_X))
    print "The training score is: %.2f, test score is %.2f" % (train_score, test_score)
    print "Params are: ", grid.best_params_
    
    # Save the results in the folder
    with open(path_name, "w") as f:
        print "Saving file: ", path_name
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)    

def main():
    if len(sys.argv) == 4:
        path, adjective, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training only the adjective %s" % (adjective)
        
        dict_file = os.path.join(path, "combined_dict.pkl")        
        all_features =  cPickle.load(open(dict_file))
        train_combined_adjectives(path, adjective, all_features, n_jobs)

    elif len(sys.argv) == 3:
        path, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the all adjectives"
        
        dict_file = os.path.join(path, "combined_dict.pkl")        
        all_features =  cPickle.load(open(dict_file))
        
        #sleep tight while random race conditions happen
        failed_adjectives = adjectives[:]        
        while len(failed_adjectives) > 0:
            adjective = failed_adjectives.pop()
            try:
                train_combined_adjectives(path, adjective, all_features, n_jobs)
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