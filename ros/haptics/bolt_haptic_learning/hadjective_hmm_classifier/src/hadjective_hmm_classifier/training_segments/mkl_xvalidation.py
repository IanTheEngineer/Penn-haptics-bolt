#! /usr/bin/python
import cPickle
import sys
import utilities
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn import preprocessing
import numpy as np
import os
from sklearn import svm 
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.metrics import f1_score
from collections import defaultdict
import time

def rbf_kernel(X, n_jobs):
    return pairwise_kernels(X, metric="rbf", n_jobs=n_jobs, gamma=0.1)

def linear_kernel(X, n_jobs):
    return pairwise_kernels(X, metric="linear", n_jobs=n_jobs)

def standardize(A):
    return (A - np.mean(A)) / np.std(A)

def begin_train(dynamic_path, static_path, out_path):

    dynamic_features = utilities.load_adjective_phase(dynamic_path)
    static_features = utilities.load_adjective_phase(static_path)
    adjective_list = utilities.adjectives
    for adjective in adjective_list:
        # File name 
        dataset_file_name = "_".join(("trained", adjective))+".pkl"
        newpath = os.path.join(out_path, "trained_adjectives")
        path_name = os.path.join(newpath, dataset_file_name)
        if os.path.exists(path_name):
            print "File %s already exists, skipping it." % path_name
            continue

        overall_best_score = 0.0
        dataset = defaultdict()
        dataset['classifier'] = None
        dataset['training_score'] = overall_best_score

        dynamic_train = utilities.get_all_train_test_features(adjective, dynamic_features) 
        y_labels = dynamic_train[1]
        object_ids = dynamic_train[2]
        dynamic_scaler = preprocessing.StandardScaler().fit(dynamic_train[0])
        dynamic_X = dynamic_scaler.transform(dynamic_train[0])
        dynamic_kernel = linear_kernel(dynamic_X, -2)
        dynamic_kernel = standardize(dynamic_kernel)

        static_train = utilities.get_all_train_test_features(adjective, static_features) 
        static_scaler = preprocessing.StandardScaler().fit(static_train[0])
        static_X = static_scaler.transform(static_train[0])
        static_kernel = linear_kernel(static_X, -2)
        static_kernel = standardize(static_kernel)

        alpha_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        #alpha_range = [0.5]
        for alpha in alpha_range:
            print "Beginning %s, alpha %1.1f at %s" % (adjective,alpha, time.asctime())
            combined_kernel = (alpha)*static_kernel + (1-alpha)*dynamic_kernel;
            trained_clf, best_score = gram_grid_search(combined_kernel, labels=y_labels, object_ids=object_ids)
            print "F1: %1.5f at %s" % (best_score, time.asctime())
            if best_score > overall_best_score:
                overall_best_score = best_score
                dataset['classifier'] = trained_clf
                dataset['training_score'] = best_score
                dataset['alpha'] = alpha

        dataset['dynamic_features'] = dynamic_features[adjective]
        dataset['static_features'] = static_features[adjective]
        dataset['adjective'] = adjective
        dataset['dynamic_scaler'] = dynamic_scaler
        dataset['dynamic_train_scaled'] = dynamic_X;
        dataset['dynamic_kernel_mean'] = np.mean(dynamic_kernel)
        dataset['dynamic_kernel_std'] = np.std(dynamic_kernel)
        dataset['static_scaler'] = static_scaler
        dataset['static_train_scaled'] = static_X;
        dataset['static_kernel_mean'] = np.mean(static_kernel)
        dataset['static_kernel_std'] = np.std(static_kernel)
   
        print "Saving trained_classifier" 

        # Save the results in the folder
        with open(path_name, "w") as f:
            print "Saving file: ", path_name
            cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)

def gram_grid_search(gram, labels, object_ids=None, n_jobs=6, score_fun=f1_score, verbose = 0):

    if (object_ids is None) or (sum(labels) <= 10):
        print "Cannot perform leave one out cross validation"
        cv = 5 # 5 fold cross validation
    else:
        # Leave one object out cross validation
        cv = cross_validation.LeavePLabelOut(object_ids, p=1,indices=True)
    parameters = {
                  #'C': np.linspace(1,1e6,1000),
                  #'C': np.linspace(1,1e6,100),
                  'C': (1, 10, 100, 1000)
                  #'C': (1.0, 10, 100, 1000, 1e4, 1e5, 1e6) 
                  #'penalty':('l1','l2'),
                  }
    # class weight normalizes the lack of positive examples
    clf = svm.SVC(class_weight='auto',kernel='precomputed')
    import pdb; pdb.set_trace()
    grid = GridSearchCV(clf, parameters, cv=cv,
                        verbose=verbose,
                        n_jobs=n_jobs,
                        score_func=score_fun,
                        )

    grid.fit(gram, labels)
    best_estimator = grid.best_estimator_
    best_score = grid.best_score_

    return best_estimator, best_score


def main():
    if len(sys.argv) == 4:
        dynamic_path, static_path, out_path = sys.argv[1:]
        begin_train(dynamic_path, static_path, out_path)
    else:
        print "Usage:"
        print "%s database path adjective phase n_jobs" % sys.argv[0]
        print "%s database path adjective n_jobs" % sys.argv[0]
        print "%s database path" % sys.argv[0]

if __name__ == "__main__":
    main()
    print "done"

