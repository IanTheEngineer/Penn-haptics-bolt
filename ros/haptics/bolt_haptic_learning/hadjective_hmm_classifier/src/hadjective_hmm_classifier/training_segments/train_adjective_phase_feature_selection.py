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
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt


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

def plot_feature_statistics(scores, train_X):
    ''' 
    Plots the Univariate feature stats as computed in compute_feature_statistics
    '''

    fig = plt.figure(1)
    plt.clf()

    X_indices = np.arange(train_X.shape[-1])

    plt.bar(X_indices, scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)', color='g')

    x_labels = ['rc1', 'rc2', 'dca1', 'dca2','dcm1', 'dcm2','ace1','ace2','acsc1', 'acsc2', 'acsv1', 'acsv2', 'acss1','acss2', 'acsk1', 'acsk2', 'taca1', 'taca2', 'tdc1', 'tdc2', 'gmin', 'gmean', 'trd','ep111','ep112','ep211', 'ep212', 'ep311','ep312', 'ep411','ep412','ep511','ep512','ep611','ep612','ep121','ep122','ep221', 'ep222', 'ep321','ep322', 'ep421','ep422','ep521','ep522','ep621','ep622'] 

    # key for above - ep_coefficient_principalcomponent_finger

    plt.xticks(X_indices, x_labels)
    fig.autofmt_xdate()

    return plt 

def compute_feature_statistics(train_X, train_Y):
    ''' 
    Univariate Feature Selection - see sklearn:
    http://scikit-learn.org/dev/auto_examples/plot_feature_selection.html#example-plot-feature-selection-py
    
    Features are not removed, only statistics computed
    '''

    selector = SelectPercentile(f_classif, percentile = 10)
    selector.fit(train_X, train_Y)
    scores = -np.log10(selector.pvalues_)
    scores /=scores.max()

    return scores, selector

def remove_feature_tree_based(train_X,train_Y):
    '''
    Removes features based on trees - see sklearn:
    http://scikit-learn.org/dev/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py

    Actually removes based on "importance"
    '''
    forest = ExtraTreesClassifier(n_estimators=1000,
                                  compute_importances = True,
                                  random_state = 0)

    forest.fit(train_X, train_Y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                  axis=0)
    indices = np.argsort(importances)[::-1]

    x_labels = ['rc1', 'rc2', 'dca1', 'dca2','dcm1', 'dcm2','ace1','ace2','acsc1', 'acsc2', 'acsv1', 'acsv2', 'acss1','acss2', 'acsk1', 'acsk2', 'taca1', 'taca2', 'tdc1', 'tdc2', 'gmin', 'gmean', 'trd','ep111','ep112','ep211', 'ep212', 'ep311','ep312', 'ep411','ep412','ep511','ep512','ep611','ep612','ep121','ep122','ep221', 'ep222', 'ep321','ep322', 'ep421','ep422','ep521','ep522','ep621','ep622']

    # Print the feature ranking
    print "Feature ranking:"

    for f in xrange(46):
        print "%d. feature %s (%f)" % (f + 1, x_labels[indices[f]], importances[indices[f]])

    # Transform the data to have only the features that are important
    x_new = forest.transform(train_X)

    return (forest, x_new)


def train_adjective_phase_classifier(path, adjective, phase, all_features, boost):
    """
    Example function on how to access all of the features
    stored in adjective_phase_set
    """

    # File name 
    dataset_file_name = "_".join(("trained_tree_feat_select", adjective, phase))+".pkl"
    newpath = os.path.join(path, "trained_adjective_phase_tree_feat_select")
    path_name = os.path.join(newpath, dataset_file_name)
    
    if os.path.exists(path_name):
        print "File %s already exists, skipping it." % path_name
        return

    print "Creating adjective %s and phase %s" % (adjective, phase)

    train_set = all_features[adjective][phase]['train']
    train_X = train_set['features']

    # Scale the data
    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    all_features[adjective][phase]['scaler'] = scaler
    all_features[adjective][phase]['train'] = train_X   # store off scaled

    train_Y = train_set['labels']
    object_ids = train_set['object_ids']

    print "Training adjective %s and phase %s" %(adjective, phase)

    # Remove features!
    all_features[adjective][phase]['tree_features'] = remove_feature_tree_based(train_X,train_Y)

    print np.shape(train_X)

    train_X = all_features[adjective][phase]['tree_features'][1]; # transformed features
    print np.shape(train_X)

    if not boost:
        trained_clf, scaler = utilities.train_svm_gridsearch(train_X = train_X,
                             train_Y = train_Y,
                             verbose=True,
                             object_ids = object_ids,
                             n_jobs = 6,
                             scale = False 
                             )   
    else: 
        trained_clf, scaler = utilities.train_gradient_boost(train_X = train_X,
                                train_Y = train_Y,
                                object_ids = object_ids,
                                verbose = True, 
                                n_jobs = 6,
                                scale = False 
                                )

    dataset = all_features[adjective][phase]
    dataset['adjective'] = adjective
    dataset['phase'] = phase
    dataset['classifier'] = trained_clf
   
    print "Saving trained_classifier" 

    # Save the results in the folder
    with open(path_name, "w") as f:
         print "Saving file: ", path_name
         cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)

def main():
    if len(sys.argv) == 6:
        path, adjective, phase, n_jobs, boost = sys.argv[1:]
        n_jobs = int(n_jobs)
        boost = bool(int(boost))
        print "Training the adjective %s for the phase %s" % (
                adjective, phase)

        loaded_features = load_adjective_phase(path)
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(train_adjective_phase_classifier)(path, adjective, phase, loaded_features, boost))

    elif len(sys.argv) == 5:
        path, adjective, n_jobs, boost = sys.argv[1:]
        n_jobs = int(n_jobs)
        boost = bool(int(boost))
        print "Training the adjective %s" % adjective
        loaded_features = load_adjective_phase(path)
 
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(train_adjective_phase_classifier)(path, adjective, phase, loaded_features, boost) 
            for phase in itertools.product(phases))
 
    elif len(sys.argv) == 4:
        path, n_jobs, boost = sys.argv[1:]
        n_jobs = int(n_jobs)
        boost = bool(int(boost))
        print "Training the all combinations of adjectives and phases"
        loaded_features = load_adjective_phase(path)
 
        p = Parallel(n_jobs=n_jobs,verbose=10)
        p(delayed(train_adjective_phase_classifier)(path, adjective, phase, loaded_features, boost) 
            for adjective, phase in itertools.product(adjectives,
                                                      phases))
    else:
        print "Usage:"
        print "%s path adjective phase n_jobs boosted" % sys.argv[0]
        print "%s path adjective n_jobs boosted" % sys.argv[0]
        print "%s path n_jobs boosted" % sys.argv[0]
        print "Path to the base directory, boosted is true/false for gradient boosting"

if __name__=="__main__":
    main()
    print "done"        
