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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pylab
import scipy.io as sio
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_kernels


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

def compute_PCA(features):
    """
    Given an array of features, performs PCA
    and returns the results of the analysis
    """

    # Fit all features for principal component
    pca = PCA(1)
    pca.fit(features)
   
    top_component = pca.transform(features)

    return top_component

def compute_kernel(features):

    #$scaler = preprocessing.StandardScaler().fit(features)
    #X_train = scaler.transform(features)
    #kernel = linear_kernel(X_train, -2)
    kernel_mean = np.mean(features,axis = 1)

    return kernel_mean
   

 
def rbf_kernel(X, n_jobs):
    return pairwise_kernels(X, metric="rbf", n_jobs=n_jobs, gamma=0.1)

def linear_kernel(X, n_jobs):
    return pairwise_kernels(X, metric="linear", n_jobs=n_jobs)

def standardize(A):
    return (A - np.mean(A)) / np.std(A)


def compute_probability_density(base_directory, adjective, motion):
    """
    Compute the probability density of positive and 
    negative adjective per motion
    """
   
    # Pull out features for adjective/motion
    all_features = load_adjective_phase(base_directory)
    adj_motion_feat = all_features[adjective][motion]['train'] 
    features = adj_motion_feat['features']
    selected_components = compute_PCA(features)
    
    # Now split them according to positive and negative
    labels = adj_motion_feat['labels'] 
    positive_components = selected_components[labels == 1]
    negative_components = selected_components[labels == 0]

    return (positive_components, negative_components)


def compute_probability_density_kernel(base_directory, classifiers, adjective, motion):
    """
    Compute the probability density of positive and 
    negative adjective per motion
    """
  
    # Load static and dynamic
    dynamic_features = utilities.load_adjective_phase('/media/data_storage/vchu/all_classifiers/icra2014/dynamic/adjective_phase_set/')
    static_features = utilities.load_adjective_phase('/media/data_storage/vchu/all_classifiers/icra2014/static/adjective_phase_set/')

    dynamic_train = dynamic_features[adjective][motion]['train']['features']
    dynamic_train_scaler = preprocessing.StandardScaler().fit(dynamic_train)
    dynamic_train_scaled_X = dynamic_train_scaler.transform(dynamic_train)
    dynamic_train_kernel = linear_kernel(dynamic_train_scaled_X, -2)
    #dynamic_train_kernel = standardize(dynamic_train_kernel)

    #Static Train
    static_train = static_features[adjective][motion]['train']['features']
    static_train_scaler = preprocessing.StandardScaler().fit(static_train)
    static_train_scaled_X = static_train_scaler.transform(static_train)
    static_train_kernel = linear_kernel(static_train_scaled_X, -2)

    # Load alphas
    classifier = []
    for mkl in classifiers:
        if (mkl['adjective']== adjective) and (mkl['phase'] == motion):
            classifier = mkl

    alpha = classifier['alpha']

    # Pull out features for adjective/motion
    #all_features = load_adjective_phase(base_directory)
    #adj_motion_feat = all_features[adjective][motion]['train'] 
    #features = adj_motion_feat['features']
    #selected_components = compute_kernel(features)
   
    train_X = (alpha)*static_train_kernel + (1-alpha)*dynamic_train_kernel
    labels = dynamic_features[adjective][motion]['train']['labels']
   
    selected_components = compute_kernel(train_X)

    # Now split them according to positive and negative
    #labels = adj_motion_feat['labels'] 
    positive_components = selected_components[labels == 1]
    negative_components = selected_components[labels == 0]

    return (positive_components, negative_components)


def densityplot(data):
    """
    Plots a histogram of daily returns from data, 
    plus fitted normal density.
    """
    
    dailyreturns = data
    pylab.hist(dailyreturns, bins=200, normed=True)
    m, M = min(dailyreturns), max(dailyreturns)
    mu = pylab.mean(dailyreturns)
    sigma = pylab.std(dailyreturns)
    grid = pylab.linspace(m, M, 100)
    densityvalues = pylab.normpdf(grid, mu, sigma)
    pylab.plot(grid, densityvalues, 'r-')
    pylab.show()

        


def template_function(base_directory):
    """
    Example function on how to access all of the features
    stored in adjective_phase_set
    """

    all_features = load_adjective_phase(base_directory)
    import pdb; pdb.set_trace()


def main():
    if len(sys.argv) == 3:
        path = sys.argv[1]
        classifier_dict = cPickle.load(open(sys.argv[2]))
 
        store_matlab = defaultdict(dict) 
 
        for adjective, phase in itertools.product(adjectives, phases):
            pos,neg = compute_probability_density_kernel(path, classifier_dict, adjective, phase)
            store_matlab[adjective][phase] = (pos,neg)            

        sio.savemat('kernel_one.mat', {'values': store_matlab})

        import pdb; pdb.set_trace() 

    else:
        print "Usage:"
        print "%s path" % sys.argv[0]
        print "Path to the base directory"

if __name__=="__main__":
    main()
    print "done"        
