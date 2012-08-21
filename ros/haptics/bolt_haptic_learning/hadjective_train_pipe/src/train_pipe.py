#!/usr/bin/env python
import roslib; roslib.load_manifest("hadjective_train_pipe")
import rospy
import numpy as np
import sys 
import os
from optparse import OptionParser
import cPickle
import bolt_learning_utilities as utilities
import matplotlib.pyplot as plt 
import sklearn.decomposition

from bolt_feature_obj import BoltFeatureObj
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs

# Loads the data from h5 table and adds labels
# Returns the dictionary of objects
def loadDataFromH5File(input_file, adjective_file):
   
    # Takes the input h5 file and converts into bolt object data
    all_bolt_data = utilities.convertH5ToBoltObjFile(input_file, None, False);
   
    # Inserts adjectives into the bolt_data  
    all_bolt_data_adj = utilities.insertAdjectiveLabels(all_bolt_data, None, adjective_file, False)

    return all_bolt_data_adj


# Takes the bolt data and extracts features to run
def BoltMotionObjToFeatureObj(all_bolt_data):
    """
    Pull out PCA components from all data

    For each object - pull out features and store in feature_obj
    with the same structure as all_bolt_data
   
        Dictionary - "tap", "slide", "slow_slide", 
                     "thermal_hold", "squeeze"

    """
    # DO PCA Calculations here 
    
    # Store in feature class object
    all_features_obj_dict = dict();

    for motion_name in all_bolt_data:
        motion_list = all_bolt_data.get(motion_name)
        print motion_name

        feature_list = list()
        # For all objects
        for motion in motion_list:
            
            bolt_feature_obj = utilities.extract_features(motion)
            
            feature_list.append(bolt_feature_obj)

        # Store all of the objects away
        all_features_obj_dict[motion_name] = feature_list
            
    return all_features_obj_dict        
    

def bolt_obj_2_feature_vector(all_bolt_data, feature_name_list):
    """
    Pull out PCA components from all data

    For each object - pull out features and store in feature_obj
    with the same structure as all_bolt_data
   
        Dictionary - "tap", "slide", "slow_slide", 
                     "thermal_hold", "squeeze"

    Directly store the features into a vector
    See createFeatureVector for more details on structure

    """
    
    # DO PCA Calculations here 
    
    # Store in feature class object
    all_features_vector_dict = dict()
    
    # Store labels
    for motion_name in all_bolt_data:
        motion_list = all_bolt_data.get(motion_name)
        print motion_name

        all_adjective_labels_dict = dict()
        feature_vector_list = list()
        # For all objects
        for motion in motion_list:
            
            # Create feature vector
            bolt_feature_obj = utilities.extract_features(motion)
            feature_vector = utilities.createFeatureVector(bolt_feature_obj, feature_name_list) 
            feature_vector_list.append(feature_vector)

            # Create label dictionary
            labels = motion.labels
            for adjective in labels:
                # Check if it is the first time adjective added
                if (all_adjective_labels_dict.has_key(adjective)):
                    adjective_array = all_adjective_labels_dict[adjective]
                else:
                    adjective_array = list()
                
                # Store array
                adjective_array.append(labels[adjective])
                all_adjective_labels_dict[adjective] = adjective_array

        # Store all of the objects away
        all_features_vector_dict[motion_name] = np.array(feature_vector_list)
     
    return (all_features_vector_dict, all_adjective_labels_dict)      


def run_dbscan(input_vector, num_clusters):
    """
    run_dbscan - expects a vector of features and the number of
                 clusters to generate

                 dbscan uses nearest neighbor metrics to compute
                 similarity

    Returns the populated clusters
    """


def run_kmeans(input_vector, num_clusters):
    """
    run_kmeans - expects a vector of features and the number of
                 clusters to generate

    Returns the populated clusters 
    """
    k_means = KMeans(init='k-means++', k=num_clusters, n_init=10)

    k_means.fit(input_vector)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_mean_labels_unique = np.unique(k_means_labels)

    return (kmean_labels, k_means_labels, k_means_cluster_centers)


def train_knn(train_vector, train_labels, N):
    """
    train_knn - expects a vector of features and a nx1 set of
                corresponding labels.  Finally the number of
                neighbors used for comparison

    Returns a trained knn classifier
    """



def train_svm(train_vector, train_label):
    """
    train_svm - expects a vector of features and a nx1 set of
                corresponding labels

    Returns the a trained SVM classifier
    """



# MAIN FUNCTION
def main(input_file, adjective_file):
 
    # Load data into the pipeline, either from an h5 and adjective
    # File or directly from a saved pkl file
    print "Loading data from file"
    if input_file.endswith(".h5"):
        all_data = loadDataFromH5File(input_file, adjective_file)
    else:
        all_data = utilities.loadBoltObjFile(input_file)

    print "loaded data"

    # Split the data into train and test
    train_data, test_data = utilities.split_data(all_data, 0.9)
    
    # Take loaded data and extract out features
    feature_name_list = ["max_pdc", "pdc_area"]
    train_feature_vector, train_adjective_dictionary = bolt_obj_2_feature_vector(train_data, feature_name_list)

    test_feature_vector, test_adjective_dictionary = bolt_obj_2_feature_vector(test_data, feature_name_list)

    # Do for all data for clustering purposes
    all_feature_vector, all_adjective_dictionary = bolt_obj_2_feature_vector(all_data, feature_name_list)
    
    print("Created feature vector containing %s" % feature_name_list)

    # Run k-means
    kmean_labels, k_means_labels, k_means_cluster_centers = run_kmeans(all_feature_vector['squeeze'], 25)

    

# Parse the command line arguments
def parse_arguments():
    """Parses the arguments provided at command line.
    
    Returns:
    (input_file, adjective_file, range)
    """
    parser = OptionParser()
    parser.add_option("-i", "--input_file", action="store", type="string", dest = "in_h5_file")
    parser.add_option("-o", "--output", action="store", type="string", dest = "out_file", default = None) 
    parser.add_option("-a", "--input_adjective", action="store", type="string", dest = "in_adjective_file")

    (options, args) = parser.parse_args()
    input_file = options.in_h5_file #this is required
   
    if options.out_file is None:
        (_, name) = os.path.split(input_file)
        name = name.split(".")[0]
        out_file = name + ".pkl"
    else:    
        out_file = options.out_file
        if len(out_file.split(".")) == 1:
            out_file = out_file + ".pkl"
    
    adjective_file = options.in_adjective_file

    return input_file, out_file, adjective_file


if __name__ == "__main__":
    input_file, out_file, adjective_file = parse_arguments()
    main(input_file, adjective_file)
