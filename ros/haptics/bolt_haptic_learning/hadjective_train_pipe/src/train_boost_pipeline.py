#!/usr/bin/env python
import roslib; roslib.load_manifest("hadjective_train_pipe")
import rospy
import numpy as np
import sys 
import os
from optparse import OptionParser
import cPickle
import bolt_learning_utilities as utilities
import extract_features as extract_features
import matplotlib.pyplot as plt 
import milk.supervised.adaboost as boost


from bolt_feature_obj import BoltFeatureObj
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import classification_report
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.decomposition import PCA 
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier


def train_weak_classifier_adjective(train_feature_objects, adjective, feature_dictionary):
    '''
    takes in a dictionary of all features
    
    returns a dictionary of weak classifiers for each feature
    '''
    
    # specify feature to be extracted
    feature_name_list = ["pdc_rise_count", "pdc_area", "pdc_max", "pac_energy", "pac_sc", "pac_sv", "pac_ss", "pac_sk", "tac_area", "tdc_exp_fit", "gripper_min", "gripper_mean", "transform_distance", "electrode_polyfit"]

    # store weak svms
    svm_motion_store = dict()

    # store scalers
    scaler_motion_store = dict()

    # store scalers
    classifiers = dict()
    
    # for each motion (slide, squeeze, etc.)
    for motion in train_feature_objects:
        motion_train_set = train_feature_objects[motion]
        
        # pull out the features specified as a vector 
        train_feature_vector, train_label_dict = utilities.feature_obj_2_feature_vector(motion_train_set, feature_name_list)
    
        # create scaler
        scaler_motion_store[motion] = preprocessing.Scaler().fit(train_feature_vector)
        train_feature_vector_scaled = scaler_motion_store[motion].transform(train_feature_vector)
     
        #params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 1,'learn_rate': 0.01, 'loss': 'deviance'} 
        #clf = ensemble.GradientBoostingClassifier(**params)
        #clf.fit(train_feature_vector_scaled, train_label_dict[1][adjective])
        clf = train_gradient_boost(train_feature_vector_scaled, train_label_dict[1][adjective], train_label_dict[0])

        classifiers[motion] = clf
 
    return (classifiers, scaler_motion_store)


def test_weak_classifier_adjective(test_feature_objects, adjective, classifiers, scalers, file_ptr, comma_file_ptr):
    '''
    takes in a dictionary of all features
    
    returns a dictionary of weak classifiers for each feature
    '''
    
    # specify feature to be extracted
    feature_name_list = ["pdc_rise_count", "pdc_area", "pdc_max", "pac_energy", "pac_sc", "pac_sv", "pac_ss", "pac_sk", "tac_area", "tdc_exp_fit", "gripper_min", "gripper_mean", "transform_distance", "electrode_polyfit"]

    motion_scores = dict()
    
    # for each motion (slide, squeeze, etc.)
    for motion in test_feature_objects:
        motion_test_set = test_feature_objects[motion]
        
        # pull out the features specified as a vector 
        test_feature_vector, test_label_dict = utilities.feature_obj_2_feature_vector(motion_test_set, feature_name_list)
    
        # create scaler
        test_feature_vector_scaled = scalers[motion].transform(test_feature_vector)
      
        clf = classifiers[motion]

        results = clf.predict(test_feature_vector_scaled)

        print classification_report(test_label_dict[1][adjective], results)

        file_ptr.write('Adjective: ' + adjective + '     Motion name: '+motion)
        file_ptr.write('\n'+classification_report(test_label_dict[1][adjective], results)+ '\n\n')

        motion_scores[motion] = f1_score(test_label_dict[1][adjective], results)

    # Pull out the best motion
    best_motion, best_score = utilities.get_best_motion(motion_scores)

    # Write results to comma split file to load
    comma_file_ptr.write(adjective+','+str(motion_scores['tap'])+',' +str(motion_scores['squeeze'])+','+str(motion_scores['thermal_hold'])+','+str(motion_scores['slide'])+','+str(motion_scores['slide_fast'])+','+best_motion +'\n')
 
     
# Training a single GBC given a training set
def train_gradient_boost(train_vector, train_labels, object_ids):
    """
    train_svm - expects a vector of features and a nx1 set of
                corresponding labels

    Returns a trained Gradient Boosting classifier
    """
   
    # Create the obj_id_vector for cross validation
    lpl = cross_validation.LeavePLabelOut(object_ids, p=1,indices=True)
    
    parameters = {'n_estimators': [1000], 'learn_rate': [1e-1, 1e-2, 1e-3]} 

    # Grid search with nested cross-validation
    #parameters = {'kernel': ['rbf'], 'C': [1, 1e1, 1e2, 1e3, 1e4], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4]}
    clf = GridSearchCV(GradientBoostingClassifier(max_depth=4), parameters, score_func=f1_score, cv=lpl)

    # Train the SVM using the best parameters
    clf.fit(train_vector, train_labels)
    clf_best = clf.best_estimator_
   
    return clf_best


def train_weak_classifier_motion(motion_train_set, adjective, feature_dictionary):
    '''
    Takes the feature_object_train_set and trains the specified
    feature.

    Will return a single trained SVM
    '''

    # Store SVM for each feature
    svm_store = dict()
    scaler_store = dict()

    import pdb; pdb.set_trace()
    # For each feature set (pdc, pac, etc.)
    for feature in feature_dictionary:

        # Pull out the list of features        
        feature_list = feature_dictionary[feature]
       
        # Pull out the features specified as a vector 
        train_feature_vector, train_label_dict = utilities.feature_obj_2_feature_vector(motion_train_set, feature_list)
    
        # Create scaler
        scaler_store[feature] = preprocessing.Scaler().fit(train_feature_vector)
        train_feature_vector_scaled = scaler_store[feature].transform(train_feature_vector)
 
        # Train the SVM
        svm_store[feature] = train_svm(train_feature_vector_scaled, train_label_dict[1][adjective], train_label_dict[0])

    return (svm_store, scaler_store)




# Training a single SVM given a training set
def train_svm(train_vector, train_labels):
    """
    train_svm - expects a vector of features and a nx1 set of
                corresponding labels

    Returns a trained SVM classifier
    """
    
    # Create the obj_id_vector for cross validation
    lpl = cross_validation.LeavePLabelOut(np.array(train_vector[1]), p=1,indices=True)

    # Grid search with nested cross-validation
    parameters = {'kernel': ['rbf'], 'C': [1, 1e1, 1e2, 1e3, 1e4], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4]}
    svm = GridSearchCV(SVC(probability=True), parameters, score_func=f1_score, cv=lpl)

    # Train the SVM using the best parameters
    svm.fit(train_vector, train_labels)
    svm_best = svm.best_estimator_
   
    return svm_best


def main(feature_objects_set_file, pca_file, output_svm_filename, split):

    # Load data into the pipeline. First check
    # for feature object pkl files
    print "Loading data from file\n"

    # Load Input Set
    if feature_objects_set_file == None:
        raise Exception('No input feature object file')
    else:
        feature_objects_set = cPickle.load(open(feature_objects_set_file, 'r'))

    # Load PCA dictionary
    if pca_file == None:
        raise Exception('No PCA dictionary provided')
    else:
        pca_dict = cPickle.load(open(pca_file, 'r'))

    # Check output SVM filename
    if output_svm_filename == None:
        print 'Warning: No filename given - writing to all_svm_boost_classifiers.pkl'
        output_svm_filename = 'all_svm_boost_classifiers.pkl'

    # Store parameters for each feature list
    feature_dictionary = dict()
    feature_dictionary['pdc'] = ['pdc_rise_count', 'pdc_area', 'pdc_max']
    feature_dictionary['pac'] = ['pac_energy', 'pac_sc', 'pac_sv', 'pac_ss', 'pac_sk']
    feature_dictionary['temperature'] = ['tac_area', 'tdc_exp_fit']
    feature_dictionary['gripper'] = ['gripper_min', 'gripper_mean', 'transform_distance']
    feature_dictionary['electrodes'] = ['electrode_polyfit']

    #import pdb; pdb.set_trace()

    classifiers = dict()
    scaler = dict()
    report_file = open('reports/Gradient_Boosting_Results_'+str(split)+'.txt', 'w')
    results_file = open('reports/Gradient_Boosting_Results_comma_'+str(split)+'.txt', 'w')

    results_file.write('Adjective,Tap,Squeeze,Static Hold,Slide,Slide Fast, Best Motion\n')

    adjective_list = feature_objects_set.keys()

    if split == 0:
        adjective_run = adjective_list[0:10] 
    elif split == 1:
        adjective_run = adjective_list[11:20]
    else:
        adjective_run = adjective_list[20:-1]


    # For each adjective - train a classifier
    for adj in adjective_run:
        print adj
   
        if adj in ['porous', 'elastic', 'grainy']:
            continue     
        
        # Build a weak classifier for each feature
        classifiers[adj], scaler[adj] = train_weak_classifier_adjective(feature_objects_set[adj]['train'], adj, feature_dictionary)
            
        # Test classifier
        report = test_weak_classifier_adjective(feature_objects_set[adj]['test'], adj, classifiers[adj], scaler[adj], report_file, results_file)

        cPickle.dump((classifiers[adj], scaler[adj]), open('boosting/boost_and_scaler_'+adj+'.pkl','w'), cPickle.HIGHEST_PROTOCOL)

    report_file.close() 
    cPickle.dump((classifiers, scaler), open(output_svm_filename, 'w'), cPickle.HIGHEST_PROTOCOL)

# Parse the command line arguments
def parse_arguments():
    """
    Parses the arguments provided at command line.

    Expects the input feature objects to be split by adjective
    and by test/train
    
    Returns:
    (feature_objects_set, output_svm_filename)
    """
    parser = OptionParser()
    parser.add_option("-i", "--feature_obj_file", action="store", type="string", dest = "feature_obj_file")
    parser.add_option("-o", "--output_svm", action="store", type="string", dest = "output_svm_file", default = None)
    parser.add_option("-p", "--pca_file", action="store", type="string", dest = "pca_file", default = None)
    parser.add_option("-s", "--split_flag", action="store", type="int", dest = "split_flag", default = None)

    (options, args) = parser.parse_args()

    feature_objects_set = options.feature_obj_file
    output_svm_filename = options.output_svm_file
    pca_file = options.pca_file   
    split_flag = options.split_flag

    return (feature_objects_set, pca_file, output_svm_filename, split_flag)

if __name__ == "__main__":
    feature_objects_set, pca_file, output_svm_filename, split_flag = parse_arguments()
    main(feature_objects_set, pca_file, output_svm_filename, split_flag)

