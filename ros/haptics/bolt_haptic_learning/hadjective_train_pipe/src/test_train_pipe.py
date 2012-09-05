#!/usr/bin/env python
import roslib; roslib.load_manifest("hadjective_train_pipe")
import rospy
import numpy as np
import sys 
import os
from optparse import OptionParser
import cPickle
import pickle
import bolt_learning_utilities as utilities
import extract_features as extract_features
import matplotlib.pyplot as plt 

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



# Function to start testing each classifier against a test set
def test_adj_motion_classifier(classifier_dict, adjective_name, test_feature_objs, feature_list, scaler_dict):
    '''
    Pass in the trained adjective classifiers that are motion specific and return the results
    of the motion adjectives compared to the test set
    '''

    # For each adjective
    object_run_probability = dict()
    object_run_scores = dict()
    object_run_prediction = dict()
    
    # for each motion
    for motion in test_feature_objs:
        results = []
        truth_vector = []
        motion_test_obj_list = test_feature_objs[motion]
        object_run_probability[motion] = []

        for test_obj in motion_test_obj_list:
            if test_obj.labels == None:
                prediction, probability = utilities.compute_adjective_probability_score(classifier_dict, test_obj, feature_list, adjective_name, scaler_dict)
            else:
                prediction, probability, truth = utilities.compute_adjective_probability_score(classifier_dict, test_obj, feature_list, adjective_name, scaler_dict)
                results.append(prediction)  
                truth_vector.append(truth)
            # Store the probabilities
            object_run_probability[motion].append(probability)
            object_run_prediction[motion] = prediction

        print "Motion is: %s" % motion
        if len(results) > 0: 
            print "f1 score is %s" %classification_report(truth_vector, results)
            
            # Store the scores
            object_run_scores[motion] = (recall_score(truth_vector, results))

    # Return the % of motions in a n x 5 vector (5 motions)
    probability_feature_vector = [] 
    for val in xrange(len(object_run_probability[motion])):
       
        one_run_vector = []
        for motion_name in object_run_probability:
            one_run_vector.append(object_run_probability[motion_name][val])

        probability_feature_vector.append(one_run_vector)

    return (np.array(probability_feature_vector), object_run_scores, object_run_prediction)

# MAIN FUNCTION
def main(classifiers_pkl, all_classifiers_pkl, test_feature_pkl, ensemble_test_feature_pkl, scaler_pkl):
    
    # Load data into pipeline
    print "Loading data from file"
   
    file_ptr = open(scaler_pkl)
    scaler_dict = cPickle.load(file_ptr)
    file_ptr = open(test_feature_pkl)
    test_feature_dict = cPickle.load(file_ptr)
    file_ptr = open(ensemble_test_feature_pkl)
    ensemble_feature_dict = cPickle.load(file_ptr)

    feature_name_list = ["pdc_rise_count", "pdc_area", "pdc_max", "pac_energy", "pac_sc", "pac_sv", "pac_ss", "pac_sk", "tac_area", "tdc_exp_fit", "gripper_min", "gripper_mean", "transform_distance", "electrode_polyfit"]
    if all_classifiers_pkl == None:
        file_ptr = open(classifiers_pkl)
        classifiers_dict = cPickle.load(file_ptr)

        # Parse the adjective classifier name being tested
        adjective_name = classifiers_pkl.split('/')[-1].split('_')[0]
        test_adj_motion_classifier(classifiers_dict, adjective_name, ensemble_feature_dict, feature_name_list, scaler_dict) 
    else:
        file_ptr = open(all_classifiers_pkl)
        classifiers_dict = cPickle.load(file_ptr)

        # Get the name of the classifier
        classifier_name = all_classifiers_pkl.split('_')[1]

        # Open up text file to store best 
        report_best_classifier = open("best_classifier_report_" + classifier_name+ ".txt", "w")
        
        # Store the best classifier
        best_classifiers = dict()
        report_best_classifier.write('Adjective'+','+'tap'+','+'squeeze'+','+'thermal_hold'+','+'slide'','+'slide_fast,'+'best_motion'+'\n')
        results_prediction = dict()
        # Go through all of the adjective classifiers 
        for adj in classifiers_dict:
           # Test the adjective scores 
           probility_vector, motion_scores, prediction = test_adj_motion_classifier(classifiers_dict[adj], adj, ensemble_feature_dict, feature_name_list, scaler_dict)
           # Compute the best scores
           best_motion, best_score = utilities.get_best_motion(motion_scores)
           #report_best_classifier.write('Adjective: '+adj)
           #report_best_classifier.write('\nMotion Scores:\n' + str(motion_scores))
           #report_best_classifier.write('\n\nBest Motion is: ' +best_motion+ "\n\n")
           report_best_classifier.write(adj+','+str(motion_scores['tap'])+','+str(motion_scores['squeeze'])+','+str(motion_scores['thermal_hold'])+','+str(motion_scores['slide'])+','+str(motion_scores['slide_fast'])+','+best_motion+'\n')
           results_prediction[adj] = prediction[best_motion]

           best_classifiers[adj] = (classifiers_dict[adj][best_motion], best_motion)

        # Store the pickle file
        cPickle.dump(best_classifiers, open("best_classifiers_"+classifier_name+".pkl", "w"))
        report_best_classifier.close()
        print results_prediction

# Parse the command line arguments
def parse_arguments():
    """ 
    Parses the arguments provided at command line.
    
    Returns:
    (input_file, adjective_file, range)
    """
    parser = OptionParser()
    parser.add_option("-c", "--classifier_file", action="store", type="string", dest = "classifiers")
    parser.add_option("-a", "--all_classifier_file", action="store", type="string", dest = "all_classifiers")
    parser.add_option("-t", "--input_test_feature_pkl", action="store", type="string", dest = "in_test_feature_pkl", default = None) 
    parser.add_option("-e", "--input_ensemble_test_feature_pkl", action="store", type="string", dest = "in_ensemble_test_feature_pkl", default = None) 
    parser.add_option("-s", "--input_scale_pkl", action="store", type="string", dest = "in_scale_pkl", default = None) 

    (options, args) = parser.parse_args()
  
    classifiers_pkl = options.classifiers 
    all_classifiers_pkl = options.all_classifiers
    test_feature_pkl = options.in_test_feature_pkl
    ensemble_test_feature_pkl = options.in_ensemble_test_feature_pkl
    scaler_pkl = options.in_scale_pkl

    return classifiers_pkl, all_classifiers_pkl, test_feature_pkl, ensemble_test_feature_pkl, scaler_pkl

if __name__ == "__main__":
    classifiers_pkl, all_classifiers_pkl, test_feature_pkl, ensemble_test_feature_pkl, scaler_pkl = parse_arguments()
    main(classifiers_pkl, all_classifiers_pkl, test_feature_pkl, ensemble_test_feature_pkl, scaler_pkl)

