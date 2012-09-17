#!/usr/bin/env python
import roslib; roslib.load_manifest("hadjective_train_pipe")
import rospy
import numpy as np
import sys 
import os
import glob
from optparse import OptionParser
import cPickle
import bolt_learning_utilities as utilities
import extract_features as extract_features

from bolt_feature_obj import BoltFeatureObj
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
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



def test_adj_motion_classifier(test_feature_objects, adjective, classifiers, scalers, feature_name_list, file_ptr, comma_file_ptr):
    '''
    takes in a dictionary of all features
    
    returns a dictionary of weak classifiers for each feature
    '''


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

def compute_weighted_sum_prediction(probability_scores, weights_scores):
    '''
    Given probility of 5 motions and their weights - sum and return a decision
    '''
    #import pdb; pdb.set_trace()
    final_results = []
    for run in xrange(probability_scores.shape[0]):
        
        probability_run = probability_scores[run,:]
        
        prob_final = sum(probability_run*weights_scores)
        
        if prob_final > 0.5:
            final_results.append(0)
        else:
            final_results.append(1)

    return final_results


def test_ensembled_classifier(test_feature_objects, adjective, adj_motion_classifiers, ensembled_classifier, scalers, ensemble_scaler, feature_name_list, weights, file_ptr, comma_file_ptr):
    '''
    takes in a dictionary of all features
    
    returns a dictionary of weak classifiers for each feature
    '''
   
    # Pull out the features
    test_vector, test_labels, object_ids, weights_notneeded= build_ensemble_feature_vector(test_feature_objects, adjective, adj_motion_classifiers, scalers, feature_name_list)

    # Compute results given a test vector of motions and weights
    #results = compute_weighted_sum_prediction(test_vector, weights)

    # Scale the probability vectors
    scaled_test_vector = ensemble_scaler.transform(test_vector)

    # Compute results
    results = ensembled_classifier.predict(scaled_test_vector)

    print classification_report(test_labels, results)

    file_ptr.write('Adjective: ' + adjective)
    file_ptr.write('\n'+classification_report(test_labels, results)+ '\n\n')

    # Write results to comma split file to load
    comma_file_ptr.write(adjective+','+str(precision_score(test_labels,results))+','+str(recall_score(test_labels, results))+','+str(f1_score(test_labels,results))+'\n')

    return (results, test_labels, object_ids)

def test_ensembled_classifier_object(prediction_dict, label_dict, object_name, comma_file_ptr, report_ptr):

    run_prediction = dict()
    run_truth = dict()
    adjectives = []
    for adj in prediction_dict:

        adjective_truth = label_dict[adj]
        adjective_prediction = prediction_dict[adj]
        adjectives.append(adj)

        for run in xrange(len(adjective_truth)):
            if run in run_truth:
                run_prediction[run].append(adjective_prediction[run])
                run_truth[run].append(adjective_truth[run])
            else:
                run_prediction[run] = []
                run_truth[run] = []

    # Compute and write scores to file
    for obj in run_prediction:

        results = run_prediction[obj]
        test_labels = run_truth[obj]
        object_id = object_name['sticky'][obj]
       
        # Compute the scores now per run
        comma_file_ptr.write(str(object_id)+','+str(precision_score(test_labels,results))+','+str(recall_score(test_labels, results))+','+str(f1_score(test_labels,results))+'\n')
        report_ptr.write('Object ID: '+str(object_id)+'\n\n') 
        report_ptr.write(str(adjectives)+'\n') 
        report_ptr.write('Prediction: '+str(results) + '\nTruth: '+ str(test_labels)+'\n\n')
        report_ptr.write(classification_report(test_labels, results)+ '\n\n')
        

def train_ensemble_adjective_classifier(train_feature_objects, adjective, classifiers, scalers, feature_name_list): 
    '''
    Given build classifier of 5 motions - train the single joined classifier

    Returns a single classifier with its associated scaler
    
    IMPORTANT: the feature vector of probabilities needs to be created in a specific order
    'tap', 'squeeze', 'thermal_hold', 'slide', 'slide_fast'

    '''
   
    # Pull out the features
    probability_vector, probability_labels, object_ids, weights = build_ensemble_feature_vector(train_feature_objects, adjective, classifiers, scalers, feature_name_list)

    # Create scaler for the features
    scaler = preprocessing.Scaler().fit(probability_vector)

    # Train a single SVM
    svm = train_svm(probability_vector, probability_labels, object_ids)

    return (svm, scaler, weights)


# Training a single SVM given a training set
def train_svm(train_vector, train_labels, object_ids):
    """ 
    train_svm - expects a vector of features and a nx1 set of
                corresponding labels

    Returns a trained SVM classifier
    """
    
    # Create the obj_id_vector for cross validation
    lpl = cross_validation.LeavePLabelOut(object_ids, p=1,indices=True)
    # Grid search with nested cross-validation
    parameters = {'kernel': ['linear'], 'C': [1, 1e1, 1e2, 1e3, 1e4], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 'degree':[2,3,4,5]} 
    #parameters = {'kernel': ['linear'], 'C': [1, 1e1], 'gamma': [1, 1e-1, 1e-2]} 
    svm = GridSearchCV(SVC(probability=True), parameters, score_func=f1_score, cv=lpl)

    # Train the SVM using the best parameters
    svm.fit(train_vector, train_labels)
    svm_best = svm.best_estimator_
   
    return svm_best



# Train a single ensemble adjective from 5 classifiers
def build_ensemble_feature_vector(train_feature_objects, adjective, classifiers, scalers, feature_name_list):
    '''
    Builds the feature vector set for the ensemble training

    Returns a dictionary of motion + probabilities
    '''

    motion_prob = dict()
    train_labels = None
    object_ids = None
    weights = dict() 
    
    for motion in train_feature_objects:
        motion_train_set = train_feature_objects[motion]
        
        # pull out the features specified as a vector 
        train_feature_vector, train_label_dict = utilities.feature_obj_2_feature_vector(motion_train_set, feature_name_list)

        # create scaled vector
        train_feature_vector_scaled = scalers[motion].transform(train_feature_vector)

        clf = classifiers[motion]
        
        # Get results from classifier
        results = clf.predict(train_feature_vector_scaled)
        probabilities = clf.predict_proba(train_feature_vector_scaled)[:,1] 
       
        # Store off probabilities and their labels
        motion_prob[motion] = probabilities

        # Store off real labels
        train_labels = train_label_dict[1][adjective]

        # Store off the f1 score, which becomes the weight
        weights[motion] = f1_score(train_labels, results)

        # Store off object ids
        object_ids = train_label_dict[0]

    motion_vector = np.transpose(np.vstack((motion_prob['tap'], motion_prob['squeeze'], motion_prob['thermal_hold'], motion_prob['slide'], motion_prob['slide_fast'])))

    # Weight vector
    weight_vector = np.hstack((weights['tap'], weights['squeeze'], weights['thermal_hold'],weights['slide'], weights['slide_fast']))

    return (motion_vector, train_labels, object_ids, weight_vector)



def main(feature_objects_set_file, test_feature_objects_file, adj_motion_classifier_file, output_svm_filename):

    # Load data into the pipeline. First check
    # for feature object pkl files
    print "Loading data from file\n"

    # Load Input Set
    if feature_objects_set_file == None:
        raise Exception('No input feature object file')
    else:
        feature_objects_set = cPickle.load(open(feature_objects_set_file, 'r'))

    # Load novel objects test set
    if test_feature_objects_file == None:
        print 'Warning: No Novel test set given'
    else:
        test_feature_objects = cPickle.load(open(test_feature_objects_file, 'r'))

    # Load Classifiers
    if adj_motion_classifier_file == None:
        raise Exception('No adjective-motion specific classifier folder given')
    else:
        adj_motion_dir = adj_motion_classifier_file

    # Check output SVM filename
    if output_svm_filename == None:
        print 'Warning: No filename given - writing to all_svm_boost_classifiers.pkl'
        output_svm_filename = 'ensembled_svm_classifiers.pkl'
    
    # specify feature to be extracted
    feature_name_list = ["pdc_rise_count", "pdc_area", "pdc_max", "pac_energy", "pac_sc", "pac_sv", "pac_ss", "pac_sk", "tac_area", "tdc_exp_fit", "gripper_min", "gripper_mean", "transform_distance", "electrode_polyfit"]

    ensembled_classifiers = dict()
    ensembled_scalers = dict()
    adj_motion_classifiers = dict()
    
    adjective_list = feature_objects_set.keys()
    adjective_list.sort()

    # Reports for individual classifiers
    report_file = open('reports/Gradient_Boosting_Results.txt', 'w')
    results_file = open('reports/Gradient_Boosting_Results_comma.csv', 'w')
   
    # File pointers for the final classifier results
    report_final_file = open('reports/Ensembled_SVM_Results.txt', 'w')
    results_final_file = open('reports/Ensembled_SVM_Results_comma.csv', 'w')
    results_object_file = open('reports/FinalObject_Results_comma,csv', 'w')
    # Report for object
    file_report_final = open('reports/FinalObject_Report.txt','w')
    
    # Write headers to the CSV files
    results_file.write('Adjective,tap,squeeze,static hold,slide, fast slide, best motion\n')
    results_final_file.write('Adjective, precision, recall, f1 score\n')
    results_object_file.write('Object ID, precision, recall, f1 score\n')

    
    # import all adjectives 
    for file_name in glob.glob(adj_motion_dir+'/*.pkl'):
        print file_name
        one_adj_motion = cPickle.load(open(file_name, 'r'))
        adj_name = file_name.split('_')[-1]
        adj_name = adj_name.split('.')[0]

        adj_motion_classifiers[adj_name] = one_adj_motion
        

    # Store all of the scores by adjective
    object_results = dict()
    prediction_dict = dict()
    label_dict = dict()
    object_ids_dict = dict()
    weights = dict()
    
    # For each adjective - train a classifier
    for adj in adjective_list:
        print adj

        if adj  in ['porous', 'elastic', 'grainy']:
             continue 
        #if adj not in ['soft','scratchy']:
        #    continue

        # Pull out classifiers and scalers
        adj_motion_classifier = adj_motion_classifiers[adj][0]
        adj_motion_scaler = adj_motion_classifiers[adj][1]

        # Test adjective-motion specific classifier
        # Scaler is stored with the classifiers!
        report = test_adj_motion_classifier(feature_objects_set[adj]['test'], adj, adj_motion_classifier, adj_motion_scaler, feature_name_list, report_file, results_file)
        
        # Train ensembled svm classifier
        ensembled_classifiers[adj], ensembled_scalers[adj], weights[adj] = train_ensemble_adjective_classifier(feature_objects_set[adj]['test'], adj, adj_motion_classifier, adj_motion_scaler, feature_name_list)

        cPickle.dump((ensembled_classifiers[adj], ensembled_scalers[adj]), open('ensembled/svm_and_scaler_'+adj+'.pkl','w'), cPickle.HIGHEST_PROTOCOL)

        # Test ensembled svm classifier
        prediction_dict[adj], label_dict[adj], object_ids_dict[adj] = test_ensembled_classifier(test_feature_objects, adj, adj_motion_classifier, ensembled_classifiers[adj], adj_motion_scaler, ensembled_scalers[adj], feature_name_list, weights[adj], report_final_file, results_final_file)

        #object_results[adj] = (prediction_dict, label_dict, object_ids)

    # Test by object
    test_ensembled_classifier_object(prediction_dict, label_dict, object_ids_dict, results_object_file, file_report_final)

    report_file.close()
    results_file.close()
    report_final_file.close()
    results_final_file.close()
    results_object_file.close()
    file_report_final.close()

    cPickle.dump((ensembled_classifiers, ensembled_scalers, weights), open('ensembled/'+output_svm_filename, 'w'), cPickle.HIGHEST_PROTOCOL)

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
    parser.add_option("-c", "--adjective_motion_classifiers", action="store", type="string", dest = "adjective_motion_classifier_file")
    parser.add_option("-o", "--output_svm", action="store", type="string", dest = "output_svm_file", default = None)
    parser.add_option("-t", "--test_set", action="store", type="string", dest = "test_feature_obj", default = None)

    (options, args) = parser.parse_args()

    feature_objects_set = options.feature_obj_file
    adjective_motion_classifier_file = options.adjective_motion_classifier_file 
    output_svm_filename = options.output_svm_file
    test_objects_set = options.test_feature_obj

    return (feature_objects_set, test_objects_set, adjective_motion_classifier_file, output_svm_filename)

if __name__ == "__main__":
    feature_objects_set, test_objects_set, adj_motion_classifiers, output_svm_filename = parse_arguments()
    main(feature_objects_set, test_objects_set, adj_motion_classifiers, output_svm_filename)

