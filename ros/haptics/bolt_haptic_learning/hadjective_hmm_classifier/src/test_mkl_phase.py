#! /usr/bin/python
from adjective_classifier import AdjectiveClassifier
import cPickle
import sys
import tables
import utilities
from utilities import adjectives, phases
from extract_static_features import get_train_test_objects
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from mkl_xvalidation import standardize, linear_kernel
from sklearn.metrics.pairwise import pairwise_kernels
from collections import defaultdict



def linear_kernel_test(testK, origK, n_jobs):
    return pairwise_kernels(testK, origK, metric="linear", n_jobs=n_jobs)


def test_adjective(classifier, adjective_report, store_values):
       
    adjective = classifier['adjective']
    phase = classifier['phase']
  
    dynamic_features = utilities.load_adjective_phase('/media/data_storage/vchu/all_classifiers/icra2014/dynamic/adjective_phase_set/')
    static_features = utilities.load_adjective_phase('/media/data_storage/vchu/all_classifiers/icra2014/static/adjective_phase_set/')

    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    false_positive_list = []
    false_negative_list = []
    true_positive_list = []
    true_negative_list = []

    print '\n \nTesting Adjective: %s and phase %s' % (adjective, phase)

    #Pull out test features/labels
   
    dynamic_train = dynamic_features[adjective][phase]['train']['features']
    dynamic_train_scaler = preprocessing.StandardScaler().fit(dynamic_train)
    dynamic_train_scaled_X = dynamic_train_scaler.transform(dynamic_train)
    dynamic_train_kernel = linear_kernel(dynamic_train_scaled_X, -2)
    #dynamic_train_kernel = standardize(dynamic_train_kernel)
    
    #Static Train
    static_train = static_features[adjective][phase]['train']['features']
    static_train_scaler = preprocessing.StandardScaler().fit(static_train)
    static_train_scaled_X = static_train_scaler.transform(static_train)
    static_train_kernel = linear_kernel(static_train_scaled_X, -2)

    dynamic_test = dynamic_features[adjective][phase]['test']['features'] 
    dynamic_test_scaled_X = classifier['dynamic_scaler'].transform(dynamic_test)
    dynamic_kernel = linear_kernel_test(dynamic_test_scaled_X, dynamic_train_scaled_X, -2)
    #dynamic_kernel = (dynamic_kernel - classifier['dynamic_kernel_mean']) / classifier['dynamic_kernel_std']
    
    static_test = static_features[adjective][phase]['test']['features'] 
    static_test_scaled_X = classifier['static_scaler'].transform(static_test)
    static_kernel = linear_kernel_test(static_test_scaled_X, static_train_scaled_X, -2)   

    alpha = classifier['alpha']

    test_X = (alpha)*static_kernel + (1-alpha)*dynamic_kernel
    test_Y = dynamic_features[adjective][phase]['test']['labels']
    
    object_ids = dynamic_features[adjective][phase]['test']['object_ids']
    object_names = dynamic_features[adjective][phase]['test']['object_names']

    # Pull out the classifier and merge features
    clf = classifier['classifier']

    # Predict the labels!
    output = clf.predict(test_X)
    
    # Determine if the true label and classifier prediction match
    for val in xrange(len(test_Y)):
        true_label = test_Y[val]
        predict_label = output[val]

        if true_label == 1:
            if predict_label == 1:
                true_positives += 1.0
                true_positive_list.append(object_names[val])
            else:
                false_negatives += 1.0
                false_negative_list.append(object_names[val])
        else: # label is 0
            if predict_label == 1:
                false_positives += 1.0
                false_positive_list.append(object_names[val])
            else:
                true_negatives += 1.0
                true_negative_list.append(object_names[val])
    
    # Compute statistics for the adjective
    try: 
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
    
    except ZeroDivisionError: # The case when none are found
        precision = 0
        recall = 0
    try:
        f1 = 2.0 * precision*recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    
    store_values[adjective+'_'+phase] = (f1,precision, recall)
    print "%d False Positive Objects are: %s \n" % (false_positives, sorted(false_positive_list))
    print "%d False Negative Objects are: %s \n" % (false_negatives, sorted(false_negative_list))
    print "%d True Positive Objects are: %s\n" % (true_positives, sorted(true_positive_list))
    print "%d True Negative Objects are: %s\n" % (true_negatives, sorted(true_negative_list))

    
    return (precision, recall, f1)
            

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "Usage %s classifiers" % sys.argv[0]
        sys.exit(1)

    # Load database and classifiers (adjectives) 
    classifiers_dict = cPickle.load(open(sys.argv[1]))

    # Initialize scores
    f1s= 0
    precs = 0
    recalls = 0
    total = 0
  
    # Setup text file to store values to
    adjective_report = open("adjective_phase_score_report.txt", "w")
    adjective_report.write("Adjective, %s, %s, %s, %s" % tuple(phases))
    store_scores = defaultdict(dict)

    for mkl in classifiers_dict:
        try:
            # Compute score for each adjective 
            p, r, f1 = test_adjective(mkl, adjective_report, store_scores)
            precs += p
            recalls += r
            f1s += f1
            total += 1

        except ValueError:
            print "Skipping values"
            continue
            

    for adjective in adjectives:
        adjective_report.write("\n%s, " % adjective)
        
        for phase in phases:
            print "Current phase: %s" % phase
            sp,sr,sf1 = store_scores[adjective+'_'+phase]
            print "Precision: %f, Recall: %f, F1: %f \n" % (sp, sr, sf1)

            adjective_report.write("%f, " % sf1)


            

    adjective_report.close()

    print "Average f1s: ", f1s / total
    print "Average precision: ", precs / total
    print "Average recall: ", recalls / total


