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


c_dict = defaultdict()

def test_adjective(classifier, adjective_report ):
    
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    false_positive_list = []
    false_negative_list = []
    true_positive_list = []
    true_negative_list = []

    adjective = classifier['adjective']

    dynamic_features = utilities.load_adjective_phase('/home/imcmahon/Desktop/mkl/dynamic/adjective_phase_set')
    static_features = utilities.load_adjective_phase('/home/imcmahon/Desktop/mkl/static/adjective_phase_set')
    #dynamic_features = utilities.load_adjective_phase(dynamic_path)
    #static_features = utilities.load_adjective_phase(static_path)
    

    #import pdb; pdb.set_trace()
    #Dynamic Train
    dynamic_train = utilities.get_all_train_test_features(adjective, dynamic_features, train=True)
    dynamic_train_scaler = preprocessing.StandardScaler().fit(dynamic_train[0])
    dynamic_train_scaled_X = dynamic_train_scaler.transform(dynamic_train[0])
    dynamic_train_kernel = linear_kernel(dynamic_train_scaled_X, -2) 
    dynamic_train_kernel = standardize(dynamic_train_kernel)
    #Static Train
    static_train = utilities.get_all_train_test_features(adjective, static_features, train=True)
    static_train_scaler = preprocessing.StandardScaler().fit(static_train[0])
    static_train_scaled_X = static_train_scaler.transform(static_train[0])
    static_train_kernel = linear_kernel(static_train_scaled_X, -2) 
    static_train_kernel = standardize(static_train_kernel)
    #Recompute the GRAM matrix
    #alpha = classifier['alpha'];
    #train_X = (alpha)*static_train_kernel + (1-alpha)*dynamic_train_kernel

    #import pdb; pdb.set_trace()
    dynamic_test = utilities.get_all_train_test_features(adjective, dynamic_features, train=False)
    dynamic_test_scaled_X = classifier['dynamic_scaler'].transform(dynamic_test[0])
    dynamic_kernel = linear_kernel_test(dynamic_test_scaled_X, dynamic_train_scaled_X, -2)
    dynamic_kernel = (dynamic_kernel - classifier['dynamic_kernel_mean']) / classifier['dynamic_kernel_std']
    
    static_test = utilities.get_all_train_test_features(adjective, static_features, train=False)
    static_test_scaled_X = classifier['static_scaler'].transform(static_test[0])
    static_kernel = linear_kernel_test(static_test_scaled_X, static_train_scaled_X, -2)
    static_kernel = (static_kernel - classifier['static_kernel_mean']) / classifier['static_kernel_std']

    alpha = classifier['alpha'];

    test_X = (alpha)*static_kernel + (1-alpha)*dynamic_kernel

    print '\n \nTesting Adjective: %s' % classifier['adjective']
    
    #Pull out test features/labels

    #for phase in phases:
    #import pdb; pdb.set_trace()
    #test_set = classifier['test']
    #test_X.append(test_set['features'])
    #test_X.append(test_set['features'])
    test_Y = dynamic_test[1]
    object_ids = dynamic_test[2]
    #object_names = test_set['object_names']

    
    # Pull out the classifier and merge features
    #test_X = np.concatenate(test_X, axis=1)
    #import pdb; pdb.set_trace()
    
    clf = classifier['classifier']
    c_dict[classifier['adjective']] = clf.C
    #import pdb; pdb.set_trace()
    print clf

    # Predict the labels!
    #if 'scaler' in classifier:
    #    if type(classifier['scaler']) == preprocessing.Scaler:
    #        test_X = classifier['scaler'].transform(test_X)
            
    #import pdb; pdb.set_trace() 
    output = clf.predict(test_X)
    # Determine if the true label and classifier prediction match
    for val in xrange(len(test_Y)):
        true_label = test_Y[val]
        predict_label = output[val]

        if true_label == 1:
            if predict_label == 1:
                true_positives += 1.0
                #true_positive_list.append(object_names[val])
            else:
                false_negatives += 1.0
                #false_negative_list.append(object_names[val])
        else: # label is 0
            if predict_label == 1:
                false_positives += 1.0
                #false_positive_list.append(object_names[val])
            else:
                true_negatives += 1.0
                #true_negative_list.append(object_names[val])
    
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
    print "Precision: %f, Recall: %f, F1: %f \n" % (precision, recall, f1)
    print "Alpha = %1.1f" % alpha
    adjective_report.write("%s, %1.1f, %f, %f, %f\n" % (classifier['adjective'], alpha, precision, recall, f1))

    print "%d False Positive Objects\n" % false_positives
    print "%d False Negative Objects\n" % false_negatives
    print "%d True Positive Objects\n" % true_positives
    print "%d True Negative Objects\n" % true_negatives
    
    return (precision, recall, f1)
            

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "Usage %s classifiers" % sys.argv[0]
        sys.exit(1)

    # Load database and classifiers (adjectives) 
    classifiers = cPickle.load(open(sys.argv[1]))

    # Initialize scores
    f1s= 0
    precs = 0
    recalls = 0
    total = 0
    
    # Setup text file to store values to
    adjective_f1_dict = open("adjective_f1_dict.txt", "w")
    adjective_f1_dict.write("d = {")
    adjective_report = open("adjective_score_report.txt", "w")
    adjective_report.write("Adjective, alpha, precision, recall, f1\n")

    for classifier in classifiers:
        #try:
            # Compute score for each adjective 
            p, r, f1 = test_adjective(classifier, adjective_report)
            precs += p
            recalls += r
            f1s += f1
            total += 1
            adjective_f1_dict.write("'%s': %1.5f,\n"%(classifier['adjective'], f1))

        #except ValueError:
        #    print "Skipping values"
        #    continue

    adjective_f1_dict.write("]")
    adjective_f1_dict.close()
    adjective_report.close()

    print "Average f1s: ", f1s / total
    print "Average precision: ", precs / total
    print "Average recall: ", recalls / total
    print c_dict

