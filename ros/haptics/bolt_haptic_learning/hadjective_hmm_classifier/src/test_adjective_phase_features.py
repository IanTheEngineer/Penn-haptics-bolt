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


def test_adjective(classifier, adjective_report):
        
    adjective_report.write("\n%s, " % classifier['adjective'])
    
    for phase in phases:
           
        true_positives = 0.0
        true_negatives = 0.0
        false_positives = 0.0
        false_negatives = 0.0

        false_positive_list = []
        false_negative_list = []
        true_positive_list = []
        true_negative_list = []


        print '\n \nTesting Adjective: %s and phase %s' % (classifier['adjective'], phase)

        #Pull out test features/labels
        test_set = classifier[phase]['test']
        test_X = test_set['features']
        test_Y = test_set['labels']
        object_ids = test_set['object_ids']
        object_names = test_set['object_names']

        # Pull out the classifier and merge features
        clf = classifier[phase]['classifier']

        # Predict the labels!
        if 'scaler' in classifier[phase]:
            if type(classifier[phase]['scaler']) == preprocessing.Scaler:
                test_X = classifier[phase]['scaler'].transform(test_X)

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
        print "Precision: %f, Recall: %f, F1: %f \n" % (precision, recall, f1)
    
        adjective_report.write("%f, " % f1)

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

    for adjective in adjectives:
        try:
            classifier = classifiers_dict[adjective]
            classifier['adjective'] = adjective

            # Compute score for each adjective 
            p, r, f1 = test_adjective(classifier, adjective_report)
            precs += p
            recalls += r
            f1s += f1
            total += 1

        except ValueError:
            print "Skipping values"
            continue
            

    adjective_report.close()

    print "Average f1s: ", f1s / total
    print "Average precision: ", precs / total
    print "Average recall: ", recalls / total


