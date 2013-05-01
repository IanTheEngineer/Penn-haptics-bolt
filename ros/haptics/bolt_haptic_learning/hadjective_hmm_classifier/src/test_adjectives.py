#! /usr/bin/python
from adjective_classifier import AdjectiveClassifier
import cPickle
import sys
import tables
import utilities
from utilities import adjectives
from extract_static_features import get_train_test_objects

def test_adjective(classifier, database, test_object_names, adjective_report):
           
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    false_positive_list = []
    false_negative_list = []
    true_positive_list = []
    true_negative_list = []


    print '\n \nTesting Adjective: %s' % classifier.adjective
    
    for group in utilities.iterator_over_object_groups(database):
        
        assert isinstance(group, tables.Group)
        data_dict = utilities.dict_from_h5_group(group)

        if data_dict['name'] not in test_object_names:
            continue
        
        features = classifier.extract_features(data_dict["data"])
        output = classifier.predict(features)
      
        # For this object - find out if the adjective applies
        # True label is 0 if adjective is false for this adjective
        true_labels = data_dict['adjectives']
        if classifier.adjective in true_labels:
            true_label = 1
        else:
            true_label = 0

        # Determine if the true label and classifier prediction match
        if true_label == 1:
            if output[0] == 1:
                true_positives += 1.0
                true_positive_list.append(data_dict['name'])
            else:
                false_negatives += 1.0
                false_negative_list.append(data_dict['name'])
        else: # label is 0
            if output[0] == 1:
                false_positives += 1.0
                false_positive_list.append(data_dict['name'])
            else:
                true_negatives += 1.0
                true_negative_list.append(data_dict['name'])

    # Compute statistics for the adjective
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    try:
        f1 = 2.0 * precision*recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    print "Precision: %f, Recall: %f, F1: %f \n" % (precision, recall, f1)
    adjective_report.write("%s, %f, %f, %f\n" % (classifier.adjective, precision, recall, f1))

    print "%d False Positive Objects are: %s \n" % (false_positives, sorted(false_positive_list))
    print "%d False Negative Objects are: %s \n" % (false_negatives, sorted(false_negative_list))
    print "%d True Positive Objects are: %s\n" % (true_positives, sorted(true_positive_list))
    print "%d True Negative Objects are: %s\n" % (true_negatives, sorted(true_negative_list))
    
    return (precision, recall, f1)
            

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print "Usage %s h5_database classifiers" % sys.argv[0]
        sys.exit(1)

    # Load database and classifiers (adjectives) 
    database = tables.openFile(sys.argv[1])
    classifiers = cPickle.load(open(sys.argv[2]))

    import pdb; pdb.set_trace()
    # Initialize scores
    f1s= 0
    precs = 0
    recalls = 0
    total = 0
    
    # Setup text file to store values to
    adjective_report = open("adjective_score_report.txt", "w")
    adjective_report.write("Adjective, precision, recall, f1\n")

    for classifier in classifiers:
        try:
            # Pull out the objects that we want
            train_objs, test_objs = get_train_test_objects(database, classifier.adjective)

            # Compute score for each adjective 
            p, r, f1 = test_adjective(classifier, database, test_objs, adjective_report)
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


