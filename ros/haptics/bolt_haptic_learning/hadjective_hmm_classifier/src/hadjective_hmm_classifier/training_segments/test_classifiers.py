#! /usr/bin/python
from adjective_classifier import AdjectiveClassifier
import cPickle
import sys
import tables
import utilities

to_remove = ['porous', 'elastic', 'grainy']

def test_object(obj, classifiers):
    assert isinstance(obj, tables.Group)
    data_dict = utilities.dict_from_h5_group(obj)    
    true_adjectives = [a for a in sorted(data_dict["adjectives"])
                       if a not in to_remove]

    if len(true_adjectives) == 0:
        print "Object in database has no adjectives!"
        test_classifier = False
    else:
        test_classifier = True
        print "Object %s has adjectives %s" %(data_dict["name"],
                                              " ".join(true_adjectives)
                                              )
    print "Positive classifiers:"    
    
    positives = []
    for clf in classifiers:
        if clf.adjective in to_remove:
            continue
        assert isinstance(clf, AdjectiveClassifier)
        features = clf.extract_features(data_dict["data"])
        output = clf.predict(features)
        if output[0] == 1:
            positives.append(clf.adjective)
   
    if not test_classifier:
        print "Results can't be shown"
        raise ValueError()
    if len(positives) == 0:
        "No classifiers output!"
        return (0.0, 0.0, 0.0)

    positives = sorted(positives)
    print "\t" + " ".join(positives)
    
    cls_set = set(positives)
    true_set = set(true_adjectives)
    intersection  = cls_set & true_set
    difference = true_set - cls_set

    true_length = float(len(true_set))
    clf_length = float(len(cls_set))

    true_positives = len(intersection) / clf_length
    false_negatives = (true_length - len(intersection)) / true_length
    false_positives = len(cls_set - true_set) / clf_length
    
    print "True posititives %f, False positivies %f, False negatives %f" %( true_positives,
                                                                        false_positives,
                                                                       false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    try:
        f1 = 2.0 * precision*recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    print "Precision: %f, Recall: %f, F1: %f" % (precision, recall, f1)

    return (precision, recall, f1)

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print "Usage %s h5_database classifiers" % sys.argv[0]
        sys.exit(1)
    
    database = tables.openFile(sys.argv[1])
    
    classifiers = cPickle.load(open(sys.argv[2]))
    
    f1s= 0
    precs = 0
    recalls = 0
    total = 0
    
    for group in utilities.iterator_over_object_groups(database):
        try:
            p, r, f1 = test_object(group, classifiers)
            precs += p
            recalls += r
            f1s += f1
            total += 1
        except ValueError:
            print "Skipping values"
            continue
    
    print "Average f1s: ", f1s / total
    print "Average precision: ", precs / total
    print "Average recall: ", recalls / total
    
