#! /usr/bin/python
from adjective_classifier import AdjectiveClassifier
import cPickle
import sys
import tables
import utilities

def test_object(obj, classifiers):
    assert isinstance(obj, tables.Group)
    data_dict = utilities.dict_from_h5_group(obj)    
    true_adjectives = sorted(data_dict["adjectives"])
    print "Object %s has adjectives %s" %(data_dict["name"],
                                          " ".join(true_adjectives)
                                          )
    print "Positive classifiers:"
    
    positives = []
    for clf in classifiers:
        assert isinstance(clf, AdjectiveClassifier)
        features = clf.extract_features(data_dict["data"])
        output = clf.predict(features)
        if output[0] == 1:
            positives.append(clf.adjective)
    
    positives = sorted(positives)
    print "\t" + " ".join(positives)
    
    perfect_match = all(i == j for i,j in zip(true_adjectives, positives))
    if perfect_match:
        print "Perfect match!!"
        return 1
    else:
        print "Non perfect match :("
        return 0
        

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print "Usage %s h5_database classifiers" % sys.argv[0]
        sys.exit(1)
    
    database = tables.openFile(sys.argv[1])
    
    classifiers = cPickle.load(open(sys.argv[2]))
    
    perfect_matches = 0
    total = 0
    
    for group in utilities.iterator_over_object_groups(database):
        perfect_matches += test_object(group, classifiers)
        total += 1
    
    print "\n\nNumber of perfect matches %d over %d objects" % (perfect_matches,
                                                                total)
    
    