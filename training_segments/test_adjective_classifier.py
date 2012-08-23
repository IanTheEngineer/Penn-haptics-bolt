#! /usr/bin/python
import adjective_classifier
import cPickle

adjective ="sticky"
chains_path = "/home/pezzotto/log/bigbags/bag_files/databases/PennData/chains"
adjectives_path = "/home/pezzotto/log/bigbags/bag_files/databases/PennData"
database = "/home/pezzotto/log/bigbags/bag_files/databases/PennData/all_objects_majority_four.h5"
clf = adjective_classifier.AdjectiveClassifier(adjective, chains_path)
print "Classifier created"

features, labels = clf.create_features_set(database)
print "Features: \n", features
print "Labels: \n", labels