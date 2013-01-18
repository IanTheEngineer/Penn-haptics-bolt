#! /usr/bin/python
from adjective_classifier import AdjectiveClassifier
import cPickle
import sys
import tables
import utilities
import os
 
path = sys.argv[1]
classifier_list = []

listing = os.listdir(path)
for infile in listing:

    test = cPickle.load(open(path + infile))
    classifier_list.append(test)    
    print "current file is: " + path+ infile


cPickle.dump(classifier_list, open(path + 'all_trained_adjectives.pkl', "w"), cPickle.HIGHEST_PROTOCOL)

