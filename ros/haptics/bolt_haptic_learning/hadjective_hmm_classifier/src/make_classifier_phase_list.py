#! /usr/bin/python
from adjective_classifier import AdjectiveClassifier
import cPickle
import sys
import tables
import utilities
import os
from collections import defaultdict

 
path = sys.argv[1]
classifier_dict = defaultdict(dict) 

listing = os.listdir(path)
for infile in listing:

    test = cPickle.load(open(path + infile))
    print "current file is: " + path+ infile

    chars = infile.strip(".pkl").split("_")
    chars = chars[1:] #trained
    adjective = chars[0] #adjective
    chars = chars[1:] #adjective
    phase = "_".join(chars) # merge together
 
    classifier_dict[adjective][phase] = test    



cPickle.dump(classifier_dict, open(path + 'all_trained_adjectives_phase.pkl', "w"), cPickle.HIGHEST_PROTOCOL)

