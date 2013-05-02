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



if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "Usage %s classifiers" % sys.argv[0]
        sys.exit(1)

    # Load database and classifiers (adjectives) 
    classifiers = cPickle.load(open(sys.argv[1]))

    import pdb; pdb.set_trace()

    print "loaded"
