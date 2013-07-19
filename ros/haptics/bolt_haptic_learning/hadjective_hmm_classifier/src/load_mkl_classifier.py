#!/usr/bin/env python

import cPickle
import os
import sys 
import itertools
import utilities
from utilities import adjectives, phases, sensors, static_features
import multiprocessing
import tables
import traceback
import numpy as np
from static_feature_obj import StaticFeatureObj
import upenn_features
from collections import defaultdict


def load_mkl(classifier_file):

    all_classifiers = cPickle.load(open(classifier_file))
    for adjective_classifier in all_classifiers:
        
        # Pull out the static kernel
        static_kernel
    
    import pdb; pdb.set_trace()



def main():
    if len(sys.argv) == 2:
        load_mkl(sys.argv[1])
    else:
        print "Usage:"
        print "%s path" % sys.argv[0]
        print "Path to the base directory"

if __name__=="__main__":
    main()
    print "done"

