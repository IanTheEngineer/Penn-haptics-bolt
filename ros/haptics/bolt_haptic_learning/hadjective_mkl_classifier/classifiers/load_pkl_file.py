#! /usr/bin/python
import cPickle
import os
import sys
import itertools
import numpy as np
from collections import defaultdict

def load_adjective_phase(file_path):
    
    features = cPickle.load(open(file_path))
    return features

def template_function(base_directory):
    """
    Example function on how to access all of the features
    stored in adjective_phase_set
    """

    all_features = load_adjective_phase(base_directory)
    import pdb; pdb.set_trace()

def main():
    if len(sys.argv) == 2:
        template_function(sys.argv[1])
    else:
        print "Usage:"
        print "%s path" % sys.argv[0]
        print "Path to the base directory"

if __name__=="__main__":
    main()
    print "done"        
