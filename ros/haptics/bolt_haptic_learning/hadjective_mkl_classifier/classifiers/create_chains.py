#! /usr/bin/python
import roslib; roslib.load_manifest('hadjective_mkl_classifier')
import rospy
import cPickle
import hadjective_hmm_classifier.training_segments.hmm_chain as hmm_chain
import os
import sys
import itertools
import tables
from collections import defaultdict
from sklearn.externals.joblib import Parallel, delayed
from hadjective_hmm_classifier.training_segments.adjective_classifier import AdjectiveClassifier
import numpy as np

#Check if directory exits & create it
def check_dir(f): 
    if not os.path.exists(f):
        os.makedirs(f)
        return True
    return False


def main():
    if len(sys.argv) == 3:
        database, path = sys.argv[1:]
        #n_jobs = int(n_jobs)
        print "Training all combinations of adjectives and phases"
        #p = Parallel(n_jobs=n_jobs,verbose=10)
        #p(delayed(create_single_dataset)(database, path, adjective, phase)
        #for adjective, phase in itertools.product(adjectives,
        #                                          phases))
        base_directory = path
        untrained_directory = os.path.join(base_directory, "untrained_adjectives")
        hmm_feature_directory = os.path.join(base_directory, "adjective_phase_set")
        check_dir(hmm_feature_directory)
    
        # load up all chains and save it in one pickle file
        chains_dict = defaultdict(dict)
        for adj_f in os.listdir(untrained_directory):
            full_adj_path = os.path.join(untrained_directory, adj_f)
            adj_obj = cPickle.load(open(full_adj_path))
            chains_dict[adj_f] = adj_obj

        # Save off loaded chains
        save_file_name = "_".join((untrained_directory,"loaded_chains.pkl")) 
        with open(save_file_name, "w") as f:
            print "Saving file: ", save_file_name 
            print " " 
            cPickle.dump(chains_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)

    else:
        print "Usage:"
        print "%s database path adjective phase n_jobs" % sys.argv[0]
        print "%s database path adjective n_jobs" % sys.argv[0]
        print "%s database path" % sys.argv[0]
        print "Files will be saved in path/adjective_phase_set"

if __name__ == "__main__":
    main()
    print "done"

