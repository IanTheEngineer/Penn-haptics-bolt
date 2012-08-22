#! /usr/bin/python
from adjective_classifier import AdjectiveClassifier
import cPickle
import utilities
from sklearn.externals.joblib import Parallel, delayed
import sys
import tables
import thread

adjectives = utilities.adjectives
#adjectives = ["sticky"]

h5_db = None
chains_directory = None


def create_clf(a):
    print "Creating classifier for adjective ", a
    clf = AdjectiveClassifier(a, chains_directory)
    clf.create_features_set(h5_db)
    return clf   

def main(db_filename, base_directory, out_filename, nj=6):
    
    global h5_db
    global chains_directory
    
    h5_db = db_filename
    chains_directory = base_directory
    f = open(out_filename, "w")
    
    p = Parallel(n_jobs=n_jobs,verbose=10)
    classifiers = p(delayed(create_clf)(a) 
                    for a in adjectives)
    
    cPickle.dump(classifiers, f, cPickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "usage: %s database base_directory out_file n_jobs" % sys.argv[0]
        sys.exit(1)
    
    db, base_dir, out_file, n_jobs = sys.argv[1:]
    n_jobs = int(n_jobs)
    main(db, base_dir, out_file, n_jobs)

