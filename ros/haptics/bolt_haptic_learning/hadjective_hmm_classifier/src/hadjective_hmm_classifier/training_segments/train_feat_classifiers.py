#! /usr/bin/python
from feat_adjective_classifier import FeaturesAdjectiveClassifier
import cPickle
import sys
from sklearn.externals.joblib import Parallel, delayed
import os

def train_adjective(f, untrained_directory, trained_directory, test_database):
    full_path = os.path.join(untrained_directory, f)
    #print "Opening file: %s" % f
    clf = cPickle.load(open(full_path))
    assert isinstance(clf, FeaturesAdjectiveClassifier)
    print "Training classifier %s" % clf.adjective
    
    clf.train_gridsearch(n_jobs=10)
    
    dest_filename = os.path.join(trained_directory, f)
    #print "Saving file ", dest_filename
    cPickle.dump(clf, open(dest_filename, "w"), cPickle.HIGHEST_PROTOCOL)                       

def test_file(f, trained_directory):
    if not f.endswith(".pkl"):
        return False
    if f in os.listdir(trained_directory):
        print "File %s already exist, skipping it" % f 
        return False
    else:
        return True

def load_single(f, untrained_directory):
    full_path = os.path.join(untrained_directory, f)
    print "Opening file: %s" % full_path
    clf = cPickle.load(open(full_path))
    assert isinstance(clf, FeaturesAdjectiveClassifier)
    print "Loading features for classifier %s" % clf.adjective    
    clf.load_test_test(test_database)
    cPickle.dump(clf, open(full_path, "w"), cPickle.HIGHEST_PROTOCOL)

def load_features_only(base_directory, test_database, n_jobs):
    untrained_directory = os.path.join(base_directory, "untrained_adjectives") 
        
    p = Parallel(n_jobs=n_jobs,verbose=10)
    p(delayed(load_single)(f, untrained_directory)
      for f in os.listdir(untrained_directory)
      if f.endswith(".pkl")
      )    

def main(base_directory, test_database, n_jobs):
    untrained_directory = os.path.join(base_directory, "untrained_adjectives")
    trained_directory = os.path.join(base_directory, "trained_adjectives")
    
    p = Parallel(n_jobs=n_jobs,verbose=10)
    p(delayed(train_adjective)(f, untrained_directory, 
                               trained_directory, test_database)
      for f in os.listdir(untrained_directory)
      if test_file(f, trained_directory))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "Usage: %s base_directory test_database njobs yes|no" % sys.argv[0]
        print "adjectives are in base_directory/untrained_adjectives"
        print "New classifiers will be saved in base_directory/trained_adjectives"
        print "If yes then the whole training will be performed, otherwise only\
                loading the test set will be performed"
        sys.exit(0)
    
    base_directory, test_database, n_jobs, yesno= sys.argv[1:]
    n_jobs = int(n_jobs)    
    if yesno == "yes":
        print "Doing whole training"
        main(base_directory, test_database, n_jobs)
    else:
        print "Just loading the test dataset"
        load_features_only(base_directory, test_database, n_jobs)
    