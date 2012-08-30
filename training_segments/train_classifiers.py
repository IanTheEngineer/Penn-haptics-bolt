#! /usr/bin/python
from adjective_classifier import AdjectiveClassifier
import cPickle
import sys
from sklearn.externals.joblib import Parallel, delayed
import os

def train_adjective(f, untrained_directory, trained_directory, test_database):
    full_path = os.path.join(untrained_directory, f)
    print "Opening file: %s" % f
    clf = cPickle.load(open(full_path))
    assert isinstance(clf, AdjectiveClassifier)
    print "Training classifier %s" % clf.adjective
    
    train_X, train_Y = clf.features, clf.labels
    test_X, test_Y = clf.create_features_set(test_database)
    clf.train_on_separate_dataset(train_X = train_X,
                                  train_Y = train_Y,
                                  test_X = test_X,
                                  test_Y = test_Y,
                                  verbose=True
                                  )
    
    dest_filename = os.path.join(trained_directory, f)
    print "Saving file ", dest_filename
    cPickle.dump(clf, open(dest_filename, "w"))                       

def test_file(f, trained_directory):
    if not f.endswith(".pkl"):
        return False
    if f in os.listdir(trained_directory):
        print "File %s already exist, skipping it" % f 
        return False
    else:
        return True

def main(base_directory, test_database, n_jobs):
    untrained_directory = os.path.join(base_directory, "untrained_adjectives")
    trained_directory = os.path.join(base_directory, "trained_adjectives")
    
    p = Parallel(n_jobs=n_jobs,verbose=10)
    p(delayed(train_adjective)(f, untrained_directory, 
                               trained_directory, test_database)
      for f in os.listdir(untrained_directory)
      if test_file(f, trained_directory))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage: %s base_directory test_database njobs" % sys.argv[0]
        print "adjectives are in base_directory/untrained_adjectives"
        print "New classifiers will be saved in base_directory/trained_adjectives"
        sys.exit(0)
    
    base_directory, test_database, n_jobs = sys.argv[1:]
    n_jobs = int(n_jobs)
    main(base_directory, test_database, n_jobs)
    
    