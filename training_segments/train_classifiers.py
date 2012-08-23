#! /usr/bin/python
from adjective_classifier import AdjectiveClassifier
import cPickle
import sys
from sklearn.metrics import f1_score, precision_score, recall_score


def train_single_classifier(clf):
    assert isinstance(clf, AdjectiveClassifier)
    print "Training classifier on ", clf.adjective
    clf.train_on_features()
    
    clf_out = clf.predict(clf.features)
    f1 = f1_score(clf.labels, clf_out)
    pr = precision_score(clf.labels, clf_out)
    rec = recall_score(clf.labels, clf_out)
    s = clf.score(clf.features, clf.labels)
    
    print "%s: Score: %f, F1: %f, Precision: %f, Recall: %f, Gamma: %f" %(
        clf.adjective, s, f1, pr, rec, clf.svc.gamma)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: %s classifiers_file  output_file" % sys.argv[0]
        sys.exit(0)
    
    classifiers = cPickle.load(open(sys.argv[1], "r"))
    
    for c in classifiers:
        train_single_classifier(c)
    
    print "Saving"
    cPickle.dump(classifiers, open(sys.argv[2], "w"), cPickle.HIGHEST_PROTOCOL)