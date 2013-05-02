#! /usr/bin/python
import sklearn
import numpy as np
import os
import sys
import cPickle

from safe_leave_p_out import SafeLeavePLabelOut
from sklearn.metrics import f1_score
from sklearn.externals.joblib import Parallel, delayed
from utilities import adjectives

n_static_features = 188
n_dynamic_features = 16

class BestOfThem(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__ (self, static_classifier ,
                  dynamic_classifier,
                  chooser = "and"):
        super(BestOfThem, self).__init__()
        
        self.static_classifier = static_classifier
        self.dynamic_classifier = dynamic_classifier
        self.chooser = chooser
        
        assert chooser in ("and", "or", "static", "dynamic")
        assert(static_classifier is not None)
        assert(dynamic_classifier is not None)
        
    def fit(self, X, y=None):
        pass
    
    def predict(self, X):
        static_X = X[:, :n_static_features]
        dynamic_X = X[:, n_static_features:]
        
        static_out = self.static_classifier.predict(static_X)
        dynamic_out = self.dynamic_classifier.predict(dynamic_X)
        
        if self.chooser == "and":
            return np.logical_and(static_out, dynamic_out)
        elif self.chooser == "or":
            return np.logical_or(static_out, dynamic_out)
        elif self.chooser == "static":
            return static_out
        elif self.chooser == "dynamic":
            return dynamic_out
            
        else:
            raise ValueError("Unkown parameter: %s" % self.and_or)

def train_best_of_them(path, adjective, all_features, njobs):
    
    dataset_file_name = "_".join(("trained", adjective))+".pkl"    
    newpath = os.path.join(path, "best_of_them", "trained_adjectives")
    path_name = os.path.join(newpath, dataset_file_name)        

    #startic classifier
    filename = os.path.join(path, "static_features", "trained_adjectives", 
                            "trained_%s.pkl" % adjective)
    static_classifier = cPickle.load(open(filename))['classifier']
    #dynamic classifier
    filename = os.path.join(path, "hmm_features", "trained_adjectives", 
                            "trained_%s.pkl" % adjective)
    dynamic_classifier = cPickle.load(open(filename))['classifier']
    
    train_X = all_features[adjective]['train']['X']
    train_Y = all_features[adjective]['train']['Y']
    train_ids = all_features[adjective]['train']['ids']

    print "Training adjective %s" % adjective
    
    #magic training happening here!!!
    leav_out = 5
    clf = BestOfThem(static_classifier, dynamic_classifier)  
    
    #cv = sklearn.cross_validation.LeavePLabelOut(train_ids, leav_out)
    cv = SafeLeavePLabelOut(train_ids, leav_out, 200, train_Y)
    parameters = {
        'static_classifier': [static_classifier],
        'dynamic_classifier': [dynamic_classifier], 
        'chooser': ("and", "or", "static", "dynamic"),
                }
    
    verbose = 1
    grid = sklearn.grid_search.GridSearchCV(clf, parameters,
                                            n_jobs=njobs,
                                            cv=cv, 
                                            score_func=f1_score,
                                            verbose=verbose)
    grid.fit(train_X, train_Y)
    trained_clf = grid.best_estimator_
    #end of magic training!!!
    
    dataset = all_features[adjective]
    dataset['adjective'] = adjective
    dataset['classifier'] = trained_clf
    dataset['scaler'] = False
   
    # Save the results in the folder
    with open(path_name, "w") as f:
        print "Saving file: ", path_name
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
        
    test_X = all_features[adjective]['test']['X']
    test_Y = all_features[adjective]['test']['Y']
    
    train_score =  grid.best_score_
    test_score = f1_score(test_Y, trained_clf.predict(test_X))
    print "The training score is: %.2f, test score is %.2f" % (train_score, test_score)
    #print "Params are: ", grid.best_params_
    
def main():
    if len(sys.argv) == 4:
        path, adjective, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training only the adjective %s" % (adjective)
        
        dict_file = os.path.join(path, "combined_features", "combined_dict.pkl")        
        all_features =  cPickle.load(open(dict_file))
        train_best_of_them(path, adjective, all_features, n_jobs)

    elif len(sys.argv) == 3:
        path, n_jobs = sys.argv[1:]
        n_jobs = int(n_jobs)
        print "Training the all adjectives"
        
        dict_file = os.path.join(path, "combined_features", "combined_dict.pkl")        
        all_features =  cPickle.load(open(dict_file))
        
        #sleep tight while random race conditions happen
        failed_adjectives = adjectives[:]        
        while len(failed_adjectives) > 0:
            adjective = failed_adjectives.pop()
            try:
                train_best_of_them(path, adjective, all_features, n_jobs)
            except ValueError:
                print "adjective %s has problems, retrying..."
                failed_adjectives.append(adjective)
        
        
        #p = Parallel(n_jobs=n_jobs,verbose=10)
        #p(delayed(alt_train_adjective_phase_classifier)(path, adjective, all_features, 1) 
            #for adjective in adjectives)
                                                      
    else:
        print "Usage:"
        print "%s path adjective n_jobs" % sys.argv[0]
        print "%s path n_jobs" % sys.argv[0]
        print "Path to the base directory"

if __name__=="__main__":
    main()
    print "done"        