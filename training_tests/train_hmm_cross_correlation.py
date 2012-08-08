#! /usr/bin/python

import sklearn
import sklearn.decomposition
import sklearn.hmm
import cPickle
import tables
import sklearn.grid_search
import sklearn.cross_validation
import sklearn.cluster

from pylab import *
import numpy as np

#change a few functions otherwise it won't work with the grid search
def hmm_fit(self, X, y=None, **kwargs):    
    self.n_symbols = kwargs.pop("n_symbols", 15)
    self.transmat_ = None
    self.startprob_ = None
    
    try:
        return self.__orig_fit(X,**kwargs)
    except ValueError, e:
        print "Somenthing bad happened here, fit might not have worked. Message is: ", e
        print "transmat: ", self.transmat_
        print "startprob: ", self.startprob_
#        print "Training set: ", str(X)
        print "HMM id: ", id(self)            
#        print "Startprob prior: ", self.startprob_prior
#        print "Stats: ", self._initialize_sufficient_statistics()
        print "Restarting..."
        raise
        #return self.fit(X, y, **kwargs)
        
        
        

def hmm_score(self, X, y=None, **kwargs):                   
    score = mean([self.__orig_score(x,**kwargs) for x in X])            
    if isnan(score):
        score = NINF
    return score

sklearn.hmm.MultinomialHMM.__orig_fit = sklearn.hmm.MultinomialHMM.fit
sklearn.hmm.MultinomialHMM.fit = hmm_fit
sklearn.hmm.MultinomialHMM.__orig_score = sklearn.hmm.MultinomialHMM.score
sklearn.hmm.MultinomialHMM.score = hmm_score

def train_single_hmm(training_data, n_symbols, n_jobs=1):
    #creating the grid search
    hmm = sklearn.hmm.MultinomialHMM()
    print "NEW HMM, id: ", id(hmm)

    parameters = {"n_components" : range(5,25), 
                  "n_iter" : [100],
                  }
    
    cross_validator = sklearn.cross_validation.ShuffleSplit(len(training_data), n_iterations=3, train_size=0.9)
    grid = sklearn.grid_search.GridSearchCV(hmm, parameters, cv=cross_validator, fit_params={"n_symbols":n_symbols}, verbose=10, n_jobs=n_jobs)
    grid.fit(training_data)
    return grid.best_estimator_

def train_hmms(filenames, pca, kmeans, n_jobs=1):
    hmms = {}
    for filename in filenames:
        #getting the data out of each file
        db = tables.openFile(filename)
        trajectories = [ _g for _g in db.walkGroups("/") if _g._v_depth == 1]
        fingers_0 = [g.finger_0.electrodes.read() for g in trajectories]
        fingers_1 = [g.finger_1.electrodes.read() for g in trajectories]
        all_fingers = [ hstack((f0, f1))for (f0,f1) in zip(fingers_0, fingers_1)]
        
        training_data = [kmeans.predict(pca.transform(x)) for x in all_fingers]
        print "Training HMM for ", filename
        hmm = train_single_hmm(training_data, kmeans.n_clusters, n_jobs)
        db.close()
        hmm_name = filename.partition(".")[0]
        hmms[hmm_name] = hmm

    return hmms

def test_hmms(hmms, pca, kmeans, trajectory):
    input_traj = kmeans.predict(pca.transform(trajectory))
    scores = {}
    for name, hmm in hmms.iteritems():
        scores[name] = hmm.score( [input_traj])
    return scores

def test_dataset(hmms, pca, kmeans, db):
    trajectories = [ _g for _g in db.walkGroups("/") if _g._v_depth == 1]
    fingers_0 = [g.finger_0.electrodes.read() for g in trajectories]
    fingers_1 = [g.finger_1.electrodes.read() for g in trajectories]
    all_fingers = [ hstack((f0, f1))for (f0,f1) in zip(fingers_0, fingers_1)]
    
    ret_scores = {}
    for test_trajectory, group in zip(all_fingers, trajectories):
        scores = test_hmms(hmms, pca, kmeans, test_trajectory)
        
        #nan always mess up, let's remove them
        def key(x):
            v = scores[x]
            if isnan(v):
                return np.NINF
            else:
                return v       
            
        best = max(scores, key = key)
        ret_scores[group._v_name] = best      
        print "Best for ", group._v_name, " is ", best
    return ret_scores

def main():
    np.seterr(all="raise")
    pca = cPickle.load(open("pca.pkl", "r"))
    kmeans = cPickle.load(open("kmeans.pkl", "r"))
    
    filenames = ("bouncy_foam.h5", "cork.h5", "glass_bottle.h5", "hard_rough_foam.h5", "metal_bar.h5", "soft_foam.h5")    
    #filenames = ("soft_foam.h5",)

    hmms = train_hmms(filenames, pca, kmeans, n_jobs = 6)

    cPickle.dump(hmms, open("hmms_cross.pkl","w"), cPickle.HIGHEST_PROTOCOL)
    db = tables.openFile("all_data.h5")
    scores = test_dataset(hmms, pca, kmeans, db)
    num_hits = 0
    for test_set, class_result in scores.iteritems() :
        if test_set[:len(class_result)] == class_result:
            num_hits += 1
    print "Num of correct classifications: ", num_hits, "over ", len(scores)
    print "Percentage: ", float(num_hits) / len(scores)


if __name__ == "__main__":
    main()