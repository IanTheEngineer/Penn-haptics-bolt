#! /usr/bin/python
"""
This script trains a set of HMMs on a set of files. It uses the preprocessing chain created with
create_preprocessing_chain.py. Cross validation is used to select the number of hidden states.
"""

import sklearn
import sklearn.decomposition
import sklearn.hmm
import cPickle
import tables
import sklearn.grid_search
import sklearn.cross_validation
import sklearn.cluster

import numpy as np

#=================================HACKY HACKY=======================================================#
#change a few functions otherwise it won't work with the grid search
def hmm_fit(self, X, y=None, **kwargs):    
    self.n_symbols = kwargs.pop("n_symbols", 15)
    self.transmat_ = None
    self.startprob_ = None
    
    try:
        return self.__orig_fit(X,**kwargs)
    except ValueError, e:
        print "Somenthing bad happened here, fit might not have worked. Message is: ", e
        raise

def hmm_score(self, X, y=None, **kwargs):                   
    score = np.mean([self.__orig_score(x,**kwargs) for x in X])            
    if np.isnan(score):
        score = np.NINF
    return score

sklearn.hmm.MultinomialHMM.__orig_fit = sklearn.hmm.MultinomialHMM.fit
sklearn.hmm.MultinomialHMM.fit = hmm_fit
sklearn.hmm.MultinomialHMM.__orig_score = sklearn.hmm.MultinomialHMM.score
sklearn.hmm.MultinomialHMM.score = hmm_score
#===================================================================================================#

def train_single_hmm(training_data, n_symbols, n_jobs=1):
    """
    Trains and returns single hmm over training_data. Uses cross-validation.
    
    Parameters:
    training_data: a list of sequences of data. That is a list of matrices (see MultinomialHMM.fit).
    n_symbols: the number of symbols in the output state.
    n_jobs: how many jobs to run in parallel. Set to -1 to use all the CPUs.    
    """
    
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

def train_hmms(filenames, chain, n_jobs=1):
    """
    Train several hmms, one for each class as listed by filenames.
    
    Parameters:
    filenames: a list of hd5 files, one file for each class.
    chain: the pre-processing chain created with create_preprocessing_chain.py
    n_jobs: how many jobs to run in parallel. Set to -1 to use all the CPUs.
    """
    
    hmms = {}
    for filename in filenames:
        #getting the data out of each file
        db = tables.openFile(filename)
        trajectories = [ _g for _g in db.walkGroups("/") if _g._v_depth == 1]
        fingers_0 = [g.finger_0.electrodes.read() for g in trajectories]
        fingers_1 = [g.finger_1.electrodes.read() for g in trajectories]
        all_fingers = [ np.hstack((f0, f1))for (f0,f1) in zip(fingers_0, fingers_1)]
        
        training_data = [chain.transform(x) for x in all_fingers]
        print "Training HMM for ", filename
        n_clusters = chain.get_params()['Discretizer__n_clusters']
        hmm = train_single_hmm(training_data, n_clusters, n_jobs)
        db.close()
        hmm_name = filename.partition(".")[0]
        hmms[hmm_name] = hmm

    return hmms

def test_hmms(hmms, chain, trajectory):
    """Returns the scores for a set of hmms over a single trajectory.
    
    Parameters:
    hmms: a dictionary where the keys are the names for the hmms the the values are MultinomialHMM instances.
    chain: the pre-processing chain created with create_preprocessing_chain.py
    trajectory: a [n_points, n_features] matrix to be preprocessed by chain.
    """
    
    input_traj = chain.transform(trajectory)
    scores = {}
    for name, hmm in hmms.iteritems():
        scores[name] = hmm.score( [input_traj])
    return scores

def test_dataset(hmms, chain, db):
    """Returns the scores for a set of hmms over an entire dataset
    
    Parameters:
    hmms: a dictionary where the keys are the names for the hmms the the values are MultinomialHMM instances.
    chain: the pre-processing chain created with create_preprocessing_chain.py
    db: a pytables database, created with pytables.open
    """
    
    trajectories = [ _g for _g in db.walkGroups("/") if _g._v_depth == 1]
    fingers_0 = [g.finger_0.electrodes.read() for g in trajectories]
    fingers_1 = [g.finger_1.electrodes.read() for g in trajectories]
    all_fingers = [ np.hstack((f0, f1))for (f0,f1) in zip(fingers_0, fingers_1)]
    
    ret_scores = {}
    for test_trajectory, group in zip(all_fingers, trajectories):
        scores = test_hmms(hmms, chain, test_trajectory)
        
        #nan always mess up, let's remove them
        def key(x):
            v = scores[x]
            if np.isnan(v):
                return np.NINF
            else:
                return v       
            
        best = max(scores, key = key)
        ret_scores[group._v_name] = best      
        print "Best for ", group._v_name, " is ", best
    return ret_scores

def main():
    np.seterr(all="raise")
    chain = cPickle.load(open("pre-processing_chain.pkl", "r"))
    
    filenames = ("bouncy_foam.h5", "cork.h5", "glass_bottle.h5", "hard_rough_foam.h5", "metal_bar.h5", "soft_foam.h5")    
    #filenames = ("soft_foam.h5",)

    hmms = train_hmms(filenames, chain, n_jobs = 6)

    cPickle.dump(hmms, open("hmms_subsampled.pkl","w"), cPickle.HIGHEST_PROTOCOL)
    db = tables.openFile("all_data.h5")
    scores = test_dataset(hmms, chain, db)
    num_hits = 0
    for test_set, class_result in scores.iteritems() :
        if test_set[:len(class_result)] == class_result:
            num_hits += 1
    print "Num of correct classifications: ", num_hits, "over ", len(scores)
    print "Percentage: ", float(num_hits) / len(scores)


if __name__ == "__main__":
    main()