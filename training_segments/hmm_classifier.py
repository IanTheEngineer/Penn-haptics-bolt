import sklearn.hmm
import numpy as np
import string

#change a few functions otherwise it won't work with the grid search
class MultinomialHMMClasifier(sklearn.hmm.MultinomialHMM):
    def __init__(self, n_symbols = 10, n_components=1, startprob=None, 
                 transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, thresh=1e-2, params=string.ascii_letters,
                 init_params=string.ascii_letters):
        
        super(MultinomialHMMClasifier, self).__init__(n_components, startprob,
                                                      transmat, startprob_prior,
                                                      algorithm,
                                                      random_state,
                                                      n_iter, thresh, params,
                                                      init_params)
        self.n_symbols = n_symbols
        
    def fit(self, X, y=None, **kwargs):    
        self.n_symbols = kwargs.pop("n_symbols", 15)
        self.transmat_ = None
        self.startprob_ = None
        
        try:
            return super(MultinomialHMMClasifier, self).fit(X,**kwargs)
        except ValueError, e:
            print "Somenthing bad happened here, fit might not have worked. "\
                  "Message is: ", e
            raise
    
    def score(self, X, y=None, **kwargs):                   
        score = np.mean([super(MultinomialHMMClasifier, self).score(x,**kwargs)
                         for x in X])            
        if np.isnan(score):
            score = np.NINF
        return score

#sklearn.hmm.MultinomialHMM.__orig_fit = sklearn.hmm.MultinomialHMM.fit
#sklearn.hmm.MultinomialHMM.fit = hmm_fit
#sklearn.hmm.MultinomialHMM.__orig_score = sklearn.hmm.MultinomialHMM.score
#sklearn.hmm.MultinomialHMM.score = hmm_score
