import sklearn.hmm
import numpy as np

#change a few functions otherwise it won't work with the grid search
class MultinomialHMMClasifier(sklearn.hmm.MultinomialHMM):
    def __init__(self, n_symbols = 10, *args, **kwargs):
        super(MultinomialHMMClasifier, self).__init__(*args, **kwargs)
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
