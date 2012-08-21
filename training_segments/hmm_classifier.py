import sklearn.hmm
from sklearn.base import BaseEstimator
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
                                                      transmat_prior,
                                                      algorithm,
                                                      random_state,
                                                      n_iter, thresh, params,
                                                      init_params)
        self.n_symbols = n_symbols
        
    def fit(self, X, y=None, **kwargs):    
        if "n_symbols" in kwargs:
            self.n_symbols = kwargs["n_symbols"]
        
        self.transmat_ = None
        self.startprob_ = None        
        
        newX = [x.ravel() for x in X]
        try:
            return super(MultinomialHMMClasifier, self).fit(newX,**kwargs)
        except ValueError, e:
            print "Somenthing bad happened here, fit might not have worked. "\
                  "Message is: ", e
            raise
    
    def score(self, X, y=None, **kwargs):   
        newX = [x.ravel() for x in X]
        score = np.mean([super(MultinomialHMMClasifier, self).score(x,**kwargs)
                         for x in newX])            
        if np.isnan(score):
            score = np.NINF
        return score
    
    def transform(self, X, y=None, **kwargs):
        """ I know this shouldn't be here, but I need it"""
        return X
    

class DataSplitter(BaseEstimator):
    """Class used in conjunction with the HMMClassifier. Most of the 
    classifiers treat input data as organized in a matrix N x dim where N is 
    the number of input vectors and dim is the dimensionality (number of 
    features) for each vector. An HMM wants data to be organized in batches, so
    a single input matrix has to be split into chunks. This class sits therefore
    before the HMM, splitting a monolitic (single matrix) data into smaller 
    matrices.
    
    Parameters:
    split: a list defining the length of each split. Example of split:
    >> A = repeat([range(10)], 3, axis=0).T
    array([[0, 0, 0],
       [1, 1, 1],
       [2, 2, 2],
       [3, 3, 3],
       [4, 4, 4],
       [5, 5, 5],
       [6, 6, 6],
       [7, 7, 7],
       [8, 8, 8],
       [9, 9, 9]])
    >> split(a, [[3,5,2])
    [array([[0, 0, 0],
       [1, 1, 1],
       [2, 2, 2]]),
     array([[3, 3, 3],
       [4, 4, 4],
       [5, 5, 5],
       [6, 6, 6],
       [7, 7, 7]]),
    array([[8, 8, 8],
       [9, 9, 9]])]
    """
    
    
    def __init__(self, splits=None):
        super(DataSplitter,self).__init__()
        self.splits = splits

    def transform(self, X, y=None, **kwargs):
        """Divides the data according to splits."""
        
        ret = []
        acc = 0
        for s in self.splits:
            ret.append(X[acc:acc+s, :])
            acc += s
        return ret
    
    def fit(self, X, y=None, **kwargs):
        return self

class DataCombiner(BaseEstimator):
    """This class performs the opposite of DataSplitter.
    Basically a vstack on the input
    """
    
    def __init__(self):
        super(DataCombiner,self).__init__()

    def transform(self, X, y=None, **kwargs):
        """Stacks the data"""
        return np.vstack(X)
    
    def fit(self, X, y=None, **kwargs):
        """Really does nothing here"""
        return self

#sklearn.hmm.MultinomialHMM.__orig_fit = sklearn.hmm.MultinomialHMM.fit
#sklearn.hmm.MultinomialHMM.fit = hmm_fit
#sklearn.hmm.MultinomialHMM.__orig_score = sklearn.hmm.MultinomialHMM.score
#sklearn.hmm.MultinomialHMM.score = hmm_score
