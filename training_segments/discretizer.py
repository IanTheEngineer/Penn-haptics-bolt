from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import numpy as n

from utilities import congrid

    
class Resample(BaseEstimator, TransformerMixin):
    """
    Resample an input array. Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Available methods:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates
    
    """
    def __init__(self, newshape = None, method='linear', centre=True, minusone=False,
                 original_rows = None):
        """
        
        newdims: the new dimension for resampling. It has to be a scalar and it represents the new number of rows
        for an input matrix.
        
        method: 
        linear, neighbour, nearest, linear, spline
        
        centre:
        True - interpolation points are at the centres of the bins
        False - points are at the front edge of the bin

        minusone:
        For example- inarray.shape = (i,j) & new dimensions = (x,y)
        False - inarray is resampled by factors of (i/x) * (j/y)
        True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
        This prevents extrapolation one element beyond bounds of input array.        
        
        original_rows:
        to apply the inverse transform one has to supply original_rows, i.e. the number of rows the data is supposed to
        have before discrtization. This can be done with set_params prior to calling inverse_trasform.
        """
        super(Resample).__init__(self)
        
        self.newshape = newshape
        self.method = method
        self.centre = centre
        self.minusone = minusone
        self.original_rows = original_rows
        
        
    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        return self

    def transform(self, X, y=None):
        """
        Resample the matrix X.
        
        X is of shape [n_samples, n_features]. It has to have at least 2 dimensions.
        """
       
        newshape = (self.newshape, X.shape[1])
        return congrid(X, newshape, self.method, self.centre, self.minusone)
    
    def inverse_transform(self, X):
        """
        Inverse transform of sampled data X. It requires original_shape to be set (via set_params).
        """
        if self.original_rows is None:
            raise ValueError("original shape is not set!")
        
    
        newshape = (self.original_rows, X.shape[1])
        return congrid(X, newshape, self.method, self.centre, self.minusone)        

class KMeansDiscretizer(KMeans):
    """
    See KMeans for doc and parameters.
    """
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                     tol=1e-4, precompute_distances=True,
                     verbose=0, random_state=None, copy_x=True, n_jobs=1, k=None):
        
        super(KMeansDiscretizer, self).__init__(n_clusters, init, n_init, max_iter,
                                          tol, precompute_distances,
                                          verbose, random_state, copy_x,
                                          n_jobs, k)
    def transform(self, X, y=None):
        """
        This replaces KMeans.transform by calling predict.
        """
        return self.predict(X)
    
    def inverse_transform(self, labels):
        """
        Given an array of integer labels, it returns a matrix where each
        row is the centroid corresponding to the label. This is the reconstruction
        of the data from the labels.
        """
        discretized_data = [self.cluster_centers_[l] for l in labels]
        return n.vstack(discretized_data)
    
    
class DummyItem(BaseEstimator, TransformerMixin):
    """
    Dummy item to fool a pipeline to have a classifier at the end, just to run
    inverse_transform over the entire pipeline.
    """
    def transform(self, X, y=None):
        return X
    def fit(self, X, y=None):
        return self
    def inverse_transform(self, X):
        return X