#! /usr/bin/python
"""
Create a preprocessing chain by piecing together PCA, Discretization and KMeans.
"""

import sklearn
import sklearn.hmm
import sklearn.pipeline
from hmm_classifier import MultinomialHMMClasifier, DataSplitter, DataCombiner

from sklearn.base import BaseEstimator, TransformerMixin

from discretizer import Resample, KMeansDiscretizer
from sklearn.decomposition import PCA
import utilities
import os
import cPickle
import numpy as np

class HMMChain(BaseEstimator, TransformerMixin):
    """This chain contains the steps for preprocessing and probability estimation
    of the adjectives. It behaves like an Estimator and Transformer. The parameters
    are passed to the constructor.
    """
    
    def __init__(self,
                 n_pca_components = 1,
                 n_hidden_components = 1,
                 resampling_size = 100,
                 n_discretization_symbols = 8,
                 hmm_max_iter = 100,
                 resampling_method = "linear",
                 kmeans_n_init = 100,
                 kmeans_n_jobs= 1,
                 kmeans_max_iter = 300,
                 data_splits = None,
                 my_class = None,
                 other_classes = None,
                 ):
        
        super(HMMChain, self).__init__()
        self.pca = PCA(n_pca_components)
        
        
        self.resample = Resample(resampling_size,
                                 resampling_method,
                                 )                                 
        
        self.discretizer = KMeansDiscretizer(n_clusters = n_discretization_symbols,
                                             n_init = kmeans_n_init,
                                             n_jobs = kmeans_n_jobs,
                                             max_iter = kmeans_max_iter,
                                            )
        self.hmm = MultinomialHMMClasifier(n_discretization_symbols,
                                           n_hidden_components,
                                           n_iter = hmm_max_iter,                                           
                                           )
        
        self.splitter = DataSplitter(data_splits)
        self.splitter2 = DataSplitter()
        
        
        self.combiner = DataCombiner()
        
        seq = [('Combine_1', self.combiner),
               ('PCA', self.pca),
               ('Splitter_1', self.splitter),
               ('Resample', self.resample),
               ('Combine_2', self.combiner),
               ('Discretizer', self.discretizer),
               ('Splitter_2', self.splitter2),
               ('HMM', self.hmm)
              ]
        self.pipeline = sklearn.pipeline.Pipeline(seq)
        self.my_class = my_class
        self.other_classes = other_classes
    
        
    def update_splits(self, X):
        neworig_splits = [len(x) for x in X]
        self.splitter.splits = neworig_splits
        
        new_discr_splits = [self.resampling_size for x in X]
        self.splitter2.splits = new_discr_splits

    def __load_adjective(self, path, adjective, 
                         phase, sensor):
        
        filename = os.path.join(path, adjective + ".pkl")
        with open(filename) as f:
            data = cPickle.load(f)
            
            return data[phase][sensor]        
        
    
    def perform_comparative_score(self, X,
                                  ):
        #print "Doing comparative"
        myclass = self.my_class
        other_classes = self.other_classes
        isinstance(other_classes, dict)
        total = 0.0
        this_score = self.pipeline.score(X)

        if type(self.other_classes) is tuple:
            path, phase, sensor = self.other_classes
            for adjective in utilities.adjectives:
                if adjective != myclass:
                    data = self.__load_adjective(path,
                                                 adjective,
                                                 phase,
                                                 sensor)
                    if type(data) is not list:
                        data = [data]                                    
                    self.update_splits(data)                    
                    score = self.pipeline.score(data)
                    total += score
        else:
            for c, data in other_classes.iteritems():
                if c != myclass:
                    if type(data) is not list:
                        data = [data]                    
                    self.update_splits(data)                    
                    score = self.pipeline.score(data)
                    total += score
                
        return score - total
        

    def __fix_input(self, X):
        if type(X) is np.ndarray:
            if X.dtype is np.dtype(object):
                return X.tolist()
            else:
                return [X]
        elif type(X) is list:
            return X
        else:
            return [X]
    
    def score(self, X, y=None):
        X = self.__fix_input(X)
        self.update_splits(X)
        if hasattr(self, "my_class") and self.my_class is not None and self.other_classes is not None:
            return self.perform_comparative_score(X)
        else:
            return self.pipeline.score(X, y)

    def transform(self, X):
        X = self.__fix_input(X)
        self.update_splits(X)        
        return self.pipeline.transform(X)

    def fit(self, X, y=None, **fit_params):
        X = self.__fix_input(X)
        self.update_splits(X)        
        self.pipeline.fit(X, y, **fit_params)
        return self
    
    @property    
    def n_pca_components(self):
        return self.pca.n_components
    @n_pca_components.setter
    def n_pca_components(self, value):
        self.pca.n_components = value
        
    @property
    def resampling_size(self):
        return self.resample.newshape
    @resampling_size.setter
    def resampling_size(self, value):
        self.resample.newshape = value
    
    @property
    def resampling_method(self):
        return self.resample.method
    @resampling_method.setter
    def resampling_method(self, value):
        self.resample.method = value
    
    @property
    def n_discretization_symbols(self):
        return self.discretizer.n_clusters
    @n_discretization_symbols.setter
    def n_discretization_symbols(self, value):
        self.discretizer.n_clusters = value    
        self.hmm.n_symbols = value
    
    @property
    def kmeans_n_init(self):
        return self.discretizer.n_init
    @kmeans_n_init.setter
    def kmeans_n_init(self, value):
        self.discretizer.n_init = value        
    
    @property
    def kmeans_n_jobs(self):
        return self.discretizer.n_jobs
    @kmeans_n_jobs.setter
    def kmeans_n_jobs(self, value):
        self.discretizer.n_jobs = value            

    @property
    def kmeans_max_iter(self):
        return self.discretizer.max_iter
    @kmeans_max_iter.setter
    def kmeans_max_iter(self, value):
        self.discretizer.max_iter = value    
    
    @property
    def n_hidden_components(self):
        return self.hmm.n_components
    @n_hidden_components.setter
    def n_hidden_components(self, value):
        self.hmm.n_components = value    
        
    @property
    def hmm_max_iter(self):
        return self.hmm.n_iter
    @hmm_max_iter.setter
    def hmm_max_iter(self, value):
        self.hmm.n_iter = value         
        
    @property
    def data_splits(self):
        return self.splitter.splits
    @data_splits.setter
    def data_splits(self, value):
        self.splitter.splits = value       
