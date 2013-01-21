#! /usr/bin/python
"""
Create a preprocessing chain by piecing together PCA and Resampling.
"""

import sklearn
import sklearn.hmm
import sklearn.pipeline
from hmm_classifier import DataSplitter, DataCombiner

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.svm import SVC

from discretizer import Resample
from sklearn.decomposition import PCA
import utilities
import os
import cPickle
import numpy as np
from copy import deepcopy
from collections import defaultdict
import tables

from TGA_python_wrapper import tga_dissimilarity

class TGAClassifier(SVC):
    def __init__(self, C = 1.0, 
                 sigma_multiplier = 1.0, 
                 T_multiplier = 1.0,
                 class_weight = None
                 ):
        super(TGAClassifier, self).__init__(kernel="precomputed", C=C,
                                            class_weight=class_weight,
                                            cache_size = 500)
        self.sigma_multiplier = sigma_multiplier
        self.T_multiplier = T_multiplier
        self.training_vectors = None
        self._T = None
        self._sigma = None
        self.gram_matrix = None

    def approximate_T(self, X):
        """The Triangular parameter T can be set to a reasonable multiple of the 
        median length, e.g 0.2 or 0.5. Note that whenever two time-series 
        length differ by more than T-1, their kernel value is equal to 0.
        """
        X = self.__fix_input(X)
        T = np.median([len(i) for i in X])
        return T

    def approximate_sigma(self, X, T = None):
        """
        The Bandwidth sigma can be set as a multiple of a simple estimate of the 
        median distance of different points observed in different time-series of 
        your training set, scaled by the square root of the median length of 
        time-series in the training set. 
        """
        X = self.__fix_input(X)
        
        if T is None:
            T = self.approximate_T(X)
        
        all_diffs = []
        for i in range(len(X)-1):
            for j in range(i+1, len(X)):
                all_diffs.append(X[i] - X[j])
        d = np.vstack(all_diffs)
        sigma = np.median(np.sqrt(np.sum(d**2,1))) * np.sqrt(T)
        return sigma        
    
    def _tga(self, x, y, sigma, T):
        x = np.array(x, copy=False, order='C')
        y = np.array(y, copy=False, order='C')
        d = tga_dissimilarity(x, y, sigma, T)
        return np.exp(-d)

    def __fix_input(self, X):
        if type(X) is np.ndarray:
            if X.dtype is np.dtype(object):
                return X.tolist()
            else:
                return [X]
        else:
            return X    
    
    def _gram(self, X, sigma, T):
        X = self.__fix_input(X)
        
        gram = np.empty((len(X), len(X)))
        for i in range(len(X)):
            for j in range(i, len(X)):
                gram[i,j] = self._tga(X[i], X[j], sigma, T)        
                gram[j,i] = gram[i,j]
        return gram
    
    def fit(self,  X, y, sample_weight=None):
        X = self.__fix_input(X)
        self._T = self.approximate_T(X) * self.T_multiplier
        self._sigma = self.approximate_sigma(X, self._T) * self.sigma_multiplier
        
        self.training_vectors = deepcopy(X)               
        self.gram_matrix = self._gram(X, self._sigma, self._T)
        return super(TGAClassifier, self).fit(self.gram_matrix, y, sample_weight)
    
    def predict(self, X):
        X = self.__fix_input(X)
        
        input_vect = np.zeros((len(X), 
                              len(self.training_vectors)))
        
        #populate the new gram matrix
        for X_i in range(len(X)):
            #consider only the support vectors
            for s in self.support_:
                input_vect[X_i, s] = self._tga(self.training_vectors[s],
                                               X[X_i],
                                               self._sigma,
                                               self._T)
        
        return super(TGAClassifier, self).predict(input_vect)

class TGAChain(BaseEstimator, TransformerMixin):
    """This chain contains the steps for preprocessing before using the TGA
    kernels.
    """
    
    def __init__(self,
                 n_pca_components = 0.95,
                 resampling_size = 100,
                 resampling_method = "linear",
                 whiten = False,
                 C = 1.0,
                 T_multiplier = 1.0,
                 sigma_multiplier = 1.0,
                 class_weight = None
                 ):
        
        super(TGAChain, self).__init__()
        self.pca = PCA(n_pca_components, whiten=whiten)
        
        
        self.resample = Resample(resampling_size,
                                 resampling_method,
                                 )                                 
        
        self.splitter = DataSplitter()
        self.combiner = DataCombiner()
        
        self.clf = TGAClassifier(C, sigma_multiplier, T_multiplier,
                                 class_weight)
        
        seq = [('Combine_1', self.combiner),
               ('PCA', self.pca),
               ('Splitter_1', self.splitter),
               ('Resample', self.resample),
               ('Classifier', self.clf),
              ]
        self.pipeline = sklearn.pipeline.Pipeline(seq)
        
    def update_splits(self, X):
        neworig_splits = [len(x) for x in X]
        self.splitter.splits = neworig_splits

    def __load_adjective(self, path, adjective, 
                         phase, sensor):
        
        filename = os.path.join(path, adjective + ".pkl")
        with open(filename) as f:
            data = cPickle.load(f)
            
            return data[phase][sensor]        
        
    
    def __fix_input(self, X):
        if type(X) is np.ndarray:
            if X.dtype is np.dtype(object):
                return X.tolist()
            else:
                return [X]
        else:
            return X

    def score(self, X, y=None):
        X = self.__fix_input(X)
        self.update_splits(X)
        return self.pipeline.score(X, y)

    def transform(self, X):
        X = self.__fix_input(X)
        self.update_splits(X)        
        return self.pipeline.transform(X)

    def fit(self, X, y=None, **fit_params):
        
        X = self.__fix_input(X)
        
        #find the right parameters        
        self.update_splits(X)        
        self.pipeline.fit(X, y, **fit_params)
        return self
    
    def predict(self, X):
        X = self.__fix_input(X)
        self.update_splits(X)
        return self.pipeline.predict(X)
    
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
    def data_splits(self):
        return self.splitter.splits
    @data_splits.setter
    def data_splits(self, value):
        self.splitter.splits = value       

    @property
    def whiten(self):
        return self.pca.whiten
    @whiten.setter
    def whiten(self, value):
        self.pca.whiten = value      
        
    @property
    def sigma_multiplier(self):
        return self.clf.sigma_multiplier
    @sigma_multiplier.setter
    def sigma_multiplier(self, value):
        self.clf.sigma_multiplier = value        

    @property
    def T_multiplier(self):
        return self.clf.T_multiplier
    @T_multiplier.setter
    def T_multiplier(self, value):
        self.clf.T_multiplier = value      \
            
    @property
    def C(self):
        return self.clf.C
    @C.setter
    def C(self, value):
        self.clf.C = value          
        
    @property
    def class_weight(self):
        return self.clf.class_weight
    @class_weight.setter
    def class_weight(self, value):
        self.clf.class_weight = value      
        

class TGAEnsemble(ClassifierMixin):
    def __init__(self, adjective, base_directory = None):
        super(TGAEnsemble, self).__init__()
        
        self.chains = defaultdict(dict)
        self.adjective = adjective
        if base_directory is not None:
            self.load_directory(base_directory)
            
    def load_directory(self, base_directory):
        num_total = 0
        for f in os.listdir(base_directory):            
            if not f.endswith('.pkl'):
                continue
            if not self.adjective in f.split("_"):
                continue
            path_name = os.path.join(base_directory, f)
            with open(path_name, "r") as file_path:
                tga = cPickle.load(file_path)
            chars = f.strip(".pkl").split("_")
            chars = chars[1:] #chain
            chars = chars[1:] #adjective
            sensor = chars.pop()
            phase = "_".join(chars) #silly me for the choice of separator!
            self.chains[phase][sensor] = tga
            num_total += 1
        if num_total != 16:
            raise ValueError("Only %d classifiers found, expected %d. The dict\
            is \n%s" %(num_total, 16, 
                     "\n".join([str((k1, k2)) 
                                for (k1,v) in self.chains.iteritems()
                                for k2 in v.iterkeys()])
                     ))
    
    def classification_labels(self, X):
        """
        X: list of dictionaries d, each with the structure:
            d[phase][sensor] = data
        """
        if isinstance(X, tables.Group):
            X = utilities.dict_from_h5_group(X)['data']
        
        if type(X) is not list:
            X = [X]
        ret = []
        for x in X:        
            scores = []
            for phase, v in x.iteritems():                
                for sensor, data in v.iteritems():
                    try:
                        chain = self.chains[phase][sensor]                            
                        scores.append(chain.predict(data)[0])
                    except KeyError:
                        print "No key for %s %s" %(phase, sensor)
            ret.append(scores)
        return ret            