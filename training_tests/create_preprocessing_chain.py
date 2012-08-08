#! /usr/bin/python
"""
Create a preprocessing chain by piecing together PCA, Discretization and KMeans.
"""

import sklearn
import sklearn.decomposition
import sklearn.hmm
import cPickle
import tables
import sklearn.grid_search
import sklearn.cross_validation
import sklearn.cluster
import sklearn.pipeline

import numpy as np

from discretizer import Resample, DummyItem, KMeansDiscretizer

def create_chain(resampling_size = 100, n_clusters=15):
    """
    Create a sklearn.pipeline.Pipeline chain with PCA, Discretization and KMeans. The last
    element of the chain is DummyItem to allow chain.inverse_transform to use all the memebrs
    of the chain.
    
    Parameters:
    resampling_size: the number of elements to use in the discretization.
    n_clusters: the number of clusters for KMeansDiscretizer.
    """
    
    np.seterr(all="raise")
    pca = cPickle.load(open("pca.pkl", "r"))
    
    all_data = tables.openFile("all_data.h5")
    trajectories = [ _g for _g in all_data.walkGroups("/") if _g._v_depth == 1]
    fingers_0 = [g.finger_0.electrodes.read() for g in trajectories]
    fingers_1 = [g.finger_1.electrodes.read() for g in trajectories]
    all_fingers = [ np.hstack((f0, f1))for (f0,f1) in zip(fingers_0, fingers_1)]
    
    resampler = Resample()
    resampler.centre = False
    resampler.newshape = resampling_size
    pca_data = [resampler.transform(pca.transform(x)) for x in all_fingers]
    
    discretizer = KMeansDiscretizer(n_clusters=n_clusters, 
                                    max_iter=1000, 
                                    n_jobs=-1, 
                                    n_init=20)
    discretizer.fit(np.vstack(pca_data))
    
    seq = seq = [("PCA" , pca), 
                 ("Resample", resampler), 
                 ("Discretizer", discretizer), 
                 ("dummy", DummyItem())]
    chain = sklearn.pipeline.Pipeline(seq)
    return chain