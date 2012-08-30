import cPickle
import os
from collections import defaultdict
import numpy as np
import tables

import utilities
from sklearn.base import ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import cross_validation


class AdjectiveClassifier(ClassifierMixin):
    def __init__(self, adjective, base_directory = None):
        super(AdjectiveClassifier, self).__init__()
        
        self.chains = defaultdict(dict)
        self.adjective = adjective
        if base_directory is not None:
            self.load_directory(base_directory)
        
        self.svc = None
        self.labels = None
        self.features = None
        
    def load_directory(self, base_directory):
        for f in os.listdir(base_directory):
            if not f.endswith('.pkl'):
                continue
            if not self.adjective in f:
                continue
            path_name = os.path.join(base_directory, f)
            with open(path_name, "r") as file_path:
                hmm = cPickle.load(file_path)
            chars = f.strip(".pkl").split("_")
            chars = chars[1:] #chain
            chars = chars[1:] #adjective
            sensor = chars.pop()
            phase = "_".join(chars) #silly me for the choice of separator!
            self.chains[phase][sensor] = hmm
    
    def extract_features(self, X):
        """
        X: list of dictionaries d, each with the structure:
            d[phase][sensor] = data
        """
        if isinstance(X, tables.Group):
            X = utilities.dict_from_h5_group(X,
                                             utilities.phases,
                                             utilities.sensors,
                                             )        
        
        if type(X) is not list:
            X = [X]
        ret = []
        for x in X:
            scores = []
            for phase, v in x.iteritems():
                for sensor, data in v.iteritems():
                    try:
                        chain = self.chains[phase][sensor]
                        scores.append(chain.score(data))
                    except KeyError:
                        pass
            ret.append(scores)
        return ret
            

    def create_features_set(self, database, store = False, verbose = False):
        """
        For each object in the database, run classifier.extract_features. All the
        features are then collected in a matrix.
        If the classifier's adjective is among the objects' then the feature
        is labeled with 1, otherwise 0. 

        Parameters:
        database: either a string or an open pytables file.
        
        Returns the features and the labels as two 2-dimensional matrices.
        """
        labels = []
        features = []
        for group in utilities.iterator_over_object_groups(database):
            data_dict = utilities.dict_from_h5_group(group,
                                                     utilities.phases,
                                                     utilities.sensors,
                                                     )
            if verbose:
                print "Loading object ", data_dict["name"]
            data = data_dict["data"]            
            features.append(self.extract_features(data))
            if self.adjective in data_dict["adjectives"]:
                labels.append(1)
            else:
                labels.append(0)
        
        features = np.array(features).squeeze()
        labels = np.array(labels).flatten()
        if store:
            self.features = features
            self.labels = labels
        return features, labels

    def predict(self, X):
        if isinstance(X, tables.Group):
            data_dict = utilities.dict_from_h5_group(X,
                                                     utilities.phases,
                                                     utilities.sensors,
                                                     )          
            features = self.extract_features(data_dict["data"])
        else:
            features = X
            
        return self.svc.predict(features)

    def test_on_database(self, database):
        score = 0.0
        tots = 0.0
        for g in utilities.iterator_over_object_groups(database):
            p = self.predict(g)
            label = (p[0] == 1)            
            in_adjective = (self.adjective in g.adjectives[:])
            if in_adjective == label:
                score += 1
            tots += 1
        return score / tots 
    
    def train_on_features(self, clf = None, parameters = None, cv = None):
        """Train a support vector machine classifier on the features and labels
        that have been produced using self.create_features_set.
        """        
        if not hasattr(self, "features"):
            raise ValueError("No features present, have you run create_features_set?")
        if not hasattr(self, "labels"):
            raise ValueError("No labels present, have you run create_features_set?")        

        if clf is None:
            self.svc = LinearSVC()
        else:
            self.svc = clf
        
        score_func = f1_score        
        
        if cv is None:
            cv = cross_validation.StratifiedShuffleSplit((self.labels), 
                                                         test_size=1/2.,
                                                         n_iterations=10)
        
        if parameters is None:
            parameters = {"dual":[False, False]}
            
        grid = GridSearchCV(self.svc, 
                            parameters, 
                            score_func=score_func, 
                            cv=cv, 
                            verbose=0, 
                            n_jobs=1
                            )
        grid.fit(self.features, self.labels)
        self.svc = grid.best_estimator_
        return self
    
    def train_on_separate_dataset(self, test_X, test_Y, 
                                  train_X = None, train_Y  = None,
                                  all_Cs =  None, 
                                  score_fun = None,
                                  verbose = True):
        if train_X is None and not hasattr(self, "features"):
            raise ValueError("No features present, have you run create_features_set?")
        if train_Y is None and not hasattr(self, "labels"):
            raise ValueError("No labels present, have you run create_features_set?")        
        
        if all_Cs is None:
            all_Cs = np.linspace(1, 1e6, 1000)

        if train_X is None:
            train_X = self.features
        if train_Y is None:
            train_Y = self.labels
       
        scores = []
        for C in all_Cs:
            clf = LinearSVC(C=C, dual=False)
            clf.fit(train_X, train_Y)
            if score_fun is None:
                score = clf.score(test_X, test_Y)
            else:
                pred_Y = clf.predict(test_X)
                score = score_fun(test_Y, pred_Y)
            scores.append(score)
        
        best_C = np.argmax(all_Cs)
        self.svc = LinearSVC(C = best_C, dual = False).fit(train_X, train_Y)
        if verbose:
            if score_fun is None:
                score_training = self.svc.score(train_X, train_Y)
                score_testing = self.svc.score(test_X, test_Y)
            else:
                pred_Y = self.svc.predict(train_X)
                score_training = score_fun(train_Y, pred_Y)
                
                pred_Y = self.svc.predict(test_X)
                score_testing = score_fun(test_Y, pred_Y)                
            
            print "Training score: %f, testing score: %f" %(score_training,
                                                            score_testing)
        
        return self
        