import cPickle
import os
from collections import defaultdict
import numpy as np
import utilities
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC

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
        
        self.svc = SVC()
        
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
            

    def create_features_set(self, database):
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

            #print "Loading object ", data_dict["name"]
            data = data_dict["data"]            
            features.append(self.extract_features(data))
            if self.adjective in data_dict["adjectives"]:
                labels.append(1)
            else:
                labels.append(0)
        
        self.features = np.array(features).squeeze()
        self.labels = np.array(labels).flatten()
        return self.features, self.labels

    def predict(self, X):
        return self.svc.predict(X)
    
    def train_on_features(self):
        """Train a support vector machine classifier on the features and labels
        that have been produced using self.create_features_set.
        """        
        if not hasattr(self, "features"):
            raise ValueError("No features present, have you run create_features_set?")
        if not hasattr(self, "labels"):
            raise ValueError("No labels present, have you run create_features_set?")        
        if not hasattr(self, "svc"):
            self.svc = SVC()
        
        
        score_func = f1_score
        
        #gammas = [1, 1e-1, 1e-2, 1e-3]
        gammas = np.logspace(1e-4, 10, 200)
        #gammas = [1.000230]
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': gammas,
                             'C': [1, 10, 100, 1000]},
                            ]        
        
        cv = cross_validation.StratifiedKFold(self.labels, 5)
        grid = GridSearchCV(self.svc, tuned_parameters, score_func=score_func,
                            cv=cv, verbose = 0, n_jobs=6)
        grid.fit(self.features, self.labels)
        self.svc = grid.best_estimator_
        return self
        
        
        
        
           