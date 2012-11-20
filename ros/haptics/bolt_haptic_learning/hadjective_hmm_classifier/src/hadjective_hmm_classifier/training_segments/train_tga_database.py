#! /usr/bin/python
import cPickle
import sklearn.grid_search
import sklearn.cross_validation
import sklearn.metrics
import tga_kernel_chain
import os
import sys
import itertools
import utilities
from utilities import adjectives, phases, sensors
import multiprocessing
import tables
import traceback
import numpy as np

def train_and_save(parameters, dataset, filename):
    """Creates a TGAChain and, loads its parameters, trains it on dataset and
    saves the results in filename.    
    """
    
    chain = tga_kernel_chain.TGAChain()
    chain.set_params(**parameters)
    chain.my_class = None
    chain.other_classes = None
    
    inputs, outputs, _, _ = create_dataset_crossvalidation(dataset)    
    chain.fit(inputs, outputs)

    score = chain.score(inputs, outputs)
    
    display_name = os.path.split(filename)[1].split(".")[0]
    longline = "========%s========" % display_name
    print
    print longline
    print "After training the score is ", score
    print "with parameters: %s" % parameters
    print "=" * len(longline)
    print
    
    with open(filename, "w") as f:
        print "Saving file: ", filename
        cPickle.dump(chain, f, protocol=cPickle.HIGHEST_PROTOCOL)


def train_dataset(dataset):
    """Uses cross validation to train a HMM chain. Returns the parameters 
    that yield the best result.
    
    CHANGE THE PARAMETERS HERE TO USE A SUITABLE RANGE, THESE ARE ONLY HERE FOR
    TESTING!
    """
    
    inputs, outputs, train_indexes, test_indexes = create_dataset_crossvalidation(dataset)
    cv = [(train_indexes, test_indexes)]
    
    #parameters = [
    #          dict(n_pca_components = [0.95, 0.97],                   
    #               resampling_size=[20, 50, 100, 200],
    #               whiten = [False, True],
    #               C = [50, 500, 1000],
    #               T_multiplier = [0.1, 0.25, 0.5, 0.7],
    #               sigma_multiplier = [0.01, 0.1, 1.0, 2, 5, 10],
    #               class_weight = ('auto', None)
    #               ),          
    #          ]
    parameters = [
              dict(n_pca_components = [0.95],                   
                   resampling_size=[50, 100],
                   whiten = [True,],
                   C = [500],
                   T_multiplier = [0.25, 0.5],
                   sigma_multiplier = [5, 10, 15],
                   class_weight = ('auto',)
                   ),          
              ]
             
    chain = tga_kernel_chain.TGAChain()
    grid = sklearn.grid_search.GridSearchCV(chain, parameters,
                                            cv = cv,
                                            verbose = 10,
                                            n_jobs = 12,
                                            refit = False,
                                            score_func = sklearn.metrics.f1_score,
                                            )
    grid.fit(inputs, outputs)
    
    return grid.best_params_

def create_dataset_crossvalidation(dataset):
    """Uses a nicely organized h5 file to return train and test sets in a way
    that can be used with GridSearch."""
    
    train, test = dataset
    train_indexes = range(len(train[0]))
    test_indexes = range(len(train[0]), len(train[0]) + len(test[0]))
    
    all_inputs = train[0] + test[0]
    all_outputs = train[1] + test[1]
    
    return (all_inputs,
            all_outputs,
            train_indexes,
            test_indexes)

def load_dataset(database, adjective, phase, sensor):
    """Loads the data from a dataset corresponding to an adjective, phase and
    sensor."""
    
    msg = []
    if adjective not in adjectives:
        raise ValueError("%s is not a known adjective" % adjective)
    if phase not in phases:
        raise ValueError("%s is not a known phase" % phase)
    if sensor not in sensors:
        raise ValueError("%s is not a known sensor" % sensor)    
    
    included_names = set()    
    train_set = []
    test_set = []
    train_group = database.getNode("/train_test_sets", adjective).train
    for name, g in train_group._v_children.iteritems():
        if name not in included_names:
            train_set.append(utilities.dict_from_h5_group(g, 
                                                          [phase], 
                                                          [sensor])["data"][phase][sensor]
                             )
            included_names.add(name)
            #msg.append("0 Adding " + name + " to positive train")
                    
    test_group = database.getNode("/train_test_sets", adjective).test
    for name, g in test_group._v_children.iteritems():
        if name not in included_names:            
            test_set.append(utilities.dict_from_h5_group(g, 
                                                         [phase], 
                                                         [sensor])["data"][phase][sensor]
                            )
            included_names.add(name)
            #msg.append("1 Adding " + name + " to positive test")
    
    train_label = [1] * len(train_set)
    test_label =  [1] * len(test_set)
    
    #now take all the other adjectives for negative class
    for other_adj in database.getNode("/train_test_sets"):
        if other_adj._v_name == adjective:
            continue
        
        train_group = other_adj.train
        for name, g in train_group._v_children.iteritems():
            if name not in included_names:
                train_set.append(utilities.dict_from_h5_group(g, 
                                                          [phase], 
                                                          [sensor])["data"][phase][sensor]                        
                                 )
                included_names.add(name)
                #msg.append("2 Adding " + name  +" to negative train")
        
        test_group = other_adj.test
        for name, g in test_group._v_children.iteritems():
            if name not in included_names:        
                test_set.append(utilities.dict_from_h5_group(g, 
                                                         [phase],                                                          
                                                         [sensor])["data"][phase][sensor]
                                )
                included_names.add(name)
                #msg.append("3 Adding " + name  +" to negative test")
                        
    train_label += [0] * (len(train_set) - len(train_label))
    test_label += [0] * (len(test_set) - len(test_label))
    
    return (train_set, train_label), (test_set, test_label)

def train_single_dataset(database, path, adjective, phase, sensor):
    """Trains a HMM on single segment, i.e. one adjective, one phase and one
    sensor."""
    
    chain_file_name = "_".join(("tga", adjective, phase, sensor)) + ".pkl"
    newpath = os.path.join(path, "tgas")
    path_name = os.path.join(newpath, chain_file_name)    
    if os.path.exists(path_name):
        print "File %s already exists, skipping it." % path_name
        return
        
    print "Training adjective %s, phase %s, sensor %s" %(
        adjective, phase, sensor)
    
    database = tables.openFile(database)
    dataset = load_dataset(database, adjective, phase, sensor)
    
    if len(dataset) is 0:
        print "Empty dataset???"
        return    
    
    try:
        parameters = train_dataset(dataset)
        print "Parameters finished, spawning a process to save..."
            
        p = multiprocessing.Process(target = train_and_save,
                                    args = (parameters, dataset, path_name)
                                    )
        p.daemon = False
        p.start()
    except:
        raise
        #print "==========ERROR WHILE DOING ", (adjective, phase, sensor)
        #tb = traceback.format_exc()
        #print "Traceback: \n", tb
        

def main():
    if len(sys.argv) == 6:
        database, path, adjective, phase, sensor = sys.argv[1:]
        train_single_dataset(database, path, adjective, phase, sensor)
    elif len(sys.argv) == 5:
        database, path, phase, sensor = sys.argv[1:]
        print "Training all the adjectives for phase %s and sensor %s" %(
            phase, sensor)
        for adjective in adjectives:
            train_single_dataset(database, path, adjective, phase, sensor)
    elif len(sys.argv) == 4:
        database, path, adjective = sys.argv[1:]
        print "Training all the phases and sensors for adjective %s" %(
                    adjective)
        for phase, sensor in itertools.product(phases, sensors):
            train_single_dataset(database, path, adjective, phase, sensor)
    elif len(sys.argv) == 3:
        database, path = sys.argv[1:]
        print "Training all combinations of adjectives, phases and sensor"
        for adjective, phase, sensor in itertools.product(adjectives, 
                                                          phases, sensors):

            train_single_dataset(database, path, adjective, phase, sensor)
    else:
        print "Usage:"
        print "%s database path adjective phase sensor" % sys.argv[0]
        print "%s database path phase sensor" % sys.argv[0]
        print "%s database path adjective" % sys.argv[0]
        print "%s database path" % sys.argv[0]
        print "Files will be saved in path/tgas"

if __name__ == "__main__":
    main()
    print "done"
    
