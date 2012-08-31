#! /usr/bin/python
import cPickle
import sklearn.grid_search
import sklearn.cross_validation
import hmm_chain
import os
import sys
import itertools
import utilities
from utilities import adjectives, phases, sensors
import multiprocessing
import tables

def train_and_save(parameters, dataset, filename):
    chain = hmm_chain.HMMChain()
    chain.set_params(**parameters)
    chain.my_class = None
    chain.other_classes = None
    
    dataset, _, _ = create_dataset_crossvalidation(dataset)
    chain.fit(dataset)

    score = chain.score(dataset)
    
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
    dataset, train_indexes, test_indexes = create_dataset_crossvalidation(dataset)
    cv = [(train_indexes, test_indexes)]
    
    parameters = [
              dict(n_pca_components = [0.97],
                   n_hidden_components=[12, 15, 18], 
                   resampling_size=[20, 25, 30], 
                   n_discretization_symbols=[5, 10, 12],
                   hmm_max_iter = [100],
                   #kmeans_max_iter = [1000]
                   ),  
              #dict(n_pca_components = [0.97],
                   #n_hidden_components=[40, 50], 
                   #resampling_size=[20], 
                   #n_discretization_symbols=[30, ],
                   #hmm_max_iter = [2000],
                   ##kmeans_max_iter = [1000]
                   #),              
              ]
        
    #print "Using parameters:\n", parameters    
    
    chain = hmm_chain.HMMChain()
    
    grid = sklearn.grid_search.GridSearchCV(chain, parameters,
                                            cv = cv,
                                            verbose = 10,
                                            n_jobs = 6,
                                            refit = False                                            
                                            )
    grid.fit(dataset)
    
    return grid.best_params_

def create_dataset_crossvalidation(dataset):
    train, test = dataset
    train_indexes = range(len(train))
    test_indexes = range(len(train), len(train) + len(test))
    return (train + test,
            train_indexes,
            test_indexes)

def load_dataset(database, adjective, phase, sensor):
    if adjective not in adjectives:
        raise ValueError("%s is not a known adjective" % adjective)
    if phase not in phases:
        raise ValueError("%s is not a known phase" % phase)
    if sensor not in sensors:
        raise ValueError("%s is not a known sensor" % sensor)    

    train_group = database.getNode("/train_test_sets", adjective).train
    train_set = [utilities.dict_from_h5_group(g, [phase], [sensor])["data"][phase][sensor]
                    for g in train_group._v_children.values()]
    test_group = database.getNode("/train_test_sets", adjective).test
    test_set = [utilities.dict_from_h5_group(g, [phase], [sensor])["data"][phase][sensor]
                    for g in test_group._v_children.values()]
    
    return train_set, test_set

def train_single_dataset(database, path, adjective, phase, sensor):
    
    chain_file_name = "_".join(("chain", adjective, phase, sensor)) + ".pkl"
    newpath = os.path.join(path, "chains")
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
        print "==========ERROR WHILE DOING ", (adjective, phase, sensor)
        

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
        print "Files will be saved in path/chains"

if __name__ == "__main__":
    main()
    print "done"
    