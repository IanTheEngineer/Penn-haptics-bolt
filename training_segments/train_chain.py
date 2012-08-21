#! /usr/bin/python
import cPickle
import sklearn.grid_search
import sklearn.cross_validation
import hmm_chain
import os
import sys
import itertools

adjectives = ['sticky',
              'deformable',
              'hard',
              'hollow',
              'springy',
              'fuzzy',
              'rough',
              'thick',
              'compact',
              'elastic',
              'smooth',
              'metallic',
              'unpleasant',
              'plasticky',
              'meshy',
              'nice',
              'hairy',
              'compressible',
              'fibrous',
              'squishy',
              'gritty',
              'textured',
              'bumpy',
              'grainy',
              'scratchy',
              'cool',
              'absorbant',
              'stiff',
              'solid',
              'crinkly',
              'porous',
              'warm',
              'slippery',
              'thin',
              'sparse',
              'soft']
phases = ["SQUEEZE_SET_PRESSURE_SLOW", "HOLD_FOR_10_SECONDS", 
          "SLIDE_5CM", "MOVE_DOWN_5CM"]
sensors = ["electrodes", "pac", "pdc", "tac", "tdc"]

def train_dataset(dataset, parameters):
    chain = hmm_chain.HMMChain()
    cross_validator = sklearn.cross_validation.ShuffleSplit(len(dataset), 
                                                            n_iterations=3, 
                                                            train_size=3./4.)
    grid = sklearn.grid_search.GridSearchCV(chain, parameters,
                                            cv = cross_validator,
                                            verbose = 10,
                                            n_jobs = 6
                                            )
    grid.fit(dataset)
    return grid.best_estimator_


def load_dataset(path, adjective, phase, sensor):
    if adjective not in adjectives:
        raise ValueError("%s is not a known adjective" % adjective)
    if phase not in phases:
        raise ValueError("%s is not a known phase" % phase)
    if sensor not in sensors:
        raise ValueError("%s is not a known sensor" % sensor)    
    
    filename = os.path.join(path, adjective + ".pkl")
    with open(filename) as f:
        data = cPickle.load(f)
    
    return data[phase][sensor]

def train_single_dataset(path, adjective, phase, sensor):
    dataset = load_dataset(path, adjective, phase, sensor)
    
    chain_file_name = "_".join(("chain", adjective, phase, sensor)) + ".pkl"
    newpath = os.path.join(path, "chains")
    path_name = os.path.join(newpath, chain_file_name)    
    if os.path.exists(path_name):
        print "File %s already exists, skipping it." % path_name
        return
    
    print "Training adjective %s, phase %s, sensor %s" %(
        adjective, phase, sensor)
    
    params = dict(n_pca_components = [0.95],
                  n_hidden_components=[5, 8, 11, 14], 
                  resampling_size=[10,20,30], 
                  n_discretization_symbols=[3, 4, 5, 6, 7])
    print "Using parameters:\n", params
    chain = train_dataset(dataset, params)
    
    print "After training, score is ", chain.score(dataset)
    
    
    with open(path_name, "w") as f:
        cPickle.dump(chain, f, protocol=cPickle.HIGHEST_PROTOCOL)    
    

def main():
    if len(sys.argv) == 5:
        path, adjective, phase, sensor = sys.argv[1:]
        train_single_dataset(path, adjective, phase, sensor)
    elif len(sys.argv) == 4:
        path, phase, sensor = sys.argv[1:]
        print "Training all the adjectives for phase %s and sensor %s" %(
            phase, sensor)
        for adjective in adjectives:
            train_single_dataset(path, adjective, phase, sensor)
    elif len(sys.argv) == 3:
        path, sensor = sys.argv[1:]
        print "Training all the adjectives and phases for sensor %s" %(
                    sensor)
        for adjective, phase in itertools.product(adjectives, phases):
            train_single_dataset(path, adjective, phase, sensor)
    else:
        print "Usage:"
        print "%s path adjective phase sensor" % sys.argv[0]
        print "%s path phase sensor" % sys.argv[0]
        print "%s path sensor" % sys.argv[0]
    

if __name__ == "__main__":
    main()
    print "done"
    