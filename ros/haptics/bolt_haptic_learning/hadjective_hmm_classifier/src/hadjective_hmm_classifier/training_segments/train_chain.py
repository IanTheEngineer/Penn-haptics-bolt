#! /usr/bin/python
import cPickle
import sklearn.grid_search
import sklearn.cross_validation
import hmm_chain
import os
import sys
import itertools
from utilities import adjectives, phases, sensors
import multiprocessing

def test_chain(chain, path, original_adjective, phase, sensor):
    chain.my_class = None
    chain.other_classes = None
    
    def load_data(__adjective):
        filename = os.path.join(path, __adjective + ".pkl")
        with open(filename) as f:
            data = cPickle.load(f)
            return data[phase][sensor]        
    
    original_score = chain.score(load_data(original_adjective))
    print "Original adjective %s: %f" %(original_adjective,
                                        original_score)
    goods = 0.0
    total = 0.0
    for adjective in adjectives:
        if adjective == original_adjective:
            continue
        s = chain.score(load_data(adjective))
        if s > original_score:
            atchm = "\t\tTHIS IS BAD ======="            
        else:
            atchm = "\t\tTHIS IS GOOD"
            goods += 1
        total +=1
        
        print ("Adjective: %s, score: %.3f" + atchm) % (adjective, 
                                                      s)
    print "Negatives: %d, Ratio: %f" % (total-goods, goods/ total)
                                        

def calculate_rations(chain_path, adjectives_path):

    def load_data(__adjective, __phase, __sensor):
        filename = os.path.join(adjectives_path, __adjective + ".pkl")
        with open(filename) as f:
            data = cPickle.load(f)
            return data[__phase][__sensor]        
    
    ratios = []
    for filename in os.listdir(chain_path):
        if not filename.endswith('.pkl'):
            continue        
        chars = filename.strip(".pkl").split("_")
        chars = chars[1:] #chain 
        original_adjective = chars[0]
        chars = chars[1:] #adjective
        sensor = chars.pop()
        phase = "_".join(chars) #silly me for the choice of separator!        
        chain = cPickle.load(open(os.path.join(chain_path,filename)))
        
        original_score = chain.score(load_data(original_adjective,
                                               phase, 
                                               sensor))
        goods = 0.0
        total = 0.0
        for adjective in adjectives:
            if adjective == original_adjective:
                continue
            s = chain.score(load_data(adjective, phase, sensor))
            if s > original_score:
                pass
            else:
                goods += 1
            total +=1
            
        ratio = goods/ total
        print "Adjective: %s, phase: %s, sensor: %s, ratio: %f" %(original_adjective,
                                                                  phase,
                                                                  sensor,
                                                                  ratio)
                                                                  
        ratios.append(ratio)
    return ratios 


def train_and_save(parameters, dataset, filename):
    chain = hmm_chain.HMMChain()
    chain.set_params(**parameters)
    chain.my_class = None
    chain.other_classes = None
    
    chain.fit(dataset)

    score = chain.score(dataset)
    print "After training the score is ", score
    
    with open(filename, "w") as f:
        print "Saving file: ", filename
        cPickle.dump(chain, f, protocol=cPickle.HIGHEST_PROTOCOL)
        


def train_dataset(dataset, all_adjectives, adjective):

    parameters = [
              dict(n_pca_components = [0.97],
                   n_hidden_components=[35, 40, 45], 
                   resampling_size=[20], 
                   n_discretization_symbols=[25,],
                   hmm_max_iter = [2000],
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
        
    print "Using parameters:\n", parameters    
    
    chain = hmm_chain.HMMChain()
    cross_validator = sklearn.cross_validation.ShuffleSplit(len(dataset), 
                                                            n_iterations=2, 
                                                            train_size=3./4.)
    
    for p in parameters:
        p.update(my_class = [adjective],
                 other_classes = [all_adjectives]
                 )
    grid = sklearn.grid_search.GridSearchCV(chain, parameters,
                                            cv = cross_validator,
                                            verbose = 10,
                                            n_jobs = 6,
                                            refit = False                                            
                                            )
    grid.fit(dataset)
    
    return grid.best_params_

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

def load_all_adjectives(path, phase, sensor):
    if phase not in phases:
        raise ValueError("%s is not a known phase" % phase)
    if sensor not in sensors:
        raise ValueError("%s is not a known sensor" % sensor)    

    all_data = {}
    
    for adjective in adjectives:
        filename = os.path.join(path, adjective + ".pkl")
        with open(filename) as f:
            data = cPickle.load(f)    
            all_data[adjective] = data[phase][sensor]
    
    return all_data

def train_single_dataset(path, adjective, phase, sensor):
    
    dataset = load_dataset(path, adjective, phase, sensor)
    #all_adjectives = load_all_adjectives(path, phase, sensor)
    all_adjectives = (path, phase, sensor)
    
    if len(dataset) is 0:
        print "Empty dataset???"
        return
    
    chain_file_name = "_".join(("chain", adjective, phase, sensor)) + ".pkl"
    newpath = os.path.join(path, "chains")
    path_name = os.path.join(newpath, chain_file_name)    
    if os.path.exists(path_name):
        print "File %s already exists, skipping it." % path_name
        return
    
    print "Training adjective %s, phase %s, sensor %s" %(
        adjective, phase, sensor)
    
    parameters = train_dataset(dataset, all_adjectives, adjective)
    print "Parameters finished, spawning a process to save..."
        
    p = multiprocessing.Process(target = train_and_save,
                                args = (parameters, dataset, path_name)
                                )
    p.daemon = False
    p.start()

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
        path, adjective = sys.argv[1:]
        print "Training all the phases and sensors for adjective %s" %(
                    adjective)
        for phase, sensor in itertools.product(phases, sensors):
            train_single_dataset(path, adjective, phase, sensor)
    elif len(sys.argv) == 2:
        path = sys.argv[1]
        print "Training all combinations of adjectives, phases and sensor"
        for adjective, phase, sensor in itertools.product(adjectives, 
                                                          phases, sensors):
            #try:
                #train_single_dataset(path, adjective, phase, sensor)
            #except Exception, e:
                #print "Got a problem, error is: ", e
            train_single_dataset(path, adjective, phase, sensor)
    else:
        print "Usage:"
        print "%s path adjective phase sensor" % sys.argv[0]
        print "%s path phase sensor" % sys.argv[0]
        print "%s path adjective" % sys.argv[0]
        print "%s path" % sys.argv[0]
        print "Files will be saved in path/chains"

if __name__ == "__main__":
    main()
    print "done"
    