import tables
import numpy as np
import sys
import collections
import cPickle
import time
import os

def aggregate(h5file, adjective, phase, field):
    assert isinstance(h5file, tables.File)
    
    adjectives_group = h5file.root.adjectives
    isinstance(adjectives_group, tables.Group)
    single_adjective = getattr(adjectives_group, adjective)
    isinstance(single_adjective, tables.Group)
    
    all_values = collections.deque()
    for element in single_adjective:
        isinstance(element, tables.Group)
        indexed = (element.state.controller_detail_state.read() == phase)
        
        #finger 0
        finger_0 = element.biotacs.finger_0
        data_0 = getattr(finger_0, field).read()
        nrows = data_0.shape[0]
        data_0 = data_0.reshape((nrows,-1))
        data_0 = data_0[indexed, :]
        
        finger_1 = element.biotacs.finger_1
        data_1 = getattr(finger_1, field).read()
        nrows = data_1.shape[0]
        data_1 = data_1.reshape((nrows,-1))
        data_1 = data_1[indexed, :]        
        
        all_values.append( np.hstack((data_0, data_1)) )
    
    return list(all_values)
        
            

def elaborate_file(adjective):
    h5file = tables.openFile(sys.argv[1])
    interesting_sets = ["SQUEEZE_SET_PRESSURE_SLOW", "HOLD_FOR_10_SECONDS", "SLIDE_5CM", "MOVE_DOWN_5CM"]
    all_fields = ["electrodes", "pac", "pdc", "tac", "tdc"]    

    all_iterations = len(interesting_sets) * len(all_fields)
    current_iteration = 1
    
    all_data = {}    
    total_start_time = time.time()
    all_times = []
    
    
    for phase in interesting_sets:
        phase_data = {}
        for field in all_fields:                        
            print "Iteration: %d/%d" % (current_iteration, all_iterations)
            current_iteration += 1                        
            
            single_iteration_time = time.time()            
            values = aggregate(h5file, adjective, phase, field)
            elapsed_this_iteration = time.time() - single_iteration_time            
            
            all_times.append(elapsed_this_iteration)
            print "Time for this iteration: %.2f, average x iteration: %.2f, total elapsed time: %.2f" % (
                elapsed_this_iteration, np.mean(all_times),  time.time() - total_start_time)
            
            phase_data[field] = values
        all_data[phase] = phase_data

    (path, _) = os.path.split(sys.argv[1])
    output_file_name = os.path.join(path, adjective + ".pkl")
    
    with open(output_file_name, "w") as f:
        cPickle.dump(all_data, f, cPickle.HIGHEST_PROTOCOL)
        
    h5file.close()
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: %s input_database" % sys.argv[0]
        sys.exit(1)
    
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
    for i, adjective in enumerate(adjectives):
        print "Adjective %d/%d: %s" % (i, len(adjectives), adjective)
        elaborate_file(adjective)
    print "Done"
        