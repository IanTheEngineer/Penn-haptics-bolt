#! /usr/bin/python

import tga_kernel_chain
import utilities
import numpy as np
import os
import sys
import tables
import cPickle
import time
from sklearn.externals.joblib import Parallel, delayed

def test_file(f, directory):
    if f in os.listdir(directory):        
        return False
    else:
        return True

def main(base_directory, db, stats_folder):
    tga_directory = os.path.join(base_directory, "tgas")
    stats_directory = os.path.join(base_directory, stats_folder)
    
    num_objects = len(list(utilities.iterator_over_object_groups(db)))

    for adj in utilities.adjectives:
        now = time.time()
        filename = "%s.pkl" % adj
        if not test_file(filename, stats_directory):
            print "File %s already exist, skipping it" % filename 
            continue
        
        try:
            clf = tga_kernel_chain.TGAEnsemble(adj, tga_directory)
        except ValueError:
            print "Adjective %s does not exist, skipping it" % adj
        
        print "\nCreating stats for adjective ", adj
        
        positives = 0.
        total = 0.
        stat = {}
        for obj in utilities.iterator_over_object_groups(db):
            total += 1.
            res = clf.classification_labels(obj)
            output = np.mean(res) > 0.5
            true_label = (clf.adjective in obj.adjectives[:])
            if  true_label == output:
                positives += 1.
                
            stat[obj._v_name] = dict(result = output,
                             true_label = true_label,
                             classifiers = res)
            totaltime = (time.time()-now)/60.0
            sys.stdout.write("\r%d/%d ->\t precision: %.2f, time: %f min" %(total, 
                                                             num_objects, 
                                                             positives / total,
                                                             totaltime
                                                            )
                             )
            sys.stdout.flush()
            
        stat['result'] = positives / total
        
        full_pathname = os.path.join(stats_directory, filename)
        f = open(full_pathname, "w")
        cPickle.dump(stat, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
        
if __name__ == "__main__":
    if not (3 <= len(sys.argv) <= 4):
        print "Usage: %s base_directory database [stats_folder]" % sys.argv[0]
        print "tgas are in base_directory/tgas"
        print "statistics will be saved in stats_folder (default base_directory/tga_stats)"
        sys.exit(0)
    
    base_directory, db_name = sys.argv[1:3]
    if len(sys.argv) == 4:
        stats_folder = sys.argv[3]
    else:
        stats_folder = os.path.join(base_directory, 
                                    "tga_stats")
    print "The stats folder will be ", stats_folder
    
    db = tables.openFile(db_name)
    main(base_directory, db, stats_folder)