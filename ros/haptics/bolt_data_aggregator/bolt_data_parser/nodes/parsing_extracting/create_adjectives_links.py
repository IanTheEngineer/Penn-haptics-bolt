#! /usr/bin/python
"""
Augment a h5 file, with entries for each object, with the corresponding 
adjective.
"""
import tables
import numpy as np
import sys
import collections
import cPickle
import time
import os

def elaborate_file(adjective):
    pass
    
def main():
    if len(sys.argv) != 3:
        print "usage: %s input_database adjectives_database" % sys.argv[0]
        sys.exit(1)
    
    #finding the list of all the objects
    main_database = tables.openFile(sys.argv[1], "r+")
    adjectives_database = tables.openFile(sys.argv[2])
    table = adjectives_database.root.clases
    all_objects = [v for v in main_database.root._v_children.values() 
                   if v._v_name != "adjectives"]
    
    adjective_names = [n for n in table.colnames if n != "object_id"]
    for obj in all_objects:
        isinstance(obj, tables.Group)
        
        #guessing the object id
        splits = obj._v_name.split("_")
        obj_id = splits[-2]

        #entry in the table
        row = table.where('object_id == "%s"' % obj_id).next()
        adjectives = [n for n in adjective_names if row[n] > 0]
        main_database.createArray(obj,
                                  "adjectives",
                                  adjectives
                                  )
        print "Object %s, adjectives: %s" % (obj._v_name, adjectives)
    main_database.close()
    adjectives_database.close()
        
        
    
if __name__ == "__main__":
    main()
    print "Done"
    
        