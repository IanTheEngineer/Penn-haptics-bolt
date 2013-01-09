#! /usr/bin/python
"""
Given the database with all the data and a separate database with the adjectives (created using cvs_to_h5.py),
create hard links in the original database to group together all the datasets belonging to the same database.
"""


import tables
import sys
import os
import numpy as np
import cPickle

def create_hard_links(h5file, adjective, groups, group_name):
    assert isinstance(h5file, tables.file.File)
    
    #newg = h5file.createGroup("/adjectives", adjective, createparents=True)
    newg = h5file.createGroup(group_name, adjective, createparents=True)
    
    for g in groups:
        name = g._v_name
        h5file.createHardLink(newg, name, g)

def main():
    if len(sys.argv) < 3:
        print "usage: %s /path/to/adjective.h5 /path/to/big_database.h5 adjective" % sys.argv[0]
        print "\texample: %s adjectives.h5 all_items.h5 thin"  % sys.argv[0]
        sys.exit(1)
        
    adjectives_file = sys.argv[1]    
    adjective_h5 = tables.openFile(adjectives_file, "r")
    
    table = adjective_h5.root.clases
    all_adjectives = [n for n in table.description._v_names if n != "object_id"]

    all_objects_file = sys.argv[2]    
    all_objects = tables.openFile(all_objects_file, "r+")
    
    for i, adjective_name  in enumerate(all_adjectives):
        
        test_exist = "/adjectives/%s" % adjective_name
        if test_exist in all_objects:
            print "Adjective %s already exists, removing it" % adjective_name
            all_objects.removeNode("/adjectives", adjective_name, recursive=True)
        
        where_condition = "%s > 0" % (adjective_name)                
        cond = [x["object_id"] for x in table.where(where_condition)]        

        print "(%d/%d) Parsing h5 file for adjective %s" % (i, len(all_adjectives), adjective_name)
        groups = [g for g in all_objects.root._v_children.values() if any(c in g._v_name for c in cond)]
        
        print "I've got %d groups. Adding hard links" % len(groups)
        create_hard_links(all_objects, adjective_name, groups, "/adjectives")

    for i, adjective_name  in enumerate(all_adjectives):

        test_exist = "/adjectives_neg/%s" % adjective_name
        if test_exist in all_objects:
            print "Negative Examples for Adjective %s already exist, removing it" % adjective_name
            all_objects.removeNode("/adjectives_neg", adjective_name, recursive=True)

        where_condition = "%s == 0" % (adjective_name)
        cond = [x["object_id"] for x in table.where(where_condition)]

        print "(%d/%d) Parsing h5 file for adjective %s" % (i, len(all_adjectives), adjective_name)
        groups = [g for g in all_objects.root._v_children.values() if any(c in g._v_name for c in cond)]

        print "I've got %d groups. Adding hard links" % len(groups)
        create_hard_links(all_objects, adjective_name, groups, "/adjectives_neg")


    adjective_h5.close()
    all_objects.close()
    print "Done"
    
if __name__ == "__main__":
    main()
