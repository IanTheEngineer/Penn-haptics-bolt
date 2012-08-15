#! /usr/bin/python
"""
Lists all the files that have object_id in their name, where object_id is extracted
from the h5 file containing all the adjectives, given some conditions.

Example usage: ./list_files_attributes.py adjectives.h5 (hard==1) & (thin==0)
"""

import tables
import glob
import os
import sys

if len(sys.argv) < 3:
    print "usage: %s input_file.h5 condition_string" % sys.argv[0]
    print "\texample: %s adjectives.h5 (hard==1) & (thin==0)"  % sys.argv[0]
    sys.exit(1)
    
input_file = sys.argv[1]
(path, name) = os.path.split(input_file)
h5file = tables.openFile(input_file, "r")

where_condition = " ".join(sys.argv[2:])

table = h5file.root.clases
cond = [x["object_id"] for x in table.where(where_condition)]
files = [f for f in glob.glob(os.path.join(path,"*.bag")) if any(s in f for s in cond)]

print " ".join(os.path.join(path,f) for f in files)
h5file.close()
sys.exit(0)