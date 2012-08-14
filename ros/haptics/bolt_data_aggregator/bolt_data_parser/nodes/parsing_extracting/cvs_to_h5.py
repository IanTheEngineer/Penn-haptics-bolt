#! /usr/bin/python
"""Reads a csv file and converts it to a h5 file"""

import tables
import csv
import sys
from optparse import OptionParser
import os
import itertools

def populate_h5(input_file, out_file, values_slice):
    
    csv_file = open(input_file, "r")
    reader = csv.reader(csv_file)
    
    #getting the headers of, the file, assume they're there    
    titles = reader.next()[1:-2] #first column is the object number, let's skip it for now
    
    csv_file.seek(0)
    
    def filter(x):
        try:
            return int(x)
        except ValueError:
            return 0    
    
    all_rows = []
    for row in itertools.islice(reader, values_slice.start, values_slice.stop, values_slice.step):
        class_number = row[0]
        new_row = [class_number]
        new_row.extend(filter(x) for x in row[1:-2]) #skip class_number and the last two empty elements
        all_rows.append(new_row)
        
    #now let's go to the h5
    h5file = tables.openFile(out_file, mode="w", title="Adjectives")
    
    description = dict(zip(titles, (tables.Int8Col(pos=i+1) for i in xrange(len(titles))) ))
    description["object_id"] = tables.StringCol(8, pos=0)
    table = h5file.createTable("/", "clases", description)
    table.append(all_rows)
    table.flush()
    h5file.close()

def parse_arguments():
    """Parses the arguments provided at command line.
    
    Returns:
    (input_file, output_file, range)
    """
    parser = OptionParser()
    parser.add_option("-i", "--input", action="store", type="string", dest = "in_file")
    parser.add_option("-o", "--output", action="store", type="string", dest = "out_file", default = None)
    parser.add_option("-r", "--range", action="store", type="string", dest = "range", 
                      help="Select the row range, format is: [start,] stop[, step]")

    (options, args) = parser.parse_args()
    input_file = options.in_file #this is required
    
    if options.out_file is None:
        (_, name) = os.path.split(input_file)
        name = name.split(".")[0]
        out_file = name + ".h5"
    else:        
        out_file = options.out_file
        if len(out_file.split(".")) == 1:
            out_file = out_file + ".h5"
    
    r_str = options.range
    s = slice(*map(lambda x: int(x.strip()) if x.strip() else None, r_str.split(':')))
    
    return input_file, out_file, s


if __name__ == "__main__":
    input_file, out_file, s = parse_arguments()
    populate_h5(input_file, out_file, s)