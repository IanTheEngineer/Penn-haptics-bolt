#!/usr/bin/env python

# Script to start loading data into pytables and convert into meaningful features
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import sys
import tables
import numpy as np
import cPickle
import bolt_learning_utilities as utilities
import extract_features as extract_features
from random import shuffle

from sklearn.decomposition import PCA
from bolt_pr2_motion_obj import BoltPR2MotionObj
from bolt_feature_obj import BoltFeatureObj

def load_data(input_filename, output_filename, adjective_filename, save_to_file, start_val, end_val):

    if not input_filename.endswith(".h5"):
        raise Exception("Input file is %s \nPlease pass in a hdf5 data file" % input_filename)

    if save_to_file: 
        if output_filename.endswith(".pkl"):
            output_filename = output_filename.split('.')[1]

    # Load the data from an h5 file
    all_data = tables.openFile(input_filename)

    # Flag to indicate if raw values are stored when normalizing
    discard_raw_flag = True 

    # Create a dictionary to store all the adjective objects
    adjective_data_store = dict()
   
    # Keep counter of the number of runs done
    num_runs = 0

    # Pull pointers to only the file heads of the data structure
    all_runs_root = [_g for _g in all_data.walkGroups("/train_test_sets") if _g._v_depth == 1]

    # Pull out a specific segment
    all_runs_segment = all_runs_root[0]

    # all objects number stored
    all_object_numbers = cPickle.load(open("all_objects_id.pkl","r"))
    #import pdb; pdb.set_trace() 
    print str(all_object_numbers)

    # For each file extract the segments and data
    for _adjectiveRun in all_runs_segment:
        if num_runs < start_val:
            num_runs += 1
            pass
        elif num_runs > end_val:
            break
        else:
            num_runs += 1
            adj = _adjectiveRun._v_name 
            print adj

            # Create dictionary to store train and test adjective
            adjective_set = dict()
            
            all_objs_of_adj = list()

            set_lists = ['train', 'test']

            # Go through each set - train and test
            for set_type in set_lists:
                children_set = eval('_adjectiveRun.'+ set_type)
               
                # Store the set types in the dictionary
                adjective_set[set_type] = list()

                # Pull out all runs associated with the set
                set_runs = [_r for _r in children_set.__iter__()]

                for _objectRun in set_runs: 

                    # Get the name of the current file and parse
                    object_full_name = _objectRun._v_name
                    object_name_split = object_full_name.split('_')
                    object_run_num = object_name_split[-1] 
                    object_name = "_".join(object_name_split[0:-1])

                    adjective_set[set_type].append(int(object_name.split('_')[-1]))
                    all_objs_of_adj.append(int(object_name.split('_')[-1]))

                    #all_object_numbers.append(int(object_name.split('_')[-1]))

                    #import pdb; pdb.set_trace()
            
            #print np.unique(all_objs_of_adj) 
            negative_adj = np.setxor1d(np.unique(all_objs_of_adj), all_object_numbers)
            num_train = int(round(len(negative_adj)*(2.0/3.0)))

            access_idx = range(len(negative_adj))
            shuffle(access_idx)
            access_np_idx = np.array(access_idx)

            train_idx = np.nonzero(access_np_idx <= num_train)[0]
            test_idx = np.nonzero(access_np_idx > num_train)[0]

            train_obj_neg_set = [negative_adj[i] for i in train_idx]
            test_obj_neg_set = [negative_adj[i] for i in test_idx]

            print train_obj_neg_set
            print test_obj_neg_set

            print np.unique(adjective_set['train'])
            print np.unique(adjective_set['test'])

            # Merge train and test sets
            adjective_set['train'] = np.concatenate((np.unique(adjective_set['train']), np.array(train_obj_neg_set)),axis=1)
            adjective_set['test'] = np.concatenate((np.unique(adjective_set['test']), np.array(test_obj_neg_set)),axis=1)

            if (save_to_file):

                pca_dict = None
                for set_type in set_lists:
                    file_name ='lorenzo_data/'+output_filename+'_'+adj+'_'+set_type+'.txt'
                    '''
                    if set_type == 'train':
                        pca_dict = build_feature_objects(adjective_set[set_type], file_name, adjective_filename, pca_dict)
                else:
                    build_feature_objects(adjective_set[set_type], file_name, adjective_filename, pca_dict)
                '''

                    file_ptr = open('lorenzo_data/'+output_filename+'_'+adj+'_'+set_type+'.pkl', "w")
                    cPickle.dump(np.unique(adjective_set[set_type]), file_ptr, cPickle.HIGHEST_PROTOCOL)
                    print np.unique(adjective_set[set_type])
                    print np.shape(adjective_set[set_type])
                    #file_ptr.write(str(np.unique(adjective_set[set_type])))
                    file_ptr.close()
                #cPickle.dump(adjective_set[set_type], file_ptr, cPickle.HIGHEST_PROTOCOL)
                #file_ptr.close()

        # Store the adjective away
        adjective_data_store[adj] = adjective_set
      
    all_object_sets_ptr = open('lorenzo_data/all_'+output_filename+'.pkl', "w")
    cPickle.dump(adjective_data_store, all_object_sets_ptr, cPickle.HIGHEST_PROTOCOL)
    all_object_sets_ptr.close()

    #np.array(all_object_numbers)
    #print np.unique(all_object_numbers)
    #objects_ptr = open('all_objects_id.pkl', 'w')
    #cPickle.dump(np.unique(all_object_numbers), objects_ptr, cPickle.HIGHEST_PROTOCOL)
    #objects_ptr.close()
    
    return adjective_data_store
    

def build_feature_objects(bolt_data, output_file, adjective_file, electrode_pca_dict):

    # Inserts adjectives into the bolt_data  
    all_bolt_data_adj = utilities.insertAdjectiveLabels(bolt_data, output_file, adjective_file, False)

    if electrode_pca_dict == None:
        print "Building Train Set"
        pca_dict = extract_features.fit_electrodes_pca(all_bolt_data_adj)
    else:
        pca_dict = electrode_pca_dict
        print "Building Test Set"
    
    # Load pickle file
    #pca_dict = cPickle.load(open('pca.pkl', 'r'))

    all_feature_obj = BoltMotionObjToFeatureObj(all_bolt_data_adj, pca_dict) 

    cPickle.dump(all_feature_obj, open(output_file, 'w')) 

    return pca_dict


# Takes the bolt data and extracts features to run
def BoltMotionObjToFeatureObj(all_bolt_data, electrode_pca_dict):
    """ 

    For each object - pull out features and store in feature_obj
    with the same structure as all_bolt_data
   
        Dictionary - "tap", "slide", "slow_slide", 
                     "thermal_hold", "squeeze"

    """

    # Store in feature class object
    all_features_obj_dict = dict();

    for motion_name in all_bolt_data:
        trial_list = all_bolt_data.get(motion_name)
        print motion_name

        feature_list = list()
        # For all objects
        for trial in trial_list:

            bolt_feature_obj = extract_features.extract_features(trial, electrode_pca_dict[motion_name])

            feature_list.append(bolt_feature_obj)

        # Store all of the objects away
        all_features_obj_dict[motion_name] = feature_list

    return all_features_obj_dict


def main():

    # Parse out the arguments passed in 
    if len(sys.argv) < 6:
        raise Exception("Usage: %s [input_file] [output_file] [adjective_file] [start_val] [end_val]", sys.argv[0])

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    adjective_filename = sys.argv[3]
    start_val = int(sys.argv[4])
    end_val = int(sys.argv[5])

    return input_filename, output_filename, adjective_filename, start_val, end_val

if __name__== "__main__":
    input_file, output_file, adjective_filename, start_val, end_val = main() 
    load_data(input_file, output_file, adjective_filename, True, start_val, end_val)

