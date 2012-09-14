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

from sklearn.decomposition import PCA
from bolt_pr2_motion_obj import BoltPR2MotionObj
from bolt_feature_obj import BoltFeatureObj

# Pulls out the data for a segment from a run
def PullDataFromRun(one_run_pytable_ptr, pull_state):
    
    # pull out controller state array
    state_array = one_run_pytable_ptr.state.controller_state[:]
    # Get index into the array that matches the state 
    idx_segment = np.nonzero(state_array == pull_state)

    # Create PR2MotionObj
    motion_object = BoltPR2MotionObj()

    # Store the state in the object
    motion_object.state = pull_state

    # Get the name of the current file and parse
    object_full_name = one_run_pytable_ptr._v_name
    object_name_split = object_full_name.split('_')
    object_run_num = object_name_split[-1] 
    object_name = "_".join(object_name_split[0:-1])

    # Store parsed info
    motion_object.name = object_name
    motion_object.run_number = int(object_run_num)

    # Biotac information
    for _finger in xrange(2):#one_run_pytable_ptr.biotacs._v_depth):

        # Create name to eval for finger
        finger_name_list = [] 
        finger_name_list.append('one_run_pytable_ptr.biotacs.finger_')
        finger_name_list.append(str(_finger))

        finger_name = ''.join(finger_name_list)

        # Electrodes
        one_set_electrode = eval(finger_name + '.electrodes[:]')
        one_motion_electrode = np.array(one_set_electrode[idx_segment])
        motion_object.electrodes.append(one_motion_electrode)
        motion_object.electrodes_mean.append(np.array(one_set_electrode[100:110, :]))

        # TDC
        one_set_tdc = eval(finger_name + '.tdc[:]')
        one_motion_tdc = np.array(one_set_tdc[idx_segment])
        motion_object.tdc.append(one_motion_tdc)
        motion_object.tdc_mean.append(np.array(one_set_tdc[100:110]))

        # TAC
        one_set_tac = eval(finger_name + '.tac[:]')
        one_motion_tac = np.array(one_set_tac[idx_segment])
        motion_object.tac.append(one_motion_tac)
        motion_object.tac_mean.append(np.array(one_set_tac[100:110]))

        # PDC
        one_set_pdc = eval(finger_name + '.pdc[:]')
        one_motion_pdc = np.array(one_set_pdc[idx_segment])
        motion_object.pdc.append(one_motion_pdc)
        motion_object.pdc_mean.append(np.array(one_set_pdc[100:110]))

        # PAC
        one_set_pac = eval(finger_name + '.pac[:]')
        one_motion_pac = np.array(one_set_pac[idx_segment])
        motion_object.pac.append(one_motion_pac)
        motion_object.pac_mean.append(np.array(one_set_pac[100:110, :]))

        #pac_flat.append(one_motion_pac_flat.reshape(1, len(one_motion_pac_flat)*22)[0])
   
    # Store gripper information
    # Velocity 
    gripper_velocity = one_run_pytable_ptr.gripper_aperture.joint_velocity[:] 
    motion_object.gripper_velocity = gripper_velocity[idx_segment]
   
    # Position
    gripper_position = one_run_pytable_ptr.gripper_aperture.joint_position[:] 
    motion_object.gripper_position = gripper_position[idx_segment]

    # Motor Effort
    gripper_effort = one_run_pytable_ptr.gripper_aperture.joint_effort[:] 
    motion_object.gripper_effort = gripper_effort[idx_segment]

    # Store accelerometer
    accelerometer = one_run_pytable_ptr.accelerometer[:] 
    motion_object.accelerometer = accelerometer[idx_segment]
   
    # Transforms
    l_tool_frame_transform_rot = one_run_pytable_ptr.transforms.rotation[:]
    motion_object.l_tool_frame_transform_rot = l_tool_frame_transform_rot[idx_segment]
   
    l_tool_frame_transform_trans = one_run_pytable_ptr.transforms.translation[:]
    motion_object.l_tool_frame_transform_trans = l_tool_frame_transform_trans[idx_segment]
   
    # Store detailed states
    detailed_state = one_run_pytable_ptr.state.controller_detail_state[:]
    motion_object.detailed_state = detailed_state[idx_segment].tolist()

    return motion_object


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
            adjective_set = dict();

            set_lists = ['train', 'test']

            # Go through each set - train and test
            for set_type in set_lists:
                children_set = eval('_adjectiveRun.'+ set_type)
               
                # Store the set types in the dictionary
                adjective_set[set_type] = dict()

                # Pull out all runs associated with the set
                set_runs = [_r for _r in children_set.__iter__()]

                for _objectRun in set_runs: 
                #import pdb; pdb.set_trace()
                
                    # Pull out tap information
                    tap_object = PullDataFromRun(_objectRun, BoltPR2MotionObj.TAP)
                    utilities.normalize_data(tap_object, discard_raw_flag)

                    # Pull out squeeze information
                    squeeze_object = PullDataFromRun(_objectRun, BoltPR2MotionObj.SQUEEZE)
                    utilities.normalize_data(squeeze_object, discard_raw_flag)

                    # Pull out hold information
                    hold_object = PullDataFromRun(_objectRun, BoltPR2MotionObj.THERMAL_HOLD) 
                    utilities.normalize_data(hold_object, discard_raw_flag)

                    # Pull out slide fast information
                    slide_fast_object = PullDataFromRun(_objectRun, BoltPR2MotionObj.SLIDE_FAST)
                    utilities.normalize_data(slide_fast_object, discard_raw_flag)

                    # Pull out slide slow information
                    slide_slow_object = PullDataFromRun(_objectRun, BoltPR2MotionObj.SLIDE)
                    utilities.normalize_data(slide_slow_object, discard_raw_flag)
             
                    if 'tap' not in adjective_set[set_type]:
                        adjective_set[set_type]['tap'] = list()
                        adjective_set[set_type]['squeeze'] = list()
                        adjective_set[set_type]['thermal_hold'] = list()
                        adjective_set[set_type]['slide'] = list()
                        adjective_set[set_type]['slide_fast'] = list()

                    adjective_set[set_type]['tap'].append(tap_object)
                    adjective_set[set_type]['squeeze'].append(squeeze_object)
                    adjective_set[set_type]['thermal_hold'].append(hold_object)
                    adjective_set[set_type]['slide'].append(slide_fast_object)
                    adjective_set[set_type]['slide_fast'].append(slide_slow_object)

            if (save_to_file):
                
                pca_dict = None
                for set_type in set_lists:
                    file_name ='lorenzo_data/'+output_filename+'_'+adj+'_'+set_type+'.pkl'
                    if set_type == 'train':
                        pca_dict = build_feature_objects(adjective_set[set_type], file_name, adjective_filename, pca_dict)
                else:
                    build_feature_objects(adjective_set[set_type], file_name, adjective_filename, pca_dict)
 
                #file_ptr = open('lorenzo_data/'+output_filename+'_'+adj+'_'+set_type+'.pkl', "w")
                #cPickle.dump(adjective_set[set_type], file_ptr, cPickle.HIGHEST_PROTOCOL)
                #file_ptr.close()

        # Store the adjective away
        #adjective_data_store[adj] = adjective_set
    
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

