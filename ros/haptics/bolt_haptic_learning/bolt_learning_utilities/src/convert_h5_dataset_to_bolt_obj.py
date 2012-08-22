#!/usr/bin/env python

# Script to start loading data into pytables and convert into meaningful features
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import sys
import tables
import numpy as np
import cPickle
import bolt_learning_utilities as utilities

from bolt_pr2_motion_obj import BoltPR2MotionObj

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
    for _finger in xrange(one_run_pytable_ptr.biotacs._v_depth):

        # Create name to eval for finger
        finger_name_list = [] 
        finger_name_list.append('one_run_pytable_ptr.biotacs.finger_')
        finger_name_list.append(str(_finger))

        finger_name = ''.join(finger_name_list)

        # Electrodes
        one_set_electrode = eval(finger_name + '.electrodes[:]')
        one_motion_electrode = np.array(one_set_electrode[idx_segment])
        motion_object.electrodes.append(one_motion_electrode)
        motion_object.electrodes_mean.append(np.array(one_set_electrode[1:10, :]))

        # TDC
        one_set_tdc = eval(finger_name + '.tdc[:]')
        one_motion_tdc = np.array(one_set_tdc[idx_segment])
        motion_object.tdc.append(one_motion_tdc)
        motion_object.tdc_mean.append(np.array(one_set_tdc[1:10]))

        # TAC
        one_set_tac = eval(finger_name + '.tac[:]')
        one_motion_tac = np.array(one_set_tac[idx_segment])
        motion_object.tac.append(one_motion_tac)
        motion_object.tac_mean.append(np.array(one_set_tac[1:10]))

        # PDC
        one_set_pdc = eval(finger_name + '.pdc[:]')
        one_motion_pdc = np.array(one_set_pdc[idx_segment])
        motion_object.pdc.append(one_motion_pdc)
        motion_object.pdc_mean.append(np.array(one_set_pdc[1:10]))

        # PAC
        one_set_pac = eval(finger_name + '.pac[:]')
        one_motion_pac = np.array(one_set_pac[idx_segment])
        motion_object.pac.append(one_motion_pac)
        motion_object.pac_mean.append(np.array(one_set_pac[1:10, :]))

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


def load_data(input_filename, output_filename, save_to_file):

    if not input_filename.endswith(".h5"):
        raise Exception("Input file is %s \nPlease pass in a hdf5 data file" % input_filename)

    if save_to_file: 
        if not output_filename.endswith(".pkl"):
            output_filename = output_filename + '.pkl'

    # Load the data from an h5 file
    all_data = tables.openFile(input_filename)

    # Flag to indicate if raw values are stored when normalizing
    discard_raw_flag = True 

    # Create a storage container for data
    tap_runs = list()
    squeeze_runs = list()
    hold_runs = list() 
    slow_slide_runs = list()
    fast_slide_runs = list()

    # Create dictonary to store the final lists
    segmented_data = dict()

    # Keep counter of the number of runs done
    num_runs = 0

    # Pull pointers to only the file heads of the data structure
    all_runs_root = [_g for _g in all_data.walkGroups("/") if _g._v_depth == 1]

    # For each file extract the segments and data
    for _objectRun in all_runs_root:
        num_runs += 1
        print num_runs
        
        # Pull out tap information
        tap_object = PullDataFromRun(_objectRun, BoltPR2MotionObj.TAP)
        utilities.normalize_data(tap_object, discard_raw_flag)
        tap_runs.append(tap_object)

        # Pull out squeeze information
        squeeze_object = PullDataFromRun(_objectRun, BoltPR2MotionObj.SQUEEZE)
        utilities.normalize_data(squeeze_object, discard_raw_flag)
        squeeze_runs.append(squeeze_object)

        # Pull out hold information
        hold_object = PullDataFromRun(_objectRun, BoltPR2MotionObj.THERMAL_HOLD) 
        utilities.normalize_data(hold_object, discard_raw_flag)
        hold_runs.append(hold_object)

        # Pull out slide fast information
        slide_fast_object = PullDataFromRun(_objectRun, BoltPR2MotionObj.SLIDE_FAST)
        utilities.normalize_data(slide_fast_object, discard_raw_flag)
        fast_slide_runs.append(slide_fast_object)

        # Pull out slide slow information
        slide_slow_object = PullDataFromRun(_objectRun, BoltPR2MotionObj.SLIDE)
        utilities.normalize_data(slide_slow_object, discard_raw_flag)
        slow_slide_runs.append(slide_slow_object)
   

    # Store all of the lists into the dictionary
    segmented_data['tap'] = tap_runs
    segmented_data['squeeze'] = squeeze_runs
    segmented_data['thermal_hold'] = hold_runs
    segmented_data['slide'] = slow_slide_runs
    segmented_data['slide_fast'] = fast_slide_runs

    # if we want to save to file
    if (save_to_file):
        file_ptr = open(output_filename, "w")
        cPickle.dump(segmented_data, file_ptr, cPickle.HIGHEST_PROTOCOL)
        file_ptr.close()

    return segmented_data

def main():

    # Parse out the arguments passed in 
    if len(sys.argv) < 3:
        raise Exception("Usage: %s [input_file] [output_file]", sys.argv[0])

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    return input_filename, output_filename

if __name__== "__main__":
    input_file, output_file = main() 
    load_data(input_file, output_file, True)

