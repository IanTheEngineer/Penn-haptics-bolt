#!/usr/bin/env python
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import numpy as np
import cPickle
import convert_h5_dataset_to_bolt_obj as converth5
import process_adjective_labels

# Convert h5 files to BolPR2MotionObj data
def convertH5ToBoltObjFile(input_file, output_file, save_to_file):
    """
    Convert data in h5 file into a single pkl file that 
    stores the information in a BoltPR2MotionObj.  
    The runs are segmented by motion and stored in a dictionary.

    Usage: convertH5ToBoltObjFile(input_file, output_file, save_to_file)
   
    Note: The function returns a dictionary that can also be saved directly
          to a pickle file.  However, if save_to_file is False, it will not
          be saved and it is up to the user to store the returned object
 
    input_file - .h5
    output_file - .pkl
    save_to_file - boolean

    Output structure:
        
        Dictionary -> "squeeze", "thermal_hold", "tap", "slide",
                      "slow_slide"

        "squeeze" -> list(BoltPR2MotionObjs)

    """

    # Call the function
    loaded_data = converth5.load_data(input_file, output_file, save_to_file)

    return loaded_data


# Load bolt obj pickle file
def loadBoltObjFile(input_file):
    """
    Takes a path to a pkl file, loads, and returns the dictionary.  See
    convertH5ToBoltObjFile for the data structure returned
    """
    # File checking
    if not input_file.endswith(".pkl"):
        raise Exception("Input file: %s is not a pkl file.\n Please pass in pkl file with extention .pkl" % input_file)
    
    # Load the file 
    file_ptr = open(input_file, "r")
    loaded_data = cPickle.load(file_ptr)

    return loaded_data



# Insert adjective labels 
def insertAdjectiveLabels(input_boltPR2Motion, output_file, adjective_file, save_to_file):
    """
    Adjective label file is given as a pytable in h5 data

    Given either a path to a pkl file or the direct dictionary of 
    boltPR2MotionObj, will insert the labels
    
    if input_boltPR2Motion is a file, the function will first load the
    file before inserting adjectives

    """
    
    if isinstance(input_boltPR2Motion, str):
        data = process_adjective_labels.load_files(input_boltPR2Motion)
    else:
        data = input_boltPR2Motion

    data_with_adjectives = process_adjective_labels.populate_BoltPR2MotionObj(data, output_file, adjective_file, save_to_file)

    return data_with_adjectives



# Normalizes the given bolt_obj.  Works directly on the object
def normalize_data(bolt_obj, discard_raw_flag = True):
    """ 
    Given a BOLTPR2MotionObj 
    Normalize the data
        - Takes the mean and subtract 
        - Adds a pac_flat field

    Usage: normalize(bolt_obj, discard_flag)
        
        Discard flag is optional - default is True
        Discard flag set False will keep the original raw
        data, which makes the data twice as large
    """
    
    num_fingers = len(bolt_obj.electrodes)

    # For each finger normalize and store
    for finger in xrange(num_fingers):
        # Electrodes
        electrodes = bolt_obj.electrodes[finger]
        electrodes_mean = bolt_obj.electrodes_mean[finger]
        bolt_obj.electrodes_normalized.append(-(electrodes - np.mean(electrodes_mean, axis = 0)))

        # PDC
        pdc = bolt_obj.pdc[finger]
        pdc_mean = bolt_obj.pdc_mean[finger]
        bolt_obj.pdc_normalized.append(pdc - np.mean(pdc_mean))

        # TDC
        tdc = bolt_obj.tdc[finger]
        tdc_mean = bolt_obj.tdc_mean[finger]
        bolt_obj.tdc_normalized.append(tdc - np.mean(tdc_mean))

        # TAC
        tac = bolt_obj.tac[finger]
        tac_mean = bolt_obj.tac_mean[finger]
        bolt_obj.tac_normalized.append(-(tac - np.mean(tac_mean)))

        # PAC
        pac = bolt_obj.pac[finger]
        pac_mean = bolt_obj.pac_mean[finger]
        pac_norm = -(pac - np.mean(pac_mean, axis = 0)) 
        bolt_obj.pac_normalized.append(pac_norm)

        # Flatten PAC
        bolt_obj.pac_flat.append( pac.reshape(1,len(pac)*22)[0])
        bolt_obj.pac_flat_normalized.append( pac_norm.reshape(1, len(pac_norm)*22)[0])

    if discard_raw_flag:
        # Clear out raw values - comment out later if they want to be stored
        # Will double the amount of data stored
        del bolt_obj.pdc[:]
        del bolt_obj.electrodes[:]
        del bolt_obj.pac[:]
        del bolt_obj.tdc[:]
        del bolt_obj.tac[:]

    
        
     
