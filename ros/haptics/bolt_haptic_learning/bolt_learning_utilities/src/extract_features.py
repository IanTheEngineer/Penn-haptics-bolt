#!/usr/bin/env python
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import numpy as np
import sklearn.decomposition

from bolt_pr2_motion_obj import BoltPR2MotionObj

# Functions to help extract features from a BoltPR2MotionObj
def extract_features(bolt_obj):
    """
    Given a BoltPR2MotionObj the function will process the streams
    and pull out specific features that are defined below:

    Histogram of peaks?
    Poly fit the lines?

    PDC
        - area under curve
        - max height of curve
        - fit line to equation?

    PAC
        - Bin into 4 equal parts and find power of signal
        - Central frequency
    
    Temperature (TAC + TDC) 
        - Area under the curve
        - Tau constant
        - Temperature Final

    Electrodes
        - PCA + polyfit

    Gripper Aperture
        - Gripper position
       
    Accelerometer
        - Z channel?

    """
    # PDC Features
    pdc_area = []
    pdc_max = []

    # PAC Features
    pac_power = []
    pac_central_frequency = []
    
    # Temperature features
    temperature_area = []
    temperature_tau = []
    temperature_final = []

    # Electrode features
    electrode_polyfit = []

    num_fingers = len(bolt_obj.electrodes)

    # Loop through each finger and store as a list
    for finger in xrange(num_fingers):
       
        # Compute pdc features 
        pdc_area.append(np.trapz(bolt_obj.pdc_normalized[finger])) 
        pdc_max.append(max(bolt_obj.pdc_normalized[finger]))

        # Compute pac features
        pac_square_split = np.array_split(np.square(bolt_obj.pac_flat_normalized[finger]), 4)
        pac_power.append(np.divide(np.sum(pac_square_split, axis = 1)), np.shape(pac_square_split[0])[0])
        
        pac_central_frequency.append()
       
        # Thermal features
        
       
        # Compute electrode features
        pca = sklearn.decomposition.PCA(n_components = 2, whiten = False)
        pca.fit(motion.electrodes_normalized[finger])
        transf_finger = pca.tranform(motion.electrodes_normalized[finger])



# Normalizes the given bolt_obj.  Works directly on the object
def normalize_data(bolt_obj, discard_raw_flag = True):
    """ 
    Given a BOLTPR2MotionObj 
    Normalize the data
        - Takes the mean and subtract 
        - Adds a pac_flat field
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

