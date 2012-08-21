#!/usr/bin/env python
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import numpy as np
import sklearn.decomposition

from bolt_pr2_motion_obj import BoltPR2MotionObj

#For texture_features
from scipy.signal import lfilter
from scipy.fftpack import fft
from scipy.signal import get_window
from scipy.signal import filtfilt
from scipy.integrate import trapz

from extract_features_thermal import thermal_features


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

    num_fingers = len(bolt_obj.electrodes_normalized)
    # Loop through each finger and store as a list
    for finger in xrange(num_fingers):
        thermal_features(bolt_obj.tdc_normalized[finger],bolt_obj.tac_normalized[finger], bolt_obj.state, bolt_obj.detailed_state)
        texture_features(bolt_obj.pac_flat_normalized[finger], bolt_obj.state, bolt_obj.detailed_state)
        '''#texture_features(bolt_obj.pac_flat[finger], bolt_obj.state, bolt_obj.detailed_state)
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
        transf_finger = pca.tranform(motion.electrodes_normalized[finger])'''



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


def rindex(lis, item):
    for i in range(len(lis)-1, -1, -1):
        if item == lis[i]:
            return i
    raise ValueError("rindex(lis, item): item not in lis")

#from pylab import *
#import matplotlib as plt
def texture_features( pac_flat, controller_state, controller_state_detail):
    """
    Given one finger's array of pac_flat this function will process textures
    and pull out specific features that are defined below:

    INPUTS: pac_flat - a normalized 22 x n x 1 numpy array of ac pressures for one movement
            controller_state - an integer value signalling the type of movement
            controller_state_detail - an n x 1 array of strings detailing the
                                      current detailed state at 100 Hz

    OUTPUTS: (total_energy, spectral_moments)
             total_energy - an integer: the integral of the spectrum of vibration
                                        with respect to frequency
             spectral_moments - a tuple: (SC, SV, SS, SK)
                         aka.   (centroid, variance, skewness, excess kurtosis)

    """
    #Choose sub-states for analysis based on controller state
    k = []
    try:
        if controller_state is BoltPR2MotionObj.THERMAL_HOLD:
            controller_state_str = "THERMAL_HOLD"
            k.append(22*controller_state_detail.index('HOLD_FOR_10_SECONDS'))
            k.append(22*(rindex(controller_state_detail,'OPEN_GRIPPER_BY_2CM_FAST')+1))
        elif controller_state is BoltPR2MotionObj.SLIDE:
            controller_state_str = "SLIDE"
            k.append(22*controller_state_detail.index('SLIDE_5CM'))
            k.append(22*(rindex(controller_state_detail,'SLIDE_5CM')+1))
        elif controller_state is BoltPR2MotionObj.SQUEEZE:
            controller_state_str = "SQUEEZE"
            k.append(22*controller_state_detail.index('SQUEEZE_SET_PRESSURE_SLOW'))
            k.append(22*(rindex(controller_state_detail,'SQUEEZE_SET_PRESSURE_SLOW')+1))
        elif controller_state is BoltPR2MotionObj.TAP:
            controller_state_str = "TAP"
            k.append(22*controller_state_detail.index('MOVE_GRIPPER_FAST_CLOSE'))
            k.append(22*(rindex(controller_state_detail,'OPEN_GRIPPER_BY_2CM_FAST')+1))
        elif controller_state is BoltPR2MotionObj.SLIDE_FAST:
            controller_state_str = "SLIDE_FAST"
            k.append(22*controller_state_detail.index('MOVE_DOWN_5CM'))
            k.append(22*(rindex(controller_state_detail,'MOVE_DOWN_5CM')+1))
        else:
            rospy.logerr('Bad Controller State in textureFeatures() with state %d' % controller_state)
    except:
        
        rospy.logerr('Detailed Controller State not found in textureFeatures() with state %d' % controller_state)
        import pdb; pdb.set_trace()
    # Select desired segment for analysis
    texture = pac_flat[k[0]:k[1]]

    # Filter the AC pressure with a 20-700 Hz band-pass FIR filter
    sample_freq = 2200; # [Hz]
    filter_order = 66;
    #from scipy_future_utils import firwin
    # b = firwin(filter_order,[2.0*20.0/sample_freq 2.0*700.0/sample_freq]);
    b = np.array([ -0.000734453028953, -0.000046996391356, -0.001575256697592, -0.001331737017349, -0.000008077427051,
                -0.002297899162801, -0.002780750937971, -0.000014231194022, -0.003355208565708, -0.005501069040353,
                -0.000217607048237, -0.004269008230790, -0.009746426984034, -0.000954778143579, -0.004360753419084,
                -0.015504199876395, -0.002765311287199, -0.002810845702213, -0.022444856580425, -0.006424172458608,
                0.001346313492583, -0.029939023426571, -0.013106851499087, 0.009483695851194, -0.037144235419593,
                -0.025129011478487, 0.024639968632464, -0.043146477287294, -0.049714697376541, 0.058786764136508,
                -0.047127552142491, -0.137377719108838, 0.271014481813890, 0.618653898229186, 0.271014481813890,
                -0.137377719108838, -0.047127552142491, 0.058786764136508, -0.049714697376541, -0.043146477287294,
                0.024639968632464, -0.025129011478487, -0.037144235419593, 0.009483695851194, -0.013106851499087,
                -0.029939023426571, 0.001346313492583, -0.006424172458608, -0.022444856580425, -0.002810845702213,
                -0.002765311287199, -0.015504199876395, -0.004360753419084, -0.000954778143579, -0.009746426984034,
                -0.004269008230790, -0.000217607048237, -0.005501069040353, -0.003355208565708, -0.000014231194022,
                -0.002780750937971, -0.002297899162801, -0.000008077427051, -0.001331737017349,-0.001575256697592,
                -0.000046996391356,-0.000734453028953])
    filt_texture = lfilter(b,1,texture)
    total = 0
    for elem in filt_texture.tolist():
        total = total + elem
    new_mean = total / len(filt_texture)
    # Remove signal bias after filtering
    filt_texture = filt_texture - new_mean

    # Calculate DFT and smooth it with a Bartlett-Hanning window
    L = float(len(filt_texture))
    texture_fft = fft(filt_texture,int(L))/L
    fft_freq = sample_freq/2.0*np.linspace(0,1, num=(round(L/2)+1) )
    #win = get_window('barthann',50)
    win =np.array([0, 0.012915713114513, 0.032019788779863, 0.057159387310553, 0.088082565902500,
                   0.124442415526911, 0.165802757166945, 0.211645303865110, 0.261378170980776, 0.314345594919613,
                   0.369838700753677, 0.427107141928132, 0.485371420930934, 0.543835688620429, 0.601700812046257,
                   0.658177496190301, 0.712499244169068, 0.763934943091322, 0.811800868911545, 0.855471913159863,
                   0.894391847205838, 0.928082455517205, 0.956151387945687, 0.978298602105584, 0.994321290061454,
                   0.994321290061454, 0.978298602105584, 0.956151387945687, 0.928082455517205, 0.894391847205838,
                   0.855471913159863, 0.811800868911545, 0.763934943091323, 0.712499244169069, 0.658177496190301,
                   0.601700812046257, 0.543835688620429, 0.485371420930934, 0.427107141928132, 0.369838700753677,
                   0.314345594919613, 0.261378170980776, 0.211645303865110, 0.165802757166945, 0.124442415526911,
                   0.088082565902500, 0.057159387310553, 0.032019788779863, 0.012915713114513, 0])

    win = win/sum(win)
    texture_fft_bhwin = filtfilt(win,[1],abs(texture_fft)**2)
    
    # Select smoothed spectra up to the max frequency that still contains data
    f_max = 100 # Hz
    k_max = (fft_freq>f_max).tolist().index(True)
    spectrum = texture_fft_bhwin[0:k_max]
    freq = fft_freq[0:k_max]

    # Total energy
    total_energy = trapz(spectrum, freq) #/ L # is divided by length what we want for comparing different movements?
   
    # Spectral Moments - centroid, variance, skewness, excess kurtosis
    SC = sum(spectrum*freq)/sum(spectrum) 
    SV = sum(spectrum*(freq-SC)**2)/sum(spectrum)
    SS = (sum(spectrum*(freq-SC)**3)/sum(spectrum))/(SV**(3.0/2.0))
    SK = (sum(spectrum * (freq-SC)**4)/sum(spectrum))/(SV**2) - 3

    spectral_moments = (SC, SV, SS, SK)
    #figure(controller_state)
    #plot(freq,spectrum)
    #grid()
    #ylabel('Spectrum')
    #xlabel('Freq (Hz)')
    #title('%s - Total_E=%.4f, SC=%.4f, SV=%.4f, SS=%.4f, SK=%.4f' %(controller_state_str,total_energy, SC, SV, SS, SK), fontsize=12)

    return (total_energy, spectral_moments)





