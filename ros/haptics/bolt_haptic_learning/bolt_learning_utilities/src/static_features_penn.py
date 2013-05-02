import numpy as np
import scipy.optimize
from scipy.signal import lfilter
from scipy.fftpack import fft
from scipy.signal import filtfilt
from collections import defaultdict

from sklearn.decomposition import PCA

# import objects to store data
from bolt_pr2_motion_obj import BoltPR2MotionObj
from static_feature_obj import StaticFeatureObj

static_features = ["pdc_rise_count", "pdc_area", "pdc_max", "pac_energy", "pac_sc", "pac_sv", "pac_ss", "pac_sk", "tac_area", "tdc_exp_fit", "gripper_min", "gripper_mean", "transform_distance", "electrode_polyfit"]


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def pdc_features(pdc_data, pdc_mean):
    """Area under the curve and max"""

    pdc_data = np.array(pdc_data)
    pdc_data = pdc_data - np.mean(pdc_mean, axis = 0)
    
    area = np.trapz(pdc_data, axis = 0)
    pdc_max = np.max(pdc_data, axis = 0)
    
    pdc_rise_count = []
    for finger in (0,1):
        filtered_pdc = smooth(pdc_data[:,finger],
                              window_len=50)
        pdc_rise_count.append(np.max(np.diff(filtered_pdc)))        
        
    return np.hstack((area, pdc_max, pdc_rise_count))

def pac_features(pac, pac_mean):
    """
    Given one finger's array of pac_flat this function will process textures
    and pull out specific features that are defined below:

    INPUTS: pac_flat - a normalized 22 x n x 1 numpy array of ac pressures for one movement

    OUTPUTS: (total_energy, spectral_moments)
             total_energy - an integer: the integral of the spectrum of vibration
                                        with respect to frequency
             spectral_moments - a tuple: (SC, SV, SS, SK)
                         aka.   (centroid, variance, skewness, excess kurtosis)

    """    
    
    def single_finger_features(pac_flat):
        #normalize
        texture =  pac_flat
    
        # Filter the AC pressure with a 20-700 Hz band-pass FIR filter
        sample_freq = 2200 # [Hz]
        filter_order = 66
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
        #total = 0
        #for elem in filt_texture.tolist():
        #total = total + elem
        #new_mean = total / len(filt_texture)
        new_mean = np.mean(filt_texture)
        
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
        total_energy = np.trapz(spectrum, freq) #/ L # is divided by length what we want for comparing different movements?
    
        # Spectral Moments - centroid, variance, skewness, excess kurtosis
        SC = sum(spectrum*freq)/sum(spectrum) 
        SV = sum(spectrum*(freq-SC)**2)/sum(spectrum)
        SS = (sum(spectrum*(freq-SC)**3)/sum(spectrum))/(SV**(3.0/2.0))
        SK = (sum(spectrum * (freq-SC)**4)/sum(spectrum))/(SV**2) - 3
    
        spectral_moments = (SC, SV, SS, SK)
    
        return (total_energy,) + spectral_moments
   
    pac_mean_cal = np.mean(pac_mean, axis = 1)
    finger0_pac = np.array(pac[0]) - pac_mean_cal[0]
    finger1_pac = np.array(pac[1]) - pac_mean_cal[1]
    #pac_flat = np.array(pac) - np.mean(pac_mean, axis = 1)    
    #finger_0 = pac_flat[:,:22].flatten()
    #finger_1 = pac_flat[:,22:].flatten()    
    finger_0 = finger0_pac.flatten()
    finger_1 = finger1_pac.flatten()   
 
    return single_finger_features(finger_0) + single_finger_features(finger_1)

def tac_features(tac, tac_mean):
    """
    Given one finger's array of tac_normalized this function will calculate the:

    INPUTS: tac_normalized - a normalized n x 1 numpy array of ac temperatures for one movement
            controller_state - an integer value signalling the type of movement
            controller_state_detail - an n x 1 array of strings detailing the
                                      current detailed state at 100 Hz

    OUTPUTS: (tac_area)
             the area under TAC curves for each state in one trial for one finger

    """
    
    tac_norm = np.array(tac) - np.mean(tac_mean, axis=0)
    tac_area = np.trapz( tac_norm, axis=0)    

    return tac_area

def tdc_features(tdc, tdc_mean):
    """
    Given one finger's array of tac_normalized this function will calculate the:

    INPUTS: tac_normalized - a normalized n x 1 numpy array of ac temperatures for one movement
            controller_state - an integer value signalling the type of movement
            controller_state_detail - an n x 1 array of strings detailing the
                                      current detailed state at 100 Hz

    OUTPUTS: (tac_area)
             the area under TAC curves for each state in one trial for one finger

    """
   
    def rindex(lis, item):
        for i in range(len(lis)-1, -1, -1):
            if item == lis[i]:
                return i
        raise ValueError("rindex(lis, item): item not in lis")

    def fit_func(t, p):
        return p[0] + p[1]*np.exp(-t/p[2])

    def erf(p, tdc_n, t):
        return sum((tdc_n-fit_func(t,p))**2)
    
    # Exponential fits for TDC 
    tdc_norm_all = np.array(tdc) - np.mean(tdc_mean, axis=0)
    
    final_fits = []

    for finger in range(tdc_norm_all.shape[1]): 
        tdc_norm = tdc_norm_all[:,finger] 
        t = np.arange(1.,len(tdc_norm)+1)
        p0 = [np.mean(tdc_norm), np.max(tdc_norm)-np.min(tdc_norm), (np.max(t)-np.min(t))/2]

        popt = scipy.optimize.fmin(erf,p0,args=(tdc_norm, t),xtol=1e-8, disp=0)

        tdc_fit = fit_func(tdc_norm,popt)
        final_fits.append(popt[2])

    return np.hstack(final_fits)


# Function to extract features from electrode data of a BoltPR2MotionObj
def electrodes_features(electrodes, mean_electrodes, pca = None):
    """
    INPUTS: electrodes - normalized electrodes vector from BoltPR2MotionObj
    
    OUTPUTS: polyfit - coefficients of optimized polynomial fit for electrode principal components
    """    
    
    # Curve-fitting function
    def electrode_poly(t, p):
        return p[0] + p[1]*t + p[2]*t**2 + p[3]*t**3 + p[4]*t**4 + p[5]*t**5
    
    # Error function
    def erf(p, electrode, t):
        return sum((electrode-electrode_poly(t,p))**2)    
    
    
    # Select desired segment for fitting
    mean_electrodes = np.hstack(mean_electrodes)
    electrodes = np.hstack(electrodes)
    electrodes = np.array(electrodes) - np.mean(mean_electrodes, axis=0)
    
    if pca is None:
        pca = PCA(4)
        pca.fit(electrodes)    

    # Apply dimensionality reduction on electrode data
    eigen_electrodes = pca.transform(electrodes)
   
    # Fitting a polynomial to the transormed data
    polyfit = []

    eigen_electrodes = np.transpose(eigen_electrodes)
    t = np.arange(1,np.size(eigen_electrodes,1)+1)
    
    for comp in range(0,4):
        p0 = [eigen_electrodes[comp][0], 0, 0, 0, 0, 0]
        p_opt = scipy.optimize.fmin(erf, p0, args = (eigen_electrodes[comp], t), xtol=1e-8, disp=0)
        polyfit = np.concatenate((polyfit,p_opt),1)
    
    return polyfit

def gripper_features(gripper_position):
    end_gripper = min(gripper_position)
    mean_gripper = np.mean(gripper_position)

    return end_gripper, mean_gripper

def transform_features(frame_transform):
    num_raw = frame_transform.shape[0]
    pick = np.array([2]*num_raw)
    height = frame_transform[np.arange(num_raw),pick]
    
    height_min = min(height.tolist())
    distance = height.tolist()[0] - height_min

    return distance



def get_all_features(sensors_dict):
    """Gets all the features from a dictionary with the data organized per sensor.
    Sensors are strings.
    
    Parameters:
    sensors_dict: a dictionary with keys; (pac, pdc, tac, electrodes) and values
    the corresponing arrays. Note that 2 fingers are assumed (organized per column).
    """
    
    pdc = pdc_features(sensors_dict['pdc'])
    pac = pac_features(sensors_dict['pac'])
    tac = tac_features(sensors_dict['tac'])
    electrodes = electrodes_features(sensors_dict['electrodes'])
    
    return np.hstack( (pdc, pac, tac, electrodes) )


def extract_static_features(bolt_obj, norm_obj):

    features = StaticFeatureObj()
    feature_array = []
 
    # PDC
    pdc_features_arr = pdc_features(np.array(bolt_obj.pdc).T, np.array(norm_obj.pdc).T)
    features.pdc_area = pdc_features_arr[0:2]
    features.pdc_max = pdc_features_arr[2:4]
    features.pdc_rise_count = pdc_features_arr[4:6]
    feature_array.append(pdc_features_arr)

    # PAC
    pac_features_arr = pac_features(bolt_obj.pac, norm_obj.pac)
    features.pac_energy = np.array((pac_features_arr[0],pac_features_arr[5]))
    features.pac_sc= np.array((pac_features_arr[1],pac_features_arr[6]))
    features.pac_sv = np.array((pac_features_arr[2],pac_features_arr[7]))
    features.pac_ss= np.array((pac_features_arr[3],pac_features_arr[8]))
    features.pac_sk = np.array((pac_features_arr[4],pac_features_arr[9]))
    feature_array.append(pac_features_arr)
    
    # TAC
    tac_features_arr = tac_features(np.array(bolt_obj.tac).T, np.array(norm_obj.tac).T)
    features.tac_area = tac_features_arr
    feature_array.append(tac_features_arr)

    # TDC 
    tdc_features_arr = tdc_features(np.array(bolt_obj.tdc).T, np.array(norm_obj.tdc).T)
    features.tdc_exp_fit = tdc_features_arr
 
    # electrodes
    electrode_features_arr = electrodes_features(bolt_obj.electrodes, norm_obj.electrodes)
    polyfit = []
    polyfit.append(electrode_features_arr[0:12])
    polyfit.append(electrode_features_arr[12:24])
 
    features.electrode_polyfit = np.array(polyfit)
    feature_array.append(electrode_features_arr)

    # Gripper
    gripper_features_arr = gripper_features(bolt_obj.gripper_position)
    features.gripper_min = gripper_features_arr[0]
    features.gripper_mean = gripper_features_arr[1]
    feature_array.append(gripper_features_arr)
 
    # Transform 
    transform_features_arr = transform_features(bolt_obj.l_tool_frame_transform_trans)
    features.transform_distance = transform_features_arr
    feature_array.append(transform_features)

    return features, np.hstack((pdc_features_arr, pac_features_arr, tac_features_arr,tdc_features_arr, electrode_features_arr, gripper_features_arr, transform_features_arr))


def pull_detailed_state(bolt_obj, detailed_state):
    '''
    Given a bolt object populated using the test_master_thread
    will pull out the section of data in the detailed state
    and use it to populate a new bolt_obj
    '''
    
    # Given a bolt_object (bolt_learning_utils) returns the segmented
    # section based on the passed in detailed state
    detailed_bolt_obj = BoltPR2MotionObj()

    # Get the index of the array for the detailed state
    state_array = bolt_obj.detailed_state
    idx_segment = [item for item in range(len(state_array)) if state_array[item] == detailed_state]

    # for each finger pull out the segment and store
    # hardcode that there are two fingers
    for _finger in xrange(2):
        
        detailed_bolt_obj.electrodes.append(bolt_obj.electrodes[_finger][idx_segment])
        detailed_bolt_obj.tdc.append(bolt_obj.tdc[_finger][idx_segment])
        detailed_bolt_obj.tac.append(bolt_obj.tac[_finger][idx_segment])
        detailed_bolt_obj.pdc.append(bolt_obj.pdc[_finger][idx_segment])
        detailed_bolt_obj.pac.append(bolt_obj.pac[_finger][idx_segment])

    detailed_bolt_obj.gripper_position = bolt_obj.gripper_position[idx_segment]
    detailed_bolt_obj.l_tool_frame_transform_trans = bolt_obj.l_tool_frame_transform_trans[idx_segment]
    
    # store off data for HMM approach
    detailed_bolt_obj.hmm = defaultdict(dict)
    detailed_bolt_obj.hmm['pac'] = np.hstack(np.array(detailed_bolt_obj.pac))
    detailed_bolt_obj.hmm['pdc'] = np.array(detailed_bolt_obj.pdc).T
    detailed_bolt_obj.hmm['electrodes'] = np.hstack(np.array(detailed_bolt_obj.electrodes))
    detailed_bolt_obj.hmm['tac'] = np.array(detailed_bolt_obj.tac).T 
   
    # store off detailed states
    detailed_bolt_obj.state = detailed_state;

    return detailed_bolt_obj


# Pull from the StaticFeatureObj and return a vector of features and labels
def createFeatureVector(static_feature_obj):
    """ 
    Given a BoltFeatureObj and a list of features to pull, returns
    the features and labels in vector form for all adjectives

    feature_list: expects a list of strings containing the name of the
                  features to extract
                  
                  Ex. ["max_pdc", "centroid"]

    Returns: Vector of features (1 x feature length)
             Features - in order of feature_list

             Ex. max_pdc = 10, centroid = [14, 10]

             returned numpy array: [10 14 10]

    """
    # Store the features with their adjectives 
    all_feature_vector = list()

    # Pull out the features from the object
    for feature in static_features:
    
        feature_vector = eval('static_feature_obj.'+ feature)
        # Make sure it is in list form 
        if isinstance(feature_vector, np.ndarray):
            feature_vector = feature_vector.tolist()

        if isinstance(feature_vector, float):
            feature_vector = [feature_vector]

        if feature == "electrode_polyfit":
            for idx in range(0,len(feature_vector[0])):
                all_feature_vector += [feature_vector[0][idx], feature_vector[1][idx]]
        else:
            all_feature_vector += feature_vector

    return np.array(all_feature_vector)
