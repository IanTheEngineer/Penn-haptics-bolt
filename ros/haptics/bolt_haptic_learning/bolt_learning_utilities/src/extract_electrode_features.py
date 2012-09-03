#!/usr/bin/env python
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import numpy as np
from sklearn.decomposition import PCA
from scipy import optimize
from bolt_pr2_motion_obj import BoltPR2MotionObj
import pylab
import extract_features

# Curve-fitting function
def electrode_poly(t, p):
    return p[0] + p[1]*t + p[2]*t**2 + p[3]*t**3 + p[4]*t**4 + p[5]*t**5

# Error function
def erf(p, electrode, t):
    return sum((electrode-electrode_poly(t,p))**2)

# Function to extract features from electrode data of a BoltPR2MotionObj
def electrode_features(electrodes, pca, controller_state, controller_state_detail):
    """
    INPUTS: electrodes - normalized electrodes vector from BoltPR2MotionObj
    
    OUTPUTS: polyfit - coefficients of optimized polynomial fit for electrode principal components

    """
    #Choose sub-states for analysis based on controller state
    k = []
    try:
        if controller_state is BoltPR2MotionObj.THERMAL_HOLD:
            controller_state_str = "THERMAL_HOLD"
            k.append(controller_state_detail.index('CLOSE_GRIPPER_SLOW_TO_POSITION'))
            k.append((extract_features.rindex(controller_state_detail,'MOVE_UP_START_HEIGHT')+1))
        elif controller_state is BoltPR2MotionObj.SLIDE:
            controller_state_str = "SLIDE"
            k.append(controller_state_detail.index('SLIDE_5CM'))
            k.append((extract_features.rindex(controller_state_detail,'SLIDE_5CM')+1))
        elif controller_state is BoltPR2MotionObj.SQUEEZE:
            controller_state_str = "SQUEEZE"
            k.append(controller_state_detail.index('SQUEEZE_SET_PRESSURE_SLOW'))
            k.append((extract_features.rindex(controller_state_detail,'OPEN_GRIPPER_BY_2CM_FAST')+1))
        elif controller_state is BoltPR2MotionObj.TAP:
            controller_state_str = "TAP"
            k.append(controller_state_detail.index('OPEN_GRIPPER_BY_2CM_FAST'))
            k.append((extract_features.rindex(controller_state_detail,'OPEN_GRIPPER_BY_2CM_FAST')+1))
        elif controller_state is BoltPR2MotionObj.SLIDE_FAST:
            controller_state_str = "SLIDE_FAST"
            k.append(controller_state_detail.index('MOVE_DOWN_5CM'))
            k.append((extract_features.rindex(controller_state_detail,'MOVE_DOWN_5CM')+1))
        else:
            rospy.logerr('Bad Controller State in textureFeatures() with state %d' % controller_state)
    except:
    
        rospy.logerr('Detailed Controller State not found in electrod_features() with state %d' % controller_state)
        import pdb; pdb.set_trace()
    
    # Select desired segment for fitting
    electrodes = electrodes[k[0]:k[1]]

    # Apply dimensionality reduction on electrode data
    eigen_electrodes = pca.fit_transform(electrodes)
    
    # Fitting a polynomial to the transormed data
    polyfit = []
    eigen_electrodes = np.transpose(eigen_electrodes)
    t = np.arange(1,np.size(eigen_electrodes,1)+1)
    for comp in range(0,2):
        p0 = [eigen_electrodes[comp][0], 0, 0, 0, 0, 0]
        p_opt = optimize.fmin(erf, p0, args = (eigen_electrodes[comp], t), xtol=1e-8, disp=0)
        polyfit = np.concatenate((polyfit,p_opt),1)


        # Plotting for verification
        #pylab.figure(controller_state)
        #pylab.subplot(2, 2, 2*finger+comp+1)
        #pylab.plot(t,eigen_electrodes[comp])
        #pylab.hold(True)
        #pylab.plot(t,electrode_poly(t, p_opt))
        #pylab.grid(True)
        #pylab.xlabel('time')
        #pylab.ylabel('normalized voltage')
        #pylab.title("State: %s    Finger: %d    Component: %d" % (controller_state_str, finger, comp+1))
    
    return (polyfit)

