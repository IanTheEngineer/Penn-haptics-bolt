#!/usr/bin/env python
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import numpy as np
from sklearn.decomposition import PCA
from scipy import optimize
import pylab

# Curve-fitting function
def electrode_poly(t, p):
    return p[0] + p[1]*t + p[2]*t**2 + p[3]*t**3 + p[4]*t**4 + p[5]*t**5 + p[6]*t**6

# Error function
def erf(p, electrode, t):
    return sum((electrode-electrode_poly(t,p))**2)

# Function to extract features from electrode data of a BoltPR2MotionObj
def electrode_features(electrodes, pca, controller_state):
    """
    INPUTS: electrodes - normalized electrodes vector from BoltPR2MotionObj
    
    OUTPUTS:

    """
    
    # Apply dimensionality reduction on electrode data
    eigen_electrodes = pca.fit_transform(electrodes)
    
    # Fitting a polynomial to the transormed data
    polyfit = []
    eigen_electrodes = np.transpose(eigen_electrodes)
    t = np.arange(1,np.size(eigen_electrodes,1)+1)
    for comp in range(0,2):
        p0 = [0, 0, 0, 0, 0, 0, 0]
        p_opt = optimize.fmin(erf, p0, args = (eigen_electrodes[comp], t), disp=0)
        polyfit = np.concatenate((polyfit,p_opt),1)

        """
        pylab.figure(comp)
        pylab.plot(t,eigen_electrodes[comp])
        pylab.hold(True)
        pylab.plot(t,electrode_poly(t, p_opt[(7*comp):((7*(comp+1))-1)]))
        pylab.grid(True)
        pylab.show()
        """

    return (polyfit)

