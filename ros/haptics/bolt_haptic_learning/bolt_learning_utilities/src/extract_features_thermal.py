#!/usr/bin/env python
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import numpy as np

from bolt_pr2_motion_obj import BoltPR2MotionObj


#For thermal_features

from scipy.integrate import trapz
import scipy.optimize

from pylab import *
import matplotlib as plt


def rindex(lis, item):
    for i in range(len(lis)-1, -1, -1):
        if item == lis[i]:
            return i
    raise ValueError("rindex(lis, item): item not in lis")

def fit_func(t, p):
    return p[0] + p[1]*np.exp(-t/p[2])

def erf(p, tdc_norm, t):
    return sum((tdc_norm-fit_func(t,p))**2)

def thermal_features(tdc_norm, tac_norm, controller_state, controller_state_detail):
    """
    Given one finger's array of tac_normalized this function will calculate the:

    INPUTS: tac_normalized - a normalized n x 1 numpy array of ac temperatures for one movement
            controller_state - an integer value signalling the type of movement
            controller_state_detail - an n x 1 array of strings detailing the
                                      current detailed state at 100 Hz

    OUTPUTS: (tac_area)
             the area under TAC curves for each state in one trial for one finger

    """
    #Choose sub-states for analysis based on controller state
    k = []
    try:
        if controller_state is BoltPR2MotionObj.THERMAL_HOLD:
            controller_state_str = "THERMAL_HOLD"
            k.append( (controller_state_detail.index('HOLD_FOR_10_SECONDS')+200) )
            k.append((rindex(controller_state_detail,'HOLD_FOR_10_SECONDS')+1))
        elif controller_state is BoltPR2MotionObj.SLIDE:
            controller_state_str = "SLIDE"
            k.append((controller_state_detail.index('SLIDE_5CM')+300))
            k.append((rindex(controller_state_detail,'SLIDE_5CM')-100))
        elif controller_state is BoltPR2MotionObj.SQUEEZE:
            controller_state_str = "SQUEEZE"
            k.append((controller_state_detail.index('SQUEEZE_SET_PRESSURE_SLOW')+150))
            k.append((rindex(controller_state_detail,'SQUEEZE_SET_PRESSURE_SLOW')-150))
       
	elif controller_state is BoltPR2MotionObj.TAP:
            controller_state_str = "TAP"
            k.append(controller_state_detail.index('MOVE_GRIPPER_FAST_CLOSE'))
            k.append((rindex(controller_state_detail,'OPEN_GRIPPER_BY_2CM_FAST')+1))

        elif controller_state is BoltPR2MotionObj.SLIDE_FAST:
            controller_state_str = "SLIDE_FAST"
            k.append(controller_state_detail.index('MOVE_DOWN_5CM'))
            k.append((rindex(controller_state_detail,'MOVE_DOWN_5CM')+1))
        else:
            rospy.logerr('Bad Controller State with state %d' % controller_state)
    except:
        
        rospy.logerr('Detailed Controller State not found with state %d' % controller_state)
        import pdb; pdb.set_trace()
    
    # Calculate area under TAC for each phase

    tac_area = trapz( tac_norm[ k[0]:k[1] ] )    

	

    # Exponential fits for TDC 

    tdc_norm = tdc_norm[ (k[0]):k[1] ]

    t = np.arange(1.,len(tdc_norm)+1)
    p0 = [np.mean(tdc_norm), np.max(tdc_norm)-np.min(tdc_norm), (np.max(t)-np.min(t))/2]
    
    popt = scipy.optimize.fmin(erf,p0,args=(tdc_norm, t),xtol=1e-8, disp=0)
   
    dummy_obj = BoltPR2MotionObj()
    tdc_fit = fit_func(tdc_norm,popt)
    ''' 
    print "Motion %s, TAC_Area %f, POPT (%f,%f,%f)" % (dummy_obj.state_string[controller_state], tac_area, popt[0], popt[1], popt[2])
	

    figure(controller_state)
    plot(t,tdc_norm,'bo',t,fit_func(t,popt),'r')
    grid()
    xlabel('Sample number')
    ylabel('TDC')
    title('%s' %(controller_state_str))
    '''

    return (tac_area,popt)





