#!/usr/bin/env python
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
from numpy import * 

# Class to store an entire run 
class BoltPR2MotionObj(object):

    DISABLED = 0
    THERMAL_HOLD = 1
    SLIDE = 2
    SQUEEZE = 3
    TAP = 5
    DONE = 4
    SLIDE_FAST = 6
    CENTER_GRIPPER = 7

    def __init__(self):

        # 2 finger x n x 19 
        self.electrodes = array([]) 
        # 2 finger x n
        self.tdc = array([]) 
        self.tac = array([]) 
        self.pdc = array([]) 
        self.pac = array([]) 
       
        # Gripper
        # n x 1
        self.gripper_velocity = array([]) 
        self.gripper_position = array([]) 
        self.gripper_effort = array([]) 

        # Accelerometer
        # n x 3 
        self.accelerometer = array([]) 

        # Store state
        self.state = self.DISABLED
   
        # Maybe store raw electrodes
