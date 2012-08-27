#!/usr/bin/env python
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
from numpy import * 

# Class to store an entire run 
class BoltFeatureObj():

    DISABLED = 0 
    THERMAL_HOLD = 1 
    SLIDE = 2 
    SQUEEZE = 3 
    TAP = 5 
    DONE = 4 
    SLIDE_FAST = 6 
    CENTER_GRIPPER = 7 

    RIGHT = 0 
    LEFT = 1 

    def __init__(self):

        # Store features by channel
        self.pdc_rise_count = []
        self.pdc_max = []  
        self.pdc_area = []

        self.pac_energy = []
        self.pac_sc = []
        self.pac_sv = []
        self.pac_ss = []
        self.pac_sk = []

        self.tac_area = []
    	self.tdc_exp_fit = []

        # Store information about object
        self.state = self.DISABLED
        self.detailed_state = []

        # Store information about run motion came from
        self.name = ""
        self.run_number = 0
        self.object_id = 0

        # Store labels
        self.labels = None

