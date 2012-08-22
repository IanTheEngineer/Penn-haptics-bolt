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

    RIGHT = 0 
    LEFT = 1


    def __init__(self):
        self.state_string = {self.DISABLED:'Disabled',
                    self.THERMAL_HOLD:'Thermal Hold',
                    self.SLIDE:'Slide',
                    self.SQUEEZE:'Squeeze',
                    self.TAP:'Tap',
                    self.DONE:'Done',
                    self.SLIDE_FAST:'Slide Fast',
                    self.CENTER_GRIPPER:'Center Gripper'
                    }

        # For each finger
        # n x 19 
        self.electrodes = [] 
        # 1 x n
        self.tdc = [] 
        self.tac = [] 
        self.pdc = []  
        # n x 22 
        self.pac = [] 
        # n * 22
        self.pac_flat = [] 

        # Store first 10 values to use as means
        self.electrodes_mean = []
        self.pdc_mean = []
        self.pac_mean = []
        self.tdc_mean = []
        self.tac_mean = []
        self.pac_mean = []
        
        # Store normalized finger values
        self.electrodes_normalized = []
        self.pdc_normalized = []
        self.pac_normalized = []
        self.tdc_normalized = []
        self.tac_normalized = []
        self.pac_flat_normalized = []

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
        self.detailed_state = []
   
        # Store information about the run the motion came from
        self.name = ""
        self.run_number = 0 
        self.object_id = 0
  
        # rot - n x 4
        # trans - n x 3
        # Store gripper information
        self.l_tool_frame_transform_rot = []
        self.l_tool_frame_transform_trans = []

        # Store the labels in a dictionary
        self.labels = None 
        
