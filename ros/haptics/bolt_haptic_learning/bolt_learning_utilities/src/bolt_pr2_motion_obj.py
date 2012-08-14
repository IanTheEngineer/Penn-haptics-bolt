#!/usr/bin/env python
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import numpy as np 

# Class to store an entire run 
class BoltPR2MotionObj(object):

    def __init__(self):

        # 2 finger x 19 x n
        self.electrodes = []
        # 2 finger x n
        self.tdc = []
        self.tac = []
        self.pdc = []
        self.pac = []
       
        # Gripper
        # 1 x n
        self.gripper_velocity = []
        self.gripper_position = []
        self.gripper_effort = []

        # Accelerometer
        # 3 x n
        self.accelerometer = [] 

        # Store state
        self.state = 0
   
        # Maybe store raw electrodes
