#!/usr/bin/env python
# Copyright (c) 2012, University of Pennsylvania
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of Pennsylvania nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#Author: Ian McMahon

'''Summary: This node gathers data from the BioTac Sensors, PR2 arm joints, 
gripper accelerometers, transforms, and BOLT controller states,
packages them nicely as a pickled class and at the end of every motion, 
sends that data out over the topic new_hadjective_motion_pickle for
classifiers to receive'''


#Import ROS essentials
import roslib; roslib.load_manifest('hadjective_test_pipe')
import rospy

#Import Python Libraries
import sys
import threading
import multiprocessing
from collections import defaultdict
import numpy as np
import cPickle
import time

#Import messages
from biotac_sensors.msg import BioTacHand
from pr2_gripper_accelerometer.msg import PR2GripperAccelerometerData
from pr2_arm_state_aggregator.msg import PR2ArmState
from std_msgs.msg import Int8, String

#Import Bolt Learning Utilities
from bolt_pr2_motion_obj import BoltPR2MotionObj

#Maybe a global lock on the publisher? We don't want pickle files colliding...
#NOTE: In practice every motion finishes > 0.2 seconds apart, so there's no collision issue.
def processMotion(task_queue, result_queue):
    #Start Publisher for Learning Algorithms
    name = multiprocessing.current_process().name
    pub = rospy.Publisher('new_hadjective_motion_pickle', String)
    print name, 'Starting at time %f' % time.time()
    #Grab the current motion from the queue
    current_motion = task_queue.get()
    # Convert the buffer received into BoltPR2MotionObj
    current_obj = current_motion.convertToBoltPR2MotionObj()
    #Pickle & Publish
    pickle_string = cPickle.dumps(current_obj, protocol=cPickle.HIGHEST_PROTOCOL)
    pub.publish(pickle_string)   
    #print pickle_string
    result_queue.put("Process %s Complete!" % name)
    print name, 'Finished at time %f' % time.time()

class BoltPR2MotionBuf(object):
    #A buffer meant solely for this file, to convert to and play nice with Vivian's BoltPR2MotionObj()
    #This is because: np.array() creation takes non-trivial time, and should only be 
    #computed once in each process that is about to publish a BoltPR2MotionObj() pkl file
    #Set up all the Constants
    DISABLED = BoltPR2MotionObj.DISABLED 
    THERMAL_HOLD = BoltPR2MotionObj.THERMAL_HOLD
    SLIDE = BoltPR2MotionObj.SLIDE 
    SQUEEZE = BoltPR2MotionObj.SQUEEZE 
    TAP = BoltPR2MotionObj.TAP 
    DONE = BoltPR2MotionObj.DONE 
    SLIDE_FAST = BoltPR2MotionObj.SLIDE_FAST 
    CENTER_GRIPPER = BoltPR2MotionObj.CENTER_GRIPPER 
    RIGHT = BoltPR2MotionObj.RIGHT 
    LEFT = BoltPR2MotionObj.LEFT

    def __init__(self,prev_state=None):
        #Initialize all lists / Dictionaries storing data to be empty
        self.electrodes = defaultdict(list)
        self.tdc = defaultdict(list)
        self.tac = defaultdict(list)
        self.pdc = defaultdict(list)
        self.pac = defaultdict(list)
        self.gripper_velocity = []
        self.gripper_position = []
        self.gripper_effort = []
        self.accelerometer = []
        self.detailed_state = []
        self.l_tool_frame_transform_rot = []
        self.l_tool_frame_transform_trans = []
        
        if prev_state == None:
            self.state = BoltPR2MotionBuf.DISABLED
        else:
            self.state = prev_state

        self.electrodes_mean = defaultdict(list)
        self.pdc_mean = defaultdict(list)
        self.pac_mean = defaultdict(list)
        self.tdc_mean = defaultdict(list)
        self.tac_mean = defaultdict(list)

    def convertToBoltPR2MotionObj(self):
        #This is where the magic happens:
        #Construct the BoltPR2MontionObj() just the way that it is expected downstream
        new_obj = BoltPR2MotionObj()
        #Populate new object
        new_obj.electrodes = [np.array(self.electrodes[self.RIGHT]), np.array(self.electrodes[self.LEFT])]
        new_obj.tdc = [np.array(self.tdc[self.RIGHT]), np.array(self.tdc[self.LEFT])]
        new_obj.tac = [np.array(self.tac[self.RIGHT]), np.array(self.tac[self.LEFT])]
        new_obj.pdc = [np.array(self.pdc[self.RIGHT]), np.array(self.pdc[self.LEFT])]
        new_obj.pac = [np.array(self.pac[self.RIGHT]), np.array(self.pac[self.LEFT])]
        new_obj.gripper_velocity = np.array(self.gripper_velocity)
        new_obj.gripper_position = np.array(self.gripper_position)
        new_obj.gripper_effort = np.array(self.gripper_effort)
        #new_obj.accelerometer = np.array(self.accelerometer)
        new_obj.accelerometer = np.zeros(shape=(5,2))
        new_obj.l_tool_frame_transform_trans = np.array(self.l_tool_frame_transform_trans)
        new_obj.l_tool_frame_transform_rot = np.array(self.l_tool_frame_transform_rot)

        new_obj.detailed_state = self.detailed_state
        new_obj.electrodes_mean = [np.array(self.electrodes_mean[self.RIGHT]), np.array(self.electrodes_mean[self.LEFT])]
        new_obj.tdc_mean = [np.array(self.tdc_mean[self.RIGHT]), np.array(self.tdc_mean[self.LEFT])]
        new_obj.tac_mean = [np.array(self.tac_mean[self.RIGHT]), np.array(self.tac_mean[self.LEFT])]
        new_obj.pdc_mean = [np.array(self.pdc_mean[self.RIGHT]), np.array(self.pdc_mean[self.LEFT])]
        new_obj.pac_mean = [np.array(self.pac_mean[self.RIGHT]), np.array(self.pac_mean[self.LEFT])]

        new_obj.state = self.state
        #return populated object
        return new_obj
        

class LanguageTestMainThread:

    def __init__(self):
        #Initialize Node for ROS
        rospy.init_node('language_test_subscribers')
        rospy.loginfo('main language test thread initializing...')

        #Create buffer message to store all received data for this motion
        self.current_motion = BoltPR2MotionBuf()
        self.last_state = BoltPR2MotionBuf.DISABLED

        # Create empty temporary buffers to store only current
        self.gripper_velocity_buf = 0
        self.gripper_position_buf = 0
        self.gripper_effort_buf = 0
        self.accelerometer_buf = 0
        self.detailed_state_buf = ''
        self.l_tool_tf_trans_buf = (0.0,0.0,0.0)
        self.l_tool_tf_rot_buf = (0.0,0.0,0.0,0.0)

        # Downsampling the accelerometer (for now)
        self.accel_downsample_counter = 0

        # Create a list of all BioTac sensor start value means
        self.electrodes_mean_list = defaultdict(list)
        self.tdc_mean_list = defaultdict(list)
        self.tac_mean_list = defaultdict(list)
        self.pdc_mean_list = defaultdict(list)
        self.pac_mean_list = defaultdict(list)
        self.mean_count = 0
        self.valid_state_tuple = (BoltPR2MotionBuf.DISABLED, BoltPR2MotionBuf.THERMAL_HOLD, BoltPR2MotionBuf.SLIDE,
                                  BoltPR2MotionBuf.SQUEEZE, BoltPR2MotionBuf.TAP,
                                  BoltPR2MotionBuf.SLIDE_FAST, BoltPR2MotionBuf.DONE)

        #Create locks for the callbacks - they are all in threads of their own
        self.accel_lock = threading.Lock()
        self.state_lock = threading.Lock()

    def disabled_clear(self, prev_state):
        # If the state is disabled, call this to clear transformations and motion data
        self.current_motion = BoltPR2MotionBuf(prev_state)
        self.last_state = BoltPR2MotionBuf.DISABLED
        self.l_tool_tf_trans_buf = (0.0,0.0,0.0)
        self.l_tool_tf_rot_buf = (0.0,0.0,0.0,0.0)
 
    def reset_run(self):
        # Called between each complete run to start fresh
        self.current_motion = BoltPR2MotionBuf()
        self.last_state = BoltPR2MotionBuf.DISABLED
        self.mean_count = 0
        self.electrodes_mean_list = defaultdict(list)
        self.tdc_mean_list = defaultdict(list)
        self.tac_mean_list = defaultdict(list)
        self.pdc_mean_list = defaultdict(list)
        self.pac_mean_list = defaultdict(list)
        self.l_tool_tf_trans_buf = (0.0,0.0,0.0)
        self.l_tool_tf_rot_buf = (0.0,0.0,0.0,0.0)

    def clear_motion(self,prev_state):
        #Called in between each motion
        #Reset current_motion, but populate mean list
        self.current_motion = BoltPR2MotionBuf(prev_state)
        self.current_motion.electrodes_mean = self.electrodes_mean_list
        self.current_motion.pdc_mean = self.pdc_mean_list
        self.current_motion.pac_mean = self.pac_mean_list
        self.current_motion.tdc_mean = self.tdc_mean_list
        self.current_motion.tac_mean = self.tac_mean_list

    def start_listeners(self):
        #Start BioTac Subscriber
        rospy.Subscriber('biotac_pub', BioTacHand, self.biotacCallback,queue_size=50)
        #Start Accelerometer Subscriber
        rospy.Subscriber('/pr2_gripper_accelerometer/data', PR2GripperAccelerometerData, self.accelerometerCallback,queue_size=500)
        #Start Gripper Controller State Subscriber
        rospy.Subscriber('/simple_gripper_controller_state', Int8, self.gripperStateCallback, queue_size=50)
        #Start Detailed Gripper Controller State Subscriber
        rospy.Subscriber('/simple_gripper_controller_state_detailed', String, self.gripperDetailedCallback, queue_size=50)
        #Start TF Subscriber
        rospy.Subscriber('/pr2_arm_state', PR2ArmState, self.lArmCallback, queue_size=50)
        #Start Publisher for Learning Algorithms
        self.pub = rospy.Publisher('new_hadjective_motion_pickle', String)

    def lArmCallback(self, msg):
        # Store tool_frame transform
        bool_idx = [(value.child_frame_id == '/l_gripper_tool_frame') for value in msg.transforms]
        tool_frame_idx = bool_idx.index(True)
        # Store l_tool_frame information
        if msg.transforms[tool_frame_idx].transform_valid:
            l_tool_tf = msg.transforms[tool_frame_idx].transform
            # Store information
            self.l_tool_tf_trans_buf = (l_tool_tf.translation.x, l_tool_tf.translation.y, l_tool_tf.translation.z)
            self.l_tool_tf_rot_buf = (l_tool_tf.rotation.x, l_tool_tf.rotation.y, l_tool_tf.rotation.z, l_tool_tf.rotation.w)

    def accelerometerCallback(self, msg):
        #Downsample the Accelerometer
        self.accel_downsample_counter = self.accel_downsample_counter + 1    
        if not self.accel_downsample_counter % 5: # 1000Hz -> 200Hz which is 2*100Hz. Yay Nyquist! 
            self.accel_downsample_counter = 0
            self.accel_lock.acquire()
            # Store accelerometer
            self.accelerometer_buf = (msg.acc_x_raw, msg.acc_y_raw, msg.acc_z_raw)
            # Store gripper
            self.gripper_position_buf = msg.gripper_joint_position
            self.gripper_velocity_buf = msg.gripper_joint_velocity
            self.gripper_effort_buf = msg.gripper_joint_effort
            self.accel_lock.release()

    def biotacCallback(self, msg):
        ZERO_TIME = 100 
        NUM_MEAN_VALS = 10

        #Wait for ZERO_TIME callbacks to pass, 
        #then store off the mean of each channel for every BioTac sensors for NUM_MEAN_VALS messages
        if len(self.tdc_mean_list[0]) < NUM_MEAN_VALS and self.mean_count > ZERO_TIME:
            num_fingers = len(msg.bt_data)
            for finger_index in xrange(num_fingers):
                self.electrodes_mean_list[finger_index].append( msg.bt_data[finger_index].electrode_data)
                self.tdc_mean_list[finger_index].append( msg.bt_data[finger_index].tdc_data)
                self.tac_mean_list[finger_index].append( msg.bt_data[finger_index].tac_data)
                self.pdc_mean_list[finger_index].append( msg.bt_data[finger_index].pdc_data)
                self.pac_mean_list[finger_index].append( msg.bt_data[finger_index].pac_data)
                if len(self.tdc_mean_list[0]) == NUM_MEAN_VALS:
                    self.state_lock.acquire()
                    self.current_motion.electrodes_mean = self.electrodes_mean_list
                    self.current_motion.pdc_mean = self.pdc_mean_list
                    self.current_motion.pac_mean = self.pac_mean_list
                    self.current_motion.tdc_mean = self.tdc_mean_list
                    self.current_motion.tac_mean = self.tac_mean_list
                    self.state_lock.release()
        else:
            self.mean_count = self.mean_count + 1

        #Lock to prevent the state controller from updating                 
        self.state_lock.acquire()
        #If in a valid controller state, start storing off all buffered data
        if self.current_motion.state in self.valid_state_tuple:
            num_fingers = len(msg.bt_data)
            for finger_index in xrange(num_fingers):    
    
                self.current_motion.tdc[finger_index].append( msg.bt_data[finger_index].tdc_data)
                self.current_motion.tac[finger_index].append( msg.bt_data[finger_index].tac_data)
                self.current_motion.pdc[finger_index].append( msg.bt_data[finger_index].pdc_data)
                self.current_motion.pac[finger_index].append( msg.bt_data[finger_index].pac_data)
                self.current_motion.electrodes[finger_index].append( msg.bt_data[finger_index].electrode_data)
            
            self.current_motion.detailed_state.append(self.detailed_state_buf)
            self.current_motion.l_tool_frame_transform_trans.append(self.l_tool_tf_trans_buf)
            self.current_motion.l_tool_frame_transform_rot.append(self.l_tool_tf_rot_buf)
            #A lock is necessary here to ensure all accelerometer reading and gripper readings are simultaneous 
            self.accel_lock.acquire()
            self.current_motion.accelerometer.append(self.accelerometer_buf)
            self.current_motion.gripper_position.append(self.gripper_position_buf)
            self.current_motion.gripper_velocity.append(self.gripper_velocity_buf)
            self.current_motion.gripper_effort.append(self.gripper_effort_buf)
            self.accel_lock.release()
        self.state_lock.release()

    def gripperDetailedCallback(self, msg):
        #Store the current detailed state
        self.detailed_state_buf = msg.data

    def gripperStateCallback(self, msg):
        #Acquire the state lock and store the current state off
        self.state_lock.acquire()
        self.current_motion.state = msg.data
        self.state_lock.release()


def main(argv):

    # Establish communication queues
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_tasks = 0

    main_thread =  LanguageTestMainThread()
    main_thread.start_listeners()

    while not rospy.is_shutdown():
        #Acquire Lock
        main_thread.state_lock.acquire()

        #If the current state is valid and the last state is valid, and its a new state...
        if  main_thread.last_state in main_thread.valid_state_tuple and \
            main_thread.last_state != main_thread.current_motion.state:
            
            print 'The state is %s' % main_thread.last_state 
             
            #Store off next state to see if we're done
            next_state = main_thread.current_motion.state
            #Close up the current current_motion and send it to a thread
            main_thread.current_motion.state = main_thread.last_state
            #Store the next state as the last state to be used to see when a change occurs
            main_thread.last_state = next_state
            #Place current_motion in the queue
            tasks.put(main_thread.current_motion)
            #Spin up a new thread
            new_process = multiprocessing.Process(target=processMotion, args=(tasks,results))
            new_process.start()
            #Reset current_motion
            main_thread.clear_motion(next_state)
            num_tasks = num_tasks + 1

            #Check to see if the motions have finished
            if next_state is BoltPR2MotionBuf.DONE:
                main_thread.reset_run()
                rospy.loginfo("Done logging Hadjective Data for now!")

        elif main_thread.last_state is not main_thread.current_motion.state:
            #Simply update the last state
            main_thread.last_state = main_thread.current_motion.state

        #Release Lock
        main_thread.state_lock.release()
    
    #Clean up all those threads!
    tasks.close()
    tasks.join_thread()

if __name__ == '__main__':
  main(sys.argv[1:])
