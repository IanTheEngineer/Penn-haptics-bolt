#!/usr/bin/env python

import roslib; roslib.load_manifest('hadjective_test_pipe')
import rospy

import sys
import threading
import multiprocessing
from collections import defaultdict
import numpy as np

from bolt_pr2_motion_obj import BoltPR2MotionObj
from extract_features import *

from biotac_sensors.msg import BioTacHand
from pr2_gripper_accelerometer.msg import PR2GripperAccelerometerData
from std_msgs.msg import Int8, String

import matplotlib.pyplot as plt
import pickle

from pylab import *


def processMotion(task_queue, result_queue):
    name = multiprocessing.current_process().name
    print name, 'Starting'
    #Grab the current motion from the queue
    current_motion = task_queue.get()
    # Convert the buffer received into BoltPR2MotionObj
    current_obj = current_motion.convertToBoltPR2MotionObj()
    normalize_data(current_obj, False)
    #MACHINE LEARNING STUFF!!
   
    #plt.plot(current_obj.pac_normalized[0])
    #plt.show()

    #output_filename = '/home/imcmahon/data/'+name
    #file_ptr = open(output_filename, "w")
    #cPickle.dump(current_obj, file_ptr, cPickle.HIGHEST_PROTOCOL)
    #file_ptr.close()

    #fig = plt.figure()
    #ax1 = fig.add_subplot(2, 2, 1)
    #ax2 = fig.add_subplot(2, 2, 2)
    #ax2 = fig.add_subplot(2, 2, 3)
    #ax3 = fig.add_subplot(2, 2, 4)
    #operate to determine stuff
    answer = current_motion.state
 
    print name, ' received motion ', answer
    result_queue.put(current_obj)

class BoltPR2MotionBuf(object):
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

    def __init__(self):
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
        self.state = BoltPR2MotionBuf.DISABLED
        self.electrodes_mean = defaultdict(list)
        self.pdc_mean = defaultdict(list)
        self.pac_mean = defaultdict(list)
        self.tdc_mean = defaultdict(list)
        self.tac_mean = defaultdict(list)

    def convertToBoltPR2MotionObj(self):
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
        new_obj.accelerometer = np.array(self.accelerometer)
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
        rospy.init_node('language_test_subscribers')
        rospy.loginfo('main language test thread initializing...')
        self.current_motion = BoltPR2MotionBuf()
        self.last_state = BoltPR2MotionBuf.DISABLED
        # Create empty lists (temporary buffers) to store all data 
        self.gripper_velocity_buf = 0
        self.gripper_position_buf = 0
        self.gripper_effort_buf = 0
        self.accelerometer_buf = 0
        self.detailed_state_buf = ''
        #Create locks for the callbacks - they are all in threads of their own
        self.accel_lock = threading.Lock()
        #self.tf_lock = threading.Lock()
        self.state_lock = threading.Lock()
        #self.detailed_lock = threading.Lock()
        #self.thread_lock = threading.Lock()
        self.accel_downsample_counter = 0
        self.electrodes_mean_list = defaultdict(list)
        self.tdc_mean_list = defaultdict(list)
        self.tac_mean_list = defaultdict(list)
        self.pdc_mean_list = defaultdict(list)
        self.pac_mean_list = defaultdict(list)
        self.valid_state_tuple = (BoltPR2MotionBuf.THERMAL_HOLD, BoltPR2MotionBuf.SLIDE,
                                  BoltPR2MotionBuf.SQUEEZE, BoltPR2MotionBuf.TAP,
                                  BoltPR2MotionBuf.SLIDE_FAST, BoltPR2MotionBuf.DONE)

    def clear_motion(self):
        #Reset current_motion, but populate mean list
        self.current_motion = BoltPR2MotionBuf()
        #self.current_motion.state = current_state
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
        #Start Publisher for Learning Algorithms
        self.pub = rospy.Publisher('hadjective_motion_pickle', String)

    def accelerometerCallback(self, msg):
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

        if len(self.tdc_mean_list[0]) < 10:
            num_fingers = len(msg.bt_data)
            for finger_index in xrange(num_fingers):    
                self.electrodes_mean_list[finger_index].append( msg.bt_data[finger_index].electrode_data)
                self.tdc_mean_list[finger_index].append( msg.bt_data[finger_index].tdc_data)
                self.tac_mean_list[finger_index].append( msg.bt_data[finger_index].tac_data)
                self.pdc_mean_list[finger_index].append( msg.bt_data[finger_index].pdc_data)
                self.pac_mean_list[finger_index].append( msg.bt_data[finger_index].pac_data)
                if len(self.tdc_mean_list[0]) is 10:
                    self.state_lock.acquire()
                    self.current_motion.electrodes_mean = self.electrodes_mean_list
                    self.current_motion.pdc_mean = self.pdc_mean_list
                    self.current_motion.pac_mean = self.pac_mean_list
                    self.current_motion.tdc_mean = self.tdc_mean_list
                    self.current_motion.tac_mean = self.tac_mean_list
                    self.state_lock.release()
        #Lock to prevent the state controller from updating                 
        self.state_lock.acquire()
        if self.current_motion.state in self.valid_state_tuple:
            num_fingers = len(msg.bt_data)
            for finger_index in xrange(num_fingers):    
    
                self.current_motion.tdc[finger_index].append( msg.bt_data[finger_index].tdc_data)
                self.current_motion.tac[finger_index].append( msg.bt_data[finger_index].tac_data)
                self.current_motion.pdc[finger_index].append( msg.bt_data[finger_index].pdc_data)
                self.current_motion.pac[finger_index].append( msg.bt_data[finger_index].pac_data)
                self.current_motion.electrodes[finger_index].append( msg.bt_data[finger_index].electrode_data)
            
            self.current_motion.detailed_state.append(self.detailed_state_buf)
            #A lock is necessary here to ensure all accelerometer reading and gripper readings are simultaneous 
            self.accel_lock.acquire()
            self.current_motion.accelerometer.append(self.accelerometer_buf)
            self.current_motion.gripper_position.append(self.gripper_position_buf)
            self.current_motion.gripper_velocity.append(self.gripper_velocity_buf)
            self.current_motion.gripper_position.append(self.gripper_effort_buf)
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
        if  main_thread.current_motion.state in main_thread.valid_state_tuple and \
            main_thread.last_state in main_thread.valid_state_tuple and \
            main_thread.last_state is not main_thread.current_motion.state:
            #if main_thread.current_motion.state == 1:
            #import pdb; pdb.set_trace()
            #Store off next state to see if we're done
            next_state = main_thread.current_motion.state
            #Close up the current current_motion and send it to a thread
            main_thread.current_motion.state = main_thread.last_state
            #Store the next state as the last state to be used to see when a change occurs
            main_thread.last_state = next_state
            #Place current_motion in the que
            #main_thread.current_motion.convertToBoltPR2MotionObj()
            #if num_tasks is 1:
            #    current_bolt_pr2_motion_obj = main_thread.current_motion.convertToBoltPR2MotionObj()
            current_obj = main_thread.current_motion.convertToBoltPR2MotionObj()
            normalize_data(current_obj, False)
            #extract_features(current_obj)
            #Pickle & Publish
            pickle_string = pickle.dumps(current_obj, protocol=pickle.HIGHEST_PROTOCOL)
            #pickle_string = cPickle.dumps(current_obj, protocol=cPickle.HIGHEST_PROTOCOL)
            main_thread.pub.publish(String(pickle_string))    
            
            #tasks.put(main_thread.current_motion)
            #Reset current_motion
            main_thread.clear_motion()

            #Spin up a new thread
            #new_process = multiprocessing.Process(target=processMotion, args=(tasks,results))
            #new_process.start()
            #num_tasks = num_tasks + 1

            #Check to see if the motions have finished
            if next_state is BoltPR2MotionBuf.DONE:
                break
            #print 't ' , time.time() - start_time

        elif main_thread.last_state is not main_thread.current_motion.state:
            #Simply update the last state
            main_thread.last_state = main_thread.current_motion.state
        #Release Lock
        main_thread.state_lock.release()
    print "Done!"
    #show()
    '''tasks.close()
    tasks.join_thread()
    result_list = []
    for i in range(num_tasks):
        #import pdb; pdb.set_trace()
        result_list.append(results.get())
        if result_list[-1].state == BoltPR2MotionBuf.SQUEEZE:
            plt.subplot(211)
            plt.plot(result_list[-1].electrodes[0])
            plt.grid(True)
            plt.title('Electrodes - Raw and Normalized')
            plt.subplot(212)
            plt.plot(result_list[-1].electrodes_normalized[0])
            plt.xlabel('time')
            plt.grid(True)
            plt.show()
            #print 'Result:', result
    done = True'''



if __name__ == '__main__':
  main(sys.argv[1:])
