#!/usr/bin/env python

import roslib; roslib.load_manifest('language_test_pipe')
import rospy
import sys
import threading
#print sys.path


from bolt_pr2_motion_obj import BoltPR2MotionObj

import multiprocessing
from collections import defaultdict

from biotac_sensors.msg import BioTacHand
from pr2_gripper_accelerometer.msg import PR2GripperAccelerometerData
from std_msgs.msg import Int8


import multiprocessing


def processMotion(task_queue, result_queue):
    name = multiprocessing.current_process().name
    print name, 'Starting'
    #Grab the current motion from the queue
    current_motion = task_queue.get()

    #operate to determine stuff
    answer = current_motion.state
 
    result_queue.put( answer )
    print name, ' received motion ', answer
    


class LanguageTestMainThread:

    def __init__(self):
        rospy.init_node('language_test_subscribers')
        rospy.loginfo('main language test thread initializing...')
        self.current_motion = BoltPR2MotionObj()
        self.last_state = BoltPR2MotionObj.DISABLED
        self.thread_lock = threading.Lock()
        # Create empty lists (temporary buffers) to store all data 
        self.electrodes = defaultdict(list)
        self.tdc = defaultdict(list)
        self.tac = defaultdict(list)
        self.pdc = defaultdict(list)
        self.pac = defaultdict(list)
        self.gripper_velocity = []
        self.gripper_position = []
        self.gripper_effort = []
        self.accelerometer = []
        self.state = BoltPR2MotionObj.DISABLED
        self.accel_downsample_counter = 0    

    def clear_motion(self):
        #Empty out all temporary buffers
        self.electrodes.clear()
        self.tdc.clear()
        self.tac.clear()
        self.pdc.clear()
        self.pac.clear()
        self.gripper_velocity.empty()
        self.gripper_position.empty()
        self.gripper_effort.empty()
        self.accelerometer.empty()
        self.state = BoltPR2MotionObj.DISABLED
        #Reset current_motion, but save off state
        next_state = self.current_motion.state
        self.current_motion = BoltPR2MotionObj()
        self.current_motion.state = next_state


    def populate_motion(self):
        import pdb; pdb.set_trace()
        pass


        #self.last_motion_state = BoltPR2MotionObj.DISABLED

    def start_listeners(self):
        #Start BioTac Subscriber
        rospy.Subscriber('biotac_pub', BioTacHand, self.biotacCallback,queue_size=1)
        #Start Accelerometer Subscriber
        rospy.Subscriber('/pr2_gripper_accelerometer/data', PR2GripperAccelerometerData, self.accelerometerCallback,queue_size=1)
        #Start Gripper Controller State Subscriber
        rospy.Subscriber('/simple_gripper_controller_state', Int8, self.gripperControllerCallback, queue_size=1)


    def accelerometerCallback(self, msg):
        self.accel_downsample_counter = self.accel_downsample_counter + 1    
        if not self.accel_downsample_counter % 10: 
            self.accel_downsample_counter = 0
            self.thread_lock.acquire()
            if self.current_motion.state not in (BoltPR2MotionObj.DISABLED, BoltPR2MotionObj.DONE, BoltPR2MotionObj.CENTER_GRIPPER):
                # Store accelerometer
                self.accelerometer.append( (msg.acc_x_raw, msg.acc_y_raw, msg.acc_z_raw) )

                # Store gripper
                self.gripper_position.append(msg.gripper_joint_position)
                self.gripper_velocity.append(msg.gripper_joint_velocity)
                self.gripper_effort.append(msg.gripper_joint_effort)

            self.thread_lock.release()

    def biotacCallback(self, msg):
        self.thread_lock.acquire()
        if self.current_motion.state not in (BoltPR2MotionObj.DISABLED, BoltPR2MotionObj.DONE, BoltPR2MotionObj.CENTER_GRIPPER):
            num_fingers = len(msg.bt_data)
            for finger_index in xrange(num_fingers):    
    
                self.tdc[finger_index].append( msg.bt_data[finger_index].tdc_data)
                self.tac[finger_index].append( msg.bt_data[finger_index].tac_data)
                self.pdc[finger_index].append( msg.bt_data[finger_index].pdc_data)
                self.pac[finger_index].append( msg.bt_data[finger_index].pac_data)
                self.electrodes[finger_index].append( msg.bt_data[finger_index].electrode_data)

        self.thread_lock.release()


    def gripperControllerCallback(self, gripper_state):
        #Save off last read state?
        #self.last_state_state = self.current_motion.state
        self.thread_lock.acquire()
        self.current_motion.state = gripper_state.data
        self.thread_lock.release()


def main(argv):

    #Look for Change in status
    #If change detected, close data (file?) and pass it into a new thread

    # Establish communication queues
    '''tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_tasks = 0
    for i in range(1,1000):
      if i%100 is 0 and i < 500:
        new_process = multiprocessing.Process(target=processMotion, args=(tasks,results))
        recorded_motion = BoltPR2MotionObj()
        recorded_motion.tdc = i
        recorded_motion.tac = i
        recorded_motion.pdc = i
        recorded_motion.pac = i
        tasks.put(recorded_motion)
        new_process.start()
        num_tasks = num_tasks + 1

    tasks.close()
    tasks.join_thread()
    
    for i in range(num_tasks):
        result = results.get()
        print 'Result:', result'''

    # Establish communication queues
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_tasks = 0

    main_thread =  LanguageTestMainThread()
    main_thread.start_listeners()
    valid_state_tuple = (BoltPR2MotionObj.THERMAL_HOLD, BoltPR2MotionObj.SLIDE,
                         BoltPR2MotionObj.SQUEEZE, BoltPR2MotionObj.TAP,
                         BoltPR2MotionObj.SLIDE_FAST, BoltPR2MotionObj.DONE)
    while not rospy.is_shutdown():
        #Acquire Lock
        main_thread.thread_lock.acquire()
        if  main_thread.current_motion.state in valid_state_tuple and \
            main_thread.last_state in valid_state_tuple and \
            main_thread.last_state is not main_thread.current_motion.state:
            #print "current state %d" % main_thread.current_motion.state 
            #print "last state %d" % main_thread.last_state

            #Store off next state to see if we're done
            next_state = main_thread.current_motion.state
            #Close up the current current_motion and send it to a thread
            main_thread.current_motion.state = main_thread.last_state
            #Store the next state as the last state to be used to see when a change occurs
            main_thread.last_state = next_state
            #Reshape the data for the threads to compute features
            main_thread.populate_motion()

            #Place current_motion in the que
            tasks.put(main_thread.current_motion)
            #Reset current_motion
            main_thread.clear_motion()

            #Spin up a new thread
            new_process = multiprocessing.Process(target=processMotion, args=(tasks,results))
            new_process.start()
            num_tasks = num_tasks + 1

            #Check to see if the motions have finished
            if next_state is BoltPR2MotionObj.DONE:
                break

        elif main_thread.last_state is not main_thread.current_motion.state:
            #Simply update the last state
            main_thread.last_state = main_thread.current_motion.state
        #Release Lock
        main_thread.thread_lock.release()

    tasks.close()
    tasks.join_thread()

    for i in range(num_tasks):
        result = results.get()
        print 'Result:', result




if __name__ == '__main__':
  main(sys.argv[1:])
