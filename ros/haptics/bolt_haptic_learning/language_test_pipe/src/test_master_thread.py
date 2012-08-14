#!/usr/bin/env python

import roslib; roslib.load_manifest('language_test_pipe')
import rospy
import sys
#print sys.path


from bolt_pr2_motion_obj import BoltPR2MotionObj

import multiprocessing

from biotac_sensors.msg import BioTacHand
from pr2_gripper_accelerometer.msg import PR2GripperAccelerometerData
from std_msgs.msg import Int8


import multiprocessing


def processMotion(task_queue, result_queue):
    #Grab the current motion from the queue
    current_motion = task_queue.get()

    #operate to determine stuff
    answer = current_motion.tdc

    result_queue.put( answer )
    


class LanguageTestMainThread:

    def __init__(self):
        rospy.init_node('Language Test Main Thread')
        rospy.loginfo('main language test thread initializing...')

    def start_listeners(self):
        #Start BioTac Subscriber
        rospy.Subscriber('biotac_pub', BioTacHand, self.biotacCallback,queue_size=1000)
        #Start Accelerometer Subscriber
        rospy.Subscriber('/pr2_gripper_accelerometer/data', PR2GripperAccelerometerData, self.accelerometerCallback,queue_size=1000)
        #Start Gripper Controller State Subscriber
        rospy.Subscriber('/simple_gripper_controller_state', Int8, self.gripperControllerCallback, queue_size=1000)


    def accelerometerCallback(self, accelerometer_data):
        pass

    def biotacCallback(self, biotac_data):
        pass
    # Stores the frame count into the message
    #data.header.frame_id = self.frame_count
    #Add Data to some buffer
    # Move to next frame 
    #self.frame_count += 1


    def gripperControllerCallback(self, gripper_status_data):
        pass


def main(argv):

    #Look for Change in status
    #If change detected, close data (file?) and pass it into a new thread

    # Establish communication queues
    tasks = multiprocessing.Queue()
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
        print 'Result:', result

if __name__ == '__main__':
  main(sys.argv[1:])
