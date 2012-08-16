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

class MyFancyClass(object):
    
    def __init__(self, name):
        self.name = name
    
    def do_something(self):
        proc_name = multiprocessing.current_process().name
        current_motion = BoltPR2MotionObj()
        current_motion.tdc = proc_name
        print current_motion
        #print 'Doing something fancy in %s for %s!' % (proc_name, self.name)

def worker(q):
    obj = q.get()
    obj.do_something()

class LanguageTestMainThread:

  def __init__(self):
    rospy.init_node('Language Test Main Thread')
    rospy.loginfo('main language test thread initializing...')

  def start_listeners(self):
    pass
    #Start BioTac Subscriber
    #rospy.Subscriber('biotac_pub', BioTacHand, self.biotacCallback,queue_size=1000)
    #Start Accelerometer Subscriber
    #rospy.Subscriber('/pr2_gripper_accelerometer/data', PR2GripperAccelerometerData, self.accelerometerCallback,queue_size=1000)
    #Start Gripper Controller State Subscriber
    #rospy.Subscriber('/simple_gripper_controller_state', Int8, self.gripperControllerCallback, queue_size=1000)


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
    queue = multiprocessing.Queue()
    for i in range(1,1000):
      if i%100 is 0 and i < 500:
        p = multiprocessing.Process(target=worker, args=(queue,))
        p.start()
        queue.put(MyFancyClass('Master Thread'))
    queue.close()
    queue.join_thread()
    p.join()
        


if __name__ == '__main__':
  main(sys.argv[1:])
