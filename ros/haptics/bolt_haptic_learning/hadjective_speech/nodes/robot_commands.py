#!/usr/bin/env python
import rospy
from std_msgs.msg import Int8 
import subprocess


def callback(data):
    
    rospy.loginfo(rospy.get_name() + ": I heard %s" % data.data)
    
    if (data.data == 0):
        subprocess.call('rosrun pr2_control_utilities pr2_joint_mover.py -l /u/imcmahon/ros/bolt_demo/pr2_rest.pos', shell=True)

    elif (data.data == 1):
        subprocess.call('rosrun move_arm_in_position move_and_detect.py', shell=True) 

    elif (data.data == 2):
        print 'got here'
        #subprocess.call('roslaunch hadjective_test_pipe start_collection_threads.launch &', shell=True)
        #subprocess.call('rosrun hadjective_test_pipe bolt_data_collector_thread.py', shell=True)
        #subprocess.call('rosrun hadjective_mkl_classifier mkl_classifi_node.py', shell=True)

    else:
        pass

def listener():
    rospy.init_node('hadjective_command_node', anonymous=True)
    rospy.Subscriber('hadjective_cmds', Int8, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
