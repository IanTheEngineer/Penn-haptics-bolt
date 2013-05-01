#!/usr/bin/env python

import roslib; roslib.load_manifest('hadjective_test_alg_template')
import rospy
import cPickle, sys
from bolt_pr2_motion_obj import BoltPR2MotionObj
from std_msgs.msg import String

def callback(msg):
    current_motion = cPickle.loads(msg.data)
    rospy.loginfo("Current Motion: %s" % current_motion.state_string[current_motion.state])

def listener():
    rospy.init_node('hadjective_test_alg')
    rospy.Subscriber("new_hadjective_motion_pickle", String, callback)
    rospy.spin()

def main(argv):
    listener()

if __name__ == '__main__':
    main(sys.argv[1:])

