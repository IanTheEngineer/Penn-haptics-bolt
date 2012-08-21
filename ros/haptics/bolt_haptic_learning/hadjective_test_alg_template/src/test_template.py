#!/usr/bin/env python

import roslib; roslib.load_manifest('hadjective_test_alg_template')
import rospy
from std_msgs.msg import String
import pickle
import cPickle, sys
from bolt_pr2_motion_obj import BoltPR2MotionObj


def callback(data):
    import pdb; pdb.set_trace()
    current_motion = BoltPR2MotionObj()
    current_motion = pickle.loads(str(data))
    rospy.loginfo("Current Motion: %s" % current_motion.state_string[current_motion.state])

def listener():
    rospy.init_node('hadjective_test_alg')
    rospy.Subscriber("hadjective_motion_pickle", String, callback)
    rospy.spin()


def main(argv):
    listener()

if __name__ == '__main__':
    main(sys.argv[1:])

