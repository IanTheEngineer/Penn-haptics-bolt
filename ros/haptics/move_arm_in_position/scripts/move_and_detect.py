#! /usr/bin/python

import roslib
roslib.load_manifest("move_arm_in_position")
import move_arm_in_position
import rospy

def main():
    rospy.loginfo("Moving to ideal position")
    haptics = move_arm_in_position.MoveToHaptics()
    #_, box = haptics.detect_and_filter()
    #haptics.move_to_ideal_position(box)
    if haptics.move_arm_to_pre_haptics():
        haptics.execute_haptics()
    
    
if __name__ == "__main__":
    rospy.init_node("move_and_detect")
    main()
    rospy.loginfo("Done")