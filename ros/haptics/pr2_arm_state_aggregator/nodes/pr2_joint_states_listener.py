#!/usr/bin/env python
#spins off a thread to listen for joint_states messages
#and provides the same information (or subsets of) upon request
#this whole script is curtousy of the ROSWiki Tutorials
import roslib
roslib.load_manifest('pr2_arm_state_aggregator')
import rospy
from sensor_msgs.msg import JointState
import threading


#holds the latest states obtained from joint_states messages
class PR2JointStatesListener:

    def __init__(self):
        self.lock = threading.Lock()
        self.name = []
        self.position = []
        self.velocity = []
        self.effort = []
        self.thread = threading.Thread(target=self.joint_states_listener)
        self.thread.start()
       
    #Kill the thread upon cleanup
    def __del__(self):
        self.thread.join()

    #thread function: listen for joint_states messages
    def joint_states_listener(self):
        rospy.Subscriber('joint_states', JointState, self.joint_states_callback)
        rospy.spin()


    #callback function: when a joint_states message arrives, save the values
    def joint_states_callback(self, msg):
        self.lock.acquire()
        self.name = msg.name
        self.position = msg.position
        self.velocity = msg.velocity
        self.effort = msg.effort
        self.lock.release()


    #returns (found, position, velocity, effort) for the joint joint_name 
    #(found is 1 if found, 0 otherwise)
    def return_joint_state(self, joint_name):

        #no messages yet
        if self.name == []:
            rospy.logerr("No robot_state messages received!\n")
            return (0, 0., 0., 0.)

        #return info for this joint
        self.lock.acquire()
        if joint_name in self.name:
            index = self.name.index(joint_name)
            position = self.position[index]
            velocity = self.velocity[index]
            effort = self.effort[index]
        #unless it's not found
        else:
            rospy.logerr("Joint %s not found!", (joint_name,))
            self.lock.release()
            return (0, 0., 0., 0.)
        self.lock.release()
        return (1, position, velocity, effort)


    #returns arrays of position, velocity, and effort
    #for a list of joints specified by name
    def return_joint_states(self, req):
        joints_found = []
        positions = []
        velocities = []
        efforts = []
        for joint_name in req:
            (found, position, velocity, effort) = self.return_joint_state(joint_name)
            joints_found.append(found)
            positions.append(position)
            velocities.append(velocity)
            efforts.append(effort)
        return (joints_found, positions, velocities, efforts)


#default main
if __name__ == "__main__":

    latestjointstates = PR2JointStatesListener()

    print "joints_states_listener started"
    rospy.spin()
