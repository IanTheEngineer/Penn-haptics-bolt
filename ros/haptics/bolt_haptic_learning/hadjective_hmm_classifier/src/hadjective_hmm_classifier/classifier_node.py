#! /usr/bin/python

import roslib
roslib.load_manifest('hadjective_hmm_classifier')
import rospy
import cPickle
import os
from bolt_pr2_motion_obj import BoltPR2MotionObj
from std_msgs.msg import String
import numpy as np

#cPickle madness
import sys
from training_segments import adjective_classifier 
from training_segments import hmm_chain
from training_segments import hmm_classifier
from training_segments import discretizer
sys.modules["adjective_classifier"] = adjective_classifier
sys.modules["hmm_chain"] = hmm_chain
sys.modules["hmm_classifier"] = hmm_classifier
sys.modules["discretizer"] = discretizer


class AdjectiveClassifierNode(object):
    def __init__(self):
        rospy.loginfo("Loading the adjectives")
        DIR = roslib.packages.get_pkg_dir("hadjective_hmm_classifier", required=True) + "/data/"
        adjectives_filename = os.path.join(DIR, "all_adjectives.pkl")        
        self.adjectives = cPickle.load(open(adjectives_filename))
        
        hadjective_msg_name = "hadjective_motion_pickle"
        rospy.Subscriber(hadjective_msg_name, String, self.callback)
        
        self.received_data = {}
        rospy.loginfo("Ready")

    def __get_phase_from_obj(self, obj):
        isinstance(obj, BoltPR2MotionObj)
        if obj.state == obj.SQUEEZE:
            phase = "SQUEEZE_SET_PRESSURE_SLOW"
        elif obj.state == obj.THERMAL_HOLD:
            phase = "HOLD_FOR_10_SECONDS"
        elif obj.state == obj.SLIDE:
            phase = "SLIDE_5CM"
        elif obj.state == obj.SLIDE_FAST:
            phase = "MOVE_DOWN_5CM"
        else:
            phase = None
        return phase
        
    def __create_data_dict(self, obj, phase):
        isinstance(obj, BoltPR2MotionObj)
        detailed_state = np.array(obj.detailed_state)
        indexes = detailed_state == phase
        
        data = {}
        electrodes = np.hstack(obj.electrodes)
        #rospy.loginfo("Electrodes shape: %s", electrodes.shape)
        pac = np.hstack(obj.pac)
        #rospy.loginfo("Pac shape: %s", pac.shape)
        pdc = np.hstack(obj.pdc)
        #rospy.loginfo("Pdc shape: %s", pdc.shape)
        tac = np.hstack(obj.tac)
        #rospy.loginfo("Tac shape: %s", tac.shape)
        
        data['electrodes'] = electrodes[indexes,:]        
        data['pac'] = pac[indexes, :]
        data['pdc'] = np.atleast_2d(pdc[indexes, :]).T
        data['tac'] = np.atleast_2d(tac[indexes, :]).T
        
        rospy.loginfo("Electrodes shape: %s", data['electrodes'].shape)
        rospy.loginfo("Pac shape: %s", data['pac'].shape)
        rospy.loginfo("Pdc shape: %s", data['pdc'].shape)
        rospy.loginfo("Tac shape: %s", data['tac'].shape)
        
        return data
    
    def callback(self, msg):
        current_motion = cPickle.loads(msg.data)
        rospy.loginfo("Current Motion: %s" % current_motion.state_string[current_motion.state])
        
        phase = self.__get_phase_from_obj(current_motion)
        if phase is None:
            rospy.loginfo("Current motion is not supported")
            return
        rospy.loginfo("Using phase: %s", phase)
        
        self.received_data[phase] = self.__create_data_dict(current_motion, phase)
        
        if phase == "MOVE_DOWN_5CM":
            rospy.loginfo("Last phase, going for classification")
            
            positives = []
            for clf in self.adjectives:
                features = clf.extract_features(self.received_data)        
                output = clf.predict(features)
                if output[0] == 1:
                    positives.append(clf.adjective)                
            
            rospy.loginfo("Classification done")
            rospy.loginfo("Adjectives: %s", " ".join(positives))
            self.received_data = {}
        

def main():
    rospy.init_node('hadjective_hmm_classifier')
    AdjectiveClassifierNode()
    rospy.spin()
    
if __name__ == "__main__":
    main()
    
