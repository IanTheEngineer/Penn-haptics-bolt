#!/usr/bin/env python

import roslib; roslib.load_manifest('hadjective_static_classifier')
import rospy
import cPickle, sys
from bolt_pr2_motion_obj import BoltPR2MotionObj
from bolt_feature_obj import BoltFeatureObj
from static_feature_obj import StaticFeatureObj
import numpy as np

import bolt_learning_utilities as bolt_utilities
import extract_features as extract_features
import static_features_penn as upenn_features
import matplotlib.pyplot as plt 
import os

from hadjective_speech.msg import Adj, AdjList
from std_msgs.msg import String
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA 
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn import preprocessing

class HadjectiveStaticClassifier(object):
    
    def __init__(self):
        rospy.loginfo("Loading the static adjective classifiers")

        # Create paths to files
        DIR = roslib.packages.get_pkg_dir("hadjective_static_classifier", required=True) + "/classifiers/" 
        all_classifiers_file = os.path.join(DIR, "all_trained_classifiers.pkl")

        # Load pickle files containing classifiers and chains
        self.all_classifiers = cPickle.load(open(all_classifiers_file))

        # Setup subscriber and publisher nodes
        rospy.Subscriber("new_hadjective_motion_pickle", String, self.callback)
        self.adjectives_pub = rospy.Publisher("/hadjective_speech", AdjList)

        rospy.loginfo("All static classifiers loaded")

    # Computes the probability vector and results
    def compute_probability_vector(self, bolt_obj):

        # First object - initialize variables to store
        # Also clears out the vectors for new run
        if bolt_obj.state  == bolt_obj.TAP:
            # Store results as they come in
            self.adjective_vectors = []
            self.all_motion_results = dict()
            self.results = dict()
        
        # Store dictionary of strings
        self.state_string = {bolt_obj.DISABLED:'disabled',
                    bolt_obj.THERMAL_HOLD:'thermal_hold',
                    bolt_obj.SLIDE:'slide',
                    bolt_obj.SQUEEZE:'squeeze',
                    bolt_obj.TAP:'tap',
                    bolt_obj.DONE:'done',
                    bolt_obj.SLIDE_FAST:'slide_fast',
                    bolt_obj.CENTER_GRIPPER:'center_gripper'
                    }   
        
        # store dictionary for detailed states
        self.detailed_states = {bolt_obj.DISABLED:'MOVE_ARM_START_POSITION',
                                bolt_obj.SQUEEZE:'SQUEEZE_SET_PRESSURE_SLOW',
                                bolt_obj.THERMAL_HOLD:'HOLD_FOR_10_SECONDS',
                                bolt_obj.SLIDE:'SLIDE_5CM',
                                bolt_obj.SLIDE_FAST:'MOVE_DOWN_5CM'
                                }

       
        # Get the current motion 
        current_motion = self.state_string[bolt_obj.state] 
        
        # Check if state passed in should be processed
        if bolt_obj.state not in self.detailed_states:
            return
        else:
            # Get detailed state if exists
            current_detailed_state = self.detailed_states[bolt_obj.state] 

            # Check if the state is a the disabled state
            if bolt_obj.state == bolt_obj.DISABLED:
                self.norm_bolt_obj = upenn_features.pull_detailed_state(bolt_obj,current_detailed_state)
                return
            else:
                self.bolt_object = upenn_features.pull_detailed_state(bolt_obj, current_detailed_state)

            # Checks to make sure that the norm obj has been created
            if not self.norm_bolt_obj:
                print "Warning: there is no normalization data"

            # Build the static features 
            static_feature_object, self.static_features_array = upenn_features.extract_static_features(self.bolt_object, self.norm_bolt_obj)
            static_feats = upenn_features.createFeatureVector(static_feature_object)
            
            # Add the current vector to all the features
            self.adjective_vectors.append(static_feats)

            print len(self.adjective_vectors)
            # Check if all motions have been performed
            # If so - feed into classifier
            if len(self.adjective_vectors) == 4:
                print 'All motions received! Computing adjective scores'
                for classifier in self.all_classifiers:
         
                    # Pull out which adjective we are working on        
                    adj = classifier['adjective'] 
            
                    # Pull out the feature vectors for static/dynamic
                    all_static_features = np.hstack(self.adjective_vectors)
               
                    # Normalize the features using scaler
                    all_static_feats_scaled = classifier['scaler'].transform(all_static_features)

                    # Predict adjective with computed kernel
                    clf = classifier['classifier']
                    self.results[adj] = clf.predict(all_static_feats_scaled)

                # Store off the adjectives that returned true
                adjectives_found = []
                for adj in self.results:
                    if self.results[adj] == 1:
                        adjectives_found.append(Adj(adj))

                publish_string = AdjList()
                publish_string = adjectives_found

                # Print and publish results!
                print "Results from Static Only classification"
                #print self.mkl_results
                print str(adjectives_found) 
                #self.adjectives_pub.publish(publish_string)


    def callback(self, msg):
        current_motion = cPickle.loads(msg.data)
        rospy.loginfo("Received Motion: %s" % current_motion.state_string[current_motion.state])
        self.compute_probability_vector(current_motion)

    def linear_kernel_test(self, testK, origK, n_jobs):
        return pairwise_kernels(testK, origK, metric="linear", n_jobs=n_jobs)

def main():
    rospy.init_node('hadjective_static_classifier')
    HadjectiveStaticClassifier()
    rospy.spin()

if __name__ == '__main__':
    main()

