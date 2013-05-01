#!/usr/bin/env python

import roslib; roslib.load_manifest('hadjective_mkl_classifier')
import rospy
import cPickle, sys
from bolt_pr2_motion_obj import BoltPR2MotionObj
from bolt_feature_obj import BoltFeatureObj
from static_feature_obj import StaticFeatureObj
from adjective_classifier import AdjectiveClassifier

import bolt_learning_utilities as bolt_utilities
import extract_features as extract_features
import upenn_features as upenn_features
import matplotlib.pyplot as plt 
import os

from std_msgs.msg import String
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA 


class HadjectiveMKLClassifier(object):
    
    def __init__(self):
        rospy.loginfo("Loading the mkl adjective classifiers")
        DIR = roslib.packages.get_pkg_dir("hadjective_mkl_classifier", required=True) + "/classifiers/" 
        all_classifiers_file = os.path.join(DIR, "all_trained_classifiers.pkl")
        hmm_chains_file = os.path.join(DIR, "loaded_chains.pkl")
        #scaler_file = os.path.join(DIR, "scaler.pkl")

        self.all_classifiers = cPickle.load(open(all_classifiers_file))
        self.hmm_chains = cPickle.load(open(hmm_chains_file))

        rospy.Subscriber("new_hadjective_motion_pickle", String, self.callback)
        self.adjectives_pub = rospy.Publisher("/feature_MKL_adjectives", String)

        rospy.loginfo("All MKL classifiers loaded")

    # Computes the probability vector and results
    def compute_probability_vector(self, bolt_obj):

        # First object - initialize variables to store
        if bolt_obj.state  == bolt_obj.TAP:
            # Store results as they come in
            self.adjective_vectors_static = dict() 
            self.adjective_vectors_dynamic = dict() 
            self.all_motion_results = dict()
        
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
            
            import pdb; pdb.set_trace()

            # Store all feature vectors into one large vector array
            # Technically we don't need to store the static 
            # features by adjective - but it's cleaner this way
            for classifier in self.all_classifiers:

                adj = classifier['adjective']

                # need to add pkl to the name b/c the hmm chain dictionary
                # takes adjective.pkl as its key
                hmm_adj_name = '.'.join((adj,'pkl'))
                
                # Initialize the dictionary with adjective
                if adj in self.adjective_vectors_static:
                    pass
                else:
                    self.adjective_vectors_static[adj] = list()
                    self.adjective_vectors_dynamic[adj] = list()
              
                # Pull out chain associated with adjective
                # ordering of sensors - [pac, pdc, electrodes, tac]
                sensor_hmm = ['pac', 'pdc', 'electrodes', 'tac']
                hmm_features_phase = [];
                for sensor in sensor_hmm:
                    hmm_chain = self.hmm_chains[hmm_adj_name].chains[current_detailed_state][sensor]
                    hmm_data = self.bolt_object.hmm[sensor]
                    hmm_features_phase.append(hmm_chain.score(hmm_data))
                   
                import pdb; pdb.set_trace()
                # Create dynamic features
                # dynamic_feat = # fill in

                # Store the feature vector by adjective away
                self.adjective_vectors_static[adj].append(static_feats)
                self.adjective_vectors_dynamic[adj].append(dynamic_feats)

            # Check if all motions have been performed
            # If so - feed into classifier
            print len(self.adjective_vectors_dynamic[adj])
            if len(self.adjective_vectors[adj]) == 16:
                
                # Create kernels for both static and dynamic

                # Merge kernels

                for adj in self.mkl_classifiers:
                    # Calculate prediction!
                    mkl_results[adj] = self.mkl_classifier[adj].predict()

                # Store off the adjectives that returned true
                adjectives_found = []
                for adj in self.mkl_results:
                    if self.mkl_results[adj] == 1:
                        adjectives_found.append(adj)

            # Print and publish results!
            print "Results from max classification"
            print self.mkl_results
            print str(adjectives_found) 
            self.adjectives_pub.publish(str(adjectives_found))


    def callback(self, msg):
        current_motion = cPickle.loads(msg.data)
        rospy.loginfo("Current Motion: %s" % current_motion.state_string[current_motion.state])
        self.compute_probability_vector(current_motion)

def main():
    rospy.init_node('hadjective_mkl_classifier')
    HadjectiveMKLClassifier()
    rospy.spin()

if __name__ == '__main__':
    main()

