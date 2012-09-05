#!/usr/bin/env python

import roslib; roslib.load_manifest('hadjective_knn_classifier')
import rospy
import cPickle, sys
from bolt_pr2_motion_obj import BoltPR2MotionObj
from bolt_feature_obj import BoltFeatureObj

import bolt_learning_utilities as utilities
import extract_features as extract_features
import matplotlib.pyplot as plt 
import os

from std_msgs.msg import String
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA 


class HadjectiveSVMClassifier(object):
    
    def __init__(self):
        rospy.loginfo("Loading the knn adjective classifiers")
        DIR = roslib.packages.get_pkg_dir("hadjective_knn_classifier", required=True) + "/classifiers/" 
        all_classifiers_file = os.path.join(DIR, "all_knn_classifiers.pkl")
        pca_file = os.path.join(DIR, "pca.pkl")
        best_motion_file = os.path.join(DIR, "best_classifiers_knn.pkl")
        scaler_file = os.path.join(DIR, "scaler.pkl")
        #ensemble_classifier_file = os.path.join(DIR, "ensemble_classifier.pkl")

        self.all_classifiers = cPickle.load(open(all_classifiers_file))
        #self.ensemble_classifiers = cPickle.load(open(ensemble_classifier_file))
        self.pca_model = cPickle.load(open(pca_file))
        self.best_motion_dict = cPickle.load(open(best_motion_file))

        self.scaler_dict = cPickle.load(open(scaler_file))
        
        # Subscriber for pickle values
        rospy.Subscriber("new_hadjective_motion_pickle", String, self.callback)

        # Publisher for results
        self.adjectives_pub = rospy.Publisher("/feature_knn_adjectives", String)
        rospy.loginfo("All knn classifiers loaded")

   
        self.feature_list = ["pdc_rise_count", "pdc_area", "pdc_max", "pac_energy", "pac_sc", "pac_sv", "pac_ss", "pac_sk", "tac_area", "tdc_exp_fit", "gripper_min", "gripper_mean", "transform_distance", "electrode_polyfit"]

    # Computes the probability vector and results
    def compute_probability_vector(self, bolt_obj):
        if bolt_obj.state  == bolt_obj.TAP:
            # Store results as they come in
            self.adjective_vectors = dict() 
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
       
        
        current_motion = self.state_string[bolt_obj.state] 
       
        # Create feature vector 
        self.bolt_object = bolt_obj 
        utilities.normalize_data(self.bolt_object)
        self.bolt_feature_object = extract_features.extract_features(self.bolt_object, self.pca_model[current_motion]) 

        # Create a dictionary to store the results in
        for adj in self.all_classifiers:
            results, prob = utilities.compute_adjective_probability_score(self.all_classifiers[adj], self.bolt_feature_object, self.feature_list, adj, self.scaler_dict)
            
            # Store off adjective probabilities for ensemble 
            if adj in self.adjective_vectors:
                pass 
            else:
                self.adjective_vectors[adj] = list()
           
            self.adjective_vectors[adj].append(prob)
           
            # Store classifier score based on best motion
            best_motion = self.best_motion_dict[adj][1]
            if current_motion  == best_motion:
                rospy.loginfo("Best Motion is: %s" % best_motion)
                self.all_motion_results[adj] = results
        
        print len(self.adjective_vectors[adj])
        if len(self.adjective_vectors[adj]) == 5:
            ensembled_results = dict() 
            print self.adjective_vectors 
            #for adj in self.adjective_vectors: 
            #    ensembled_results[adj] = self.ensemble_classifiers[adj].predict(self.adjective_vectors[adj])

            # Store off the adjectives that returned true
            adjectives_found = []
            for adj in self.all_motion_results:
                if self.all_motion_results[adj] == 1:
                    adjectives_found.append(adj)

            print "Results from max classification"
            print self.all_motion_results
            print str(adjectives_found) 
            self.adjectives_pub.publish(str(adjectives_found))

            #print ensembled_results 


    def callback(self, msg):
        current_motion = cPickle.loads(msg.data)
        rospy.loginfo("Current Motion: %s" % current_motion.state_string[current_motion.state])
        self.compute_probability_vector(current_motion)

def main():
    rospy.init_node('hadjective_knn_classifier')
    HadjectiveSVMClassifier()
    rospy.spin()

if __name__ == '__main__':
    main()

