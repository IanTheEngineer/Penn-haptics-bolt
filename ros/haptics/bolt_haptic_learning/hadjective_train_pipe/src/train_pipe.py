#!/usr/bin/env python
import roslib; roslib.load_manifest("hadjective_train_pipe")
import rospy
import numpy as np
import sys 
import os
from optparse import OptionParser
import cPickle
import pickle
import bolt_learning_utilities as utilities
import extract_features as extract_features
import matplotlib.pyplot as plt 

from bolt_feature_obj import BoltFeatureObj
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import classification_report
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.decomposition import PCA


# Loads the data from h5 table and adds labels
# Returns the dictionary of objects
def loadDataFromH5File(input_file, adjective_file):
   
    # Takes the input h5 file and converts into bolt object data
    all_bolt_data = utilities.convertH5ToBoltObjFile(input_file, None, False);
   
    # Inserts adjectives into the bolt_data  
    all_bolt_data_adj = utilities.insertAdjectiveLabels(all_bolt_data, "all_objects_majority4.pkl", adjective_file, True)

    return all_bolt_data_adj

# Fits PCA for electrode training data
def fit_electrodes_pca(train_data):

    pca_dict = dict()
     
    # Fit PCA on all trials, for each motion separately
    for motion_name in train_data:
        trial_list = train_data.get(motion_name)

        # Pull out electrode data on both fingers, all trials for one motion
        electrode_motion_data = np.concatenate((trial_list[0].electrodes_normalized[0],trial_list[0].electrodes_normalized[1]))
        for trial in range(1,len(trial_list)):
            electrode_motion_data = np.concatenate((electrode_motion_data,trial_list[trial].electrodes_normalized[0],trial_list[trial].electrodes_normalized[1]))
        

        # Store PCA by motion
        pca_dict[motion_name] = PCA(n_components=2).fit(electrode_motion_data)

    return(pca_dict)


# Takes the bolt data and extracts features to run
def BoltMotionObjToFeatureObj(all_bolt_data, electrode_pca_dict):
    """

    For each object - pull out features and store in feature_obj
    with the same structure as all_bolt_data
   
        Dictionary - "tap", "slide", "slow_slide", 
                     "thermal_hold", "squeeze"

    """

    # Store in feature class object
    all_features_obj_dict = dict();

    for motion_name in all_bolt_data:
        trial_list = all_bolt_data.get(motion_name)
        print motion_name

        feature_list = list()
        # For all objects
        for trial in trial_list:
            
            bolt_feature_obj = extract_features.extract_features(trial, electrode_pca_dict[motion_name])
            
            feature_list.append(bolt_feature_obj)

        # Store all of the objects away
        all_features_obj_dict[motion_name] = feature_list
            
    return all_features_obj_dict        
    

def bolt_obj_2_feature_vector(all_features_obj_dict, feature_name_list):
    """
    For each object - pull out features and store in feature_obj
    with the same structure as all_bolt_data
   
        Dictionary - "tap", "slide", "slow_slide", 
                     "thermal_hold", "squeeze"

    Directly store the features into a vector
    See createFeatureVector for more details on structure

    """
    
    # Store in feature class object
    all_features_vector_dict = dict()
    
    # For all motions
    for motion_name in all_features_obj_dict:
        
        feature_obj_list = all_features_obj_dict.get(motion_name)

        all_adjective_labels_dict = dict()
        feature_vector_list = list()

        # For all objects
        for bolt_feature_obj in feature_obj_list:

            # Create feature vector
            feature_vector = utilities.createFeatureVector(bolt_feature_obj, feature_name_list) 
            feature_vector_list.append(feature_vector)

            # Create label dictionary
            labels = bolt_feature_obj.labels
            for adjective in labels:
                # Check if it is the first time adjective added
                if (all_adjective_labels_dict.has_key(adjective)):
                    adjective_array = all_adjective_labels_dict[adjective]
                else:
                    adjective_array = list()
                
                # Store array
                adjective_array.append(labels[adjective])
                all_adjective_labels_dict[adjective] = adjective_array

        # Store all of the objects away
        all_features_vector_dict[motion_name] = np.array(feature_vector_list)
        
    return (all_features_vector_dict, all_adjective_labels_dict)      


def run_kmeans(input_vector, num_clusters, obj_data):
    """
    run_kmeans - expects a vector of features and the number of
                 clusters to generate

    Returns the populated clusters 
    """
    k_means = KMeans(init='k-means++', n_clusters=num_clusters, n_init=100)

    k_means.fit(input_vector)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_mean_labels_unique = np.unique(k_means_labels)

    # Pull clusters out
    clusters = dict()
    cluster_names = dict()
    cluster_ids = dict()
    cluster_all_adjectives = dict()
    
    # Get a list of all adjectives
    adjectives = obj_data[0].labels.keys()

    
    for labels in k_mean_labels_unique:
        idx = np.nonzero(k_means_labels == labels)
        clusters[labels] = [obj_data[i] for i in idx[0]]
        cluster_names[labels] = [obj.name for obj in clusters[labels]]
        cluster_ids[labels] = [obj.object_id for obj in clusters[labels]]
   
    for adj in adjectives:
        cluster_adj = dict()
        for labels in k_mean_labels_unique:
            cluster_adj[labels] = [obj.labels[adj] for obj in clusters[labels]] 
        
        cluster_all_adjectives[adj] = cluster_adj

    
    return (k_means_labels, k_means_cluster_centers, clusters)

def true_false_results(predicted_labels, true_labels):
 
    FP = (predicted_labels - true_labels).tolist().count(1)
    FN = (predicted_labels - true_labels).tolist().count(-1)
    TP = (predicted_labels & true_labels).tolist().count(1)
    TN = ((predicted_labels | true_labels) ^ True).tolist().count(1)


    return(TP, TN, FP, FN)


def matthews_corr_coef(TP,TN,FP,FN):

    try:
        MCC = (TP*TN - FP*FN)/(np.sqrt(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))))
    except:
        MCC = (TP*TN - FP*FN)/1

    return (MCC)

def create_scalers(train_feature_vector_dict):
    """
    Takes in the training feature vector dictionary, generates a scaler for each motion,
    and then returns the scalers
    """
    scaler_dict = dict()
    for motion_name in train_feature_vector_dict:
        scaler_dict[motion_name] = preprocessing.Scaler().fit(train_feature_vector_dict[motion_name])
    
    
    return scaler_dict



def train_knn(train_vector, train_labels, test_vector, test_labels, scaler):
    """
    train_knn - expects a vector of features and a nx1 set of
                corresponding labels.  Finally the number of
                neighbors used for comparison

    Returns a trained knn classifier
    """
    
    # Data scaling
    train_vector_scaled = scaler.transform(train_vector)
    test_vector_scaled = scaler.transform(test_vector)

    # Grid search with nested cross-validation
    parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15]}
    knn = GridSearchCV(KNeighborsClassifier(), parameters, score_func=f1_score, cv=4)
    knn.fit(train_vector_scaled, train_labels)
    knn_best = knn.best_estimator_
    score = f1_score(test_labels, knn.predict(test_vector_scaled))
    proba = knn.predict_proba(test_vector_scaled)
    report = classification_report(test_labels, knn.predict(test_vector_scaled))

    return (knn_best, proba, score, report)



def train_svm(train_vector, train_labels, test_vector, test_labels, scaler):
    """
    train_svm - expects a vector of features and a nx1 set of
                corresponding labels

    Returns a trained SVM classifier
    """
     
    # Data scaling
    train_vector_scaled = scaler.transform(train_vector)
    test_vector_scaled = scaler.transform(test_vector)
    
    # Grid search with nested cross-validation
    parameters = {'kernel': ['rbf'], 'C': [1, 1e1, 1e2, 1e3, 1e4], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4]}
    svm = GridSearchCV(SVC(probability=True), parameters, score_func=f1_score, cv=4)
    svm.fit(train_vector_scaled, train_labels)
    svm_best = svm.best_estimator_
    score = f1_score(test_labels, svm.predict(test_vector_scaled))
    proba = svm.predict_proba(test_vector_scaled)
    report = classification_report(test_labels, svm.predict(test_vector_scaled))

    return (svm_best, proba, score, report)


def single_train(train_vector, train_labels, test_vector, test_labels, scaler):
    """
    single_train - expects a vector of features and an nx1 set of
                   corresponding labels to train a single classifier
                   on 1 motion

    Returns trained KNN and SVM classifiers that have been optimized
    using grid search
    """
    # Run KNN
    knn, knn_proba, knn_score, knn_report = train_knn(train_vector, train_labels, test_vector, test_labels, scaler)
    print "Ran KNN\n"

    # Run SVM
    svm, svm_proba, svm_score, svm_report = train_svm(train_vector, train_labels, test_vector, test_labels, scaler)
    print "Ran SVM\n"

    return(knn, knn_proba, knn_score, knn_report, svm, svm_proba, svm_score, svm_report)


def full_train(train_feature_vector_dict, train_adjective_dict, test_feature_vector_dict, test_adjective_dict, scaler_dict):


    # Open text files for storing classification reports
    report_file_knn = open("Full_KNN_reports.txt", "w")
    report_file_svm = open("Full_SVM_reports.txt", "w")
    
    adjectives = train_adjective_dict.keys()
    all_knn_classifiers = dict()
    all_svm_classifiers = dict()
    
    # For all adjectives except warm and sparse
    for adj in adjectives:
        if  adj!="warm" and adj!="sparse":
            
            knn_classifiers = dict()
            svm_classifiers = dict()

            # For all motions
            for motion_name in train_feature_vector_dict:
            
                print "Training KNN and SVM classifiers for adjective %s, phase %s \n" %(adj, motion_name)
            
                # Train KNN and SVM classifiers using grid search with nested cv
                knn, knn_proba, knn_score, knn_report, svm, svm_proba, svm_score, svm_report = single_train(train_feature_vector_dict[motion_name], train_adjective_dict[adj], test_feature_vector_dict[motion_name], test_adjective_dict[adj], scaler_dict[motion_name])

                # Store classifiers, probabilities, and scores for each motion
                knn_classifiers[motion_name] = (knn, knn_proba, knn_score)
                svm_classifiers[motion_name] = (svm, svm_proba, svm_score)

                # Write classification reports into text files
                report_file_knn.write('Adjective: '+adj+'    Motion name: '+motion_name)
                report_file_knn.write('\nKNN report\n'+knn_report+'\n\n')
                report_file_svm.write('Adjective: '+adj+'    Motion name: '+motion_name)
                report_file_svm.write('\nSVM report\n'+svm_report+'\n\n')

                # Pkl each classifier and scaler separately
                cPickle.dump(knn, open('adjective_classifiers/'+adj+'_'+motion_name+'_knn.pkl', "w"), cPickle.HIGHEST_PROTOCOL)
                cPickle.dump(svm, open('adjective_classifiers/'+adj+'_'+motion_name+'_svm.pkl', "w"), cPickle.HIGHEST_PROTOCOL)

            
            # When trainings for a certain adjective with all five motions are done, save these classifiers
            cPickle.dump(knn_classifiers, open('adjective_classifiers/'+adj+'_knn.pkl', "w"), cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(svm_classifiers, open('adjective_classifiers/'+adj+'_svm.pkl', "w"), cPickle.HIGHEST_PROTOCOL)
    
            # Store classifiers by adjective
            all_knn_classifiers[adj] = knn_classifiers
            all_svm_classifiers[adj] = svm_classifiers

            print "Stored KNN and SVM classifiers for adjective %s in the directory adjective_classifiers\n" %(adj)
            
    report_file_knn.close()
    report_file_svm.close()

    print "Store off all of the classifiers together into one file\n" 
    cPickle.dump(all_knn_classifiers, open("all_knn_classifiers.pkl","w"), cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(all_svm_classifiers, open("all_svm_classifiers.pkl","w"), cPickle.HIGHEST_PROTOCOL)


    return (all_knn_classifiers, all_svm_classifiers)


def extract_ensemble_features(all_classifiers_dict, test_feature_vector_dict, test_adjective_dict, scaler_dict):
    
    """
    """

    ensemble_train_feature_vector_dict = dict()
    ensemble_test_feature_vector_dict = dict()
    

    report = open("testing_score_ensemble.txt", "w")
    # For all adjectives
    for adj in all_classifiers_dict:
        ensemble_train_feature_vector = []
        ensemble_test_feature_vector = []
        
        # For all motions
        for motion_name in all_classifiers_dict[adj]:
            classifier = all_classifiers_dict[adj][motion_name][0]
            train_probabilities = all_classifiers_dict[adj][motion_name][1]
            test_probabilities = classifier.predict_proba(scaler_dict[motion_name].transform(test_feature_vector_dict[motion_name]))
            ensemble_results = (classifier.predict(scaler_dict[motion_name].transform(test_feature_vector_dict[motion_name])))
            ensemble_train_feature_vector.append(train_probabilities[:,1].tolist())
            ensemble_test_feature_vector.append(test_probabilities[:,1].tolist())
            #import pdb; pdb.set_trace()
            report.write('\n\nAdjective: '+adj)
            report.write('\nMotion: '+motion_name+'\n')
            report.write(classification_report(test_adjective_dict[adj], ensemble_results))
            print "f1 score is %s" % classification_report(test_adjective_dict[adj], ensemble_results) 
        
        ensemble_train_feature_vector_dict[adj] = np.transpose(np.array(ensemble_train_feature_vector))
        ensemble_test_feature_vector_dict[adj] = np.transpose(np.array(ensemble_test_feature_vector))
    
    return (ensemble_train_feature_vector_dict, ensemble_test_feature_vector_dict)


def full_ensemble_train(train_feature_vector_dict, train_adjective_dict, test_feature_vector_dict, test_adjective_dict):
    """
    """

    # Open text file for storing classification reports
    ensemble_report_file = open("Full_Ensemble_Report.txt","w")

    all_ensemble_classifiers = dict()

    # For all adjectives
    for adj in train_adjective_dict:
                
        # Create ensemble scaler
        scaler = preprocessing.Scaler().fit(train_feature_vector_dict[adj])
        
        # Run SVM
        ensemble_svm, ensemble_proba, ensemble_score, ensemble_report = train_svm(train_feature_vector_dict[adj], train_adjective_dict[adj], test_feature_vector_dict[adj], test_adjective_dict[adj], scaler)
    
        all_ensemble_classifiers[adj] = ensemble_svm

        # Write classification reports into text file
        ensemble_report_file.write('Adjective:  '+adj+'\n')
        ensemble_report_file.write(ensemble_report)
        ensemble_report_file.write('\n\n')

        return all_ensemble_classifiers

# MAIN FUNCTION
def main(input_file, adjective_file, train_feature_pkl, test_feature_pkl, ensemble_test_feature_pkl, all_classifiers_pkl, scaler_pkl):


    # Load data into the pipeline. First check
    # for feature object pkl files
    print "Loading data from file\n"
    if train_feature_pkl == None or test_feature_pkl == None or ensemble_test_feature_pkl == None:
        # If no features, load data from either an
        # h5 and adjective file or directly from
        # a saved pkl file
        if input_file.endswith(".h5"):
            all_data = loadDataFromH5File(input_file, adjective_file)
        else:
            all_data = utilities.loadBoltObjFile(input_file)

        print "Loaded data\n"
        
    
        # Split the data into train and final test
        train_data, ensemble_test_data = utilities.split_data(all_data, 0.9)
        
        # Split the train data again into train and test
        train_data, test_data = utilities.split_data(train_data, 0.7)
        
        # Fit PCA for electrodes on training data
        print "Fitting PCA for electrode data\n"
        electrode_pca_dict = fit_electrodes_pca(train_data)
        
        # Store off PCA pkl
        cPickle.dump(electrode_pca_dict, open("pca.pkl","w"), cPickle.HIGHEST_PROTOCOL)
        print "PCA transforms stored as 'pca.pkl'\n"
                
        # Convert motion objects into feature objects
        print "Generating feature object dictionaries\n"
        train_all_features_obj_dict = BoltMotionObjToFeatureObj(train_data, electrode_pca_dict)
        test_all_features_obj_dict = BoltMotionObjToFeatureObj(test_data, electrode_pca_dict)
        ensemble_test_all_features_obj_dict = BoltMotionObjToFeatureObj(ensemble_test_data, electrode_pca_dict)
        
        # Store off feature object pkls
        cPickle.dump(train_all_features_obj_dict, open("train_feature_objs.pkl","w"), cPickle.HIGHEST_PROTOCOL)
        print "Feature object dictionary stored as 'train_feature_objs.pkl'\n"
        cPickle.dump(test_all_features_obj_dict, open("test_feature_objs.pkl","w"), cPickle.HIGHEST_PROTOCOL)
        print "Feature object dictionary stored as 'test_feature_objs.pkl'\n"
        cPickle.dump(ensemble_test_all_features_obj_dict, open("ensemble_test_feature_objs.pkl","w"), cPickle.HIGHEST_PROTOCOL)
        print "Feature object dictionary stored as 'ensemble_test_feature_objs.pkl'\n"
    
    else: 
        # Load pkl'd feature object dictionaries
        train_all_features_obj_dict = cPickle.load(open(train_feature_pkl,"r"))
        test_all_features_obj_dict = cPickle.load(open(test_feature_pkl,"r"))
        ensemble_test_all_features_obj_dict = cPickle.load(open(ensemble_test_feature_pkl,"r"))
        print "Loaded data\n"


    # Specify feature to be extracted
    feature_name_list = ["pdc_rise_count", "pdc_area", "pdc_max", "pac_energy", "pac_sc", "pac_sv", "pac_ss", "pac_sk", "tac_area", "tdc_exp_fit", "gripper_min", "gripper_mean", "transform_distance", "electrode_polyfit"]

    if all_classifiers_pkl == None or scaler_pkl == None:
        
        # Pull desired features from feature objects
        train_feature_vector_dict, train_adjective_dict = bolt_obj_2_feature_vector(train_all_features_obj_dict, feature_name_list)
        test_feature_vector_dict, test_adjective_dict = bolt_obj_2_feature_vector(test_all_features_obj_dict, feature_name_list)
        print("Created feature vector containing %s\n" % feature_name_list)

        # Create Scalers
        scaler_dict = create_scalers(train_feature_vector_dict)

        # Store off scaler dictionary
        cPickle.dump(scaler_dict, open("scaler.pkl","w"), cPickle.HIGHEST_PROTOCOL)
        print "Feature vector scalers stored as 'scaler.pkl'\n"

        # Run full train
        all_knn_classifiers, all_svm_classifiers = full_train(train_feature_vector_dict, train_adjective_dict, test_feature_vector_dict, test_adjective_dict, scaler_dict)

        # Select which algorithm to use in the ensemble phase
        all_classifiers_dict = all_svm_classifiers

    else:
        # Load pkl'd classifiers, probabilities and scores
        all_classifiers_dict = cPickle.load(open(all_classifiers_pkl,"r"))
        
        # Load pkl'd scaler dictionary
        scaler_dict = cPickle.load(open(scaler_pkl,"r"))

        # Get test labels, to be used as ensemble train labels
        test_all_features_obj_dict = cPickle.load(open(test_feature_pkl,"r"))
        test_feature_vector_dict, test_adjective_dict = bolt_obj_2_feature_vector(test_all_features_obj_dict, feature_name_list)


    
    # Pull desired bolt features from ensemble test data
    ensemble_test_feature_vector_dict, ensemble_test_adjective_dict = bolt_obj_2_feature_vector(ensemble_test_all_features_obj_dict, feature_name_list)

    # Create ensemble feature vectors out of probabilities
    ensemble_train_feature_vector_dict, ensemble_test_feature_vector_dict = extract_ensemble_features(all_classifiers_dict, ensemble_test_feature_vector_dict, ensemble_test_adjective_dict, scaler_dict)
    
    # Ensemble train labels are previous test labels
    ensemble_train_adjective_dict = test_adjective_dict
    
    import pdb; pdb.set_trace()
    for adj in ensemble_train_adjective_dict:
        count = np.sum(ensemble_train_adjective_dict[adj])
        import pdb; pdb.set_trace()
        print adj+":  %d " %count

    # Remove the adjectives 'warm' and 'sparse' from the labels dictionaries
    del ensemble_train_adjective_dict['springy']
    del ensemble_test_adjective_dict['springy']
    del ensemble_train_adjective_dict['elastic']
    del ensemble_test_adjective_dict['elastic']
    del ensemble_train_adjective_dict['meshy']
    del ensemble_test_adjective_dict['meshy']
    del ensemble_train_adjective_dict['gritty']
    del ensemble_test_adjective_dict['gritty']
    del ensemble_train_adjective_dict['warm']
    del ensemble_test_adjective_dict['warm']
    del ensemble_train_adjective_dict['textured']
    del ensemble_test_adjective_dict['textured']
    del ensemble_train_adjective_dict['absorbant']
    del ensemble_test_adjective_dict['absorbant']
    del ensemble_train_adjective_dict['crinkly']
    del ensemble_test_adjective_dict['crinkly']
    del ensemble_train_adjective_dict['porous']
    del ensemble_test_adjective_dict['porous']
    del ensemble_train_adjective_dict['grainy']
    del ensemble_test_adjective_dict['grainy']
    del ensemble_train_adjective_dict['sparse']
    del ensemble_test_adjective_dict['sparse']
    

    # Combine motion-specific classifiers for each adjective  
    all_ensemble_classifiers = full_ensemble_train(ensemble_train_feature_vector_dict, ensemble_train_adjective_dict, ensemble_test_feature_vector_dict, ensemble_test_adjective_dict)
    
    # Store off combined classifiers
    cPickle.dump(all_ensemble_classifiers, open("all_ensemble_classifiers.pkl","w"), cPickle.HIGHEST_PROTOCOL)

# Parse the command line arguments
def parse_arguments():
    """
    Parses the arguments provided at command line.
    
    Returns:
    (input_file, adjective_file, range)
    """
    parser = OptionParser()
    parser.add_option("-i", "--input_file", action="store", type="string", dest = "in_h5_file")
    parser.add_option("-o", "--output", action="store", type="string", dest = "out_file", default = None) 
    parser.add_option("-a", "--input_adjective", action="store", type="string", dest = "in_adjective_file")
    parser.add_option("-n", "--input_train_feature_pkl", action="store", type="string", dest = "in_train_feature_pkl", default = None)
    parser.add_option("-s", "--input_test_feature_pkl", action="store", type="string", dest = "in_test_feature_pkl", default = None)
    parser.add_option("-e", "--input_ensemble_test_feature_pkl", action="store", type="string", dest = "in_ensemble_test_feature_pkl", default = None)
    parser.add_option("-c", "--input_all_classifiers_pkl", action="store", type="string", dest = "in_all_classifiers_pkl", default = None)
    parser.add_option("-k", "--input_scaler_pkl", action="store", type="string", dest = "in_scaler_pkl", default = None)

    (options, args) = parser.parse_args()
    input_file = options.in_h5_file #this is required
   
    #if options.out_file is None:
        #(_, name) = os.path.split(input_file)
        #name = name.split(".")[0]
        #out_file = name + ".pkl"
    #else:    
        #out_file = options.out_file
        #if len(out_file.split(".")) == 1:
            #out_file = out_file + ".pkl"
    out_file = options.out_file

    adjective_file = options.in_adjective_file

    train_feature_pkl = options.in_train_feature_pkl
    test_feature_pkl = options.in_test_feature_pkl
    ensemble_test_feature_pkl = options.in_ensemble_test_feature_pkl

    all_classifiers_pkl = options.in_all_classifiers_pkl
    scaler_pkl = options.in_scaler_pkl

    return input_file, out_file, adjective_file, train_feature_pkl, test_feature_pkl, ensemble_test_feature_pkl, all_classifiers_pkl, scaler_pkl


if __name__ == "__main__":
    input_file, out_file, adjective_file, train_feature_pkl, test_feature_pkl, ensemble_test_feature_pkl, all_classifiers_pkl, scaler_pkl = parse_arguments()
    main(input_file, adjective_file, train_feature_pkl, test_feature_pkl, ensemble_test_feature_pkl, all_classifiers_pkl, scaler_pkl)
