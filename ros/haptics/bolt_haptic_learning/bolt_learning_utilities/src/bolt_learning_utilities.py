#!/usr/bin/env python
import roslib; roslib.load_manifest("bolt_learning_utilities")
import rospy
import numpy as np
import cPickle
import convert_h5_dataset_to_bolt_obj as converth5
import process_adjective_labels
from random import shuffle

from bolt_feature_obj import BoltFeatureObj
# Convert h5 files to BolPR2MotionObj data
def convertH5ToBoltObjFile(input_file, output_file, save_to_file):
    """
    Convert data in h5 file into a single pkl file that 
    stores the information in a BoltPR2MotionObj.  
    The runs are segmented by motion and stored in a dictionary.

    Usage: convertH5ToBoltObjFile(input_file, output_file, save_to_file)
   
    Note: The function returns a dictionary that can also be saved directly
          to a pickle file.  However, if save_to_file is False, it will not
          be saved and it is up to the user to store the returned object
 
    input_file - .h5
    output_file - .pkl
    save_to_file - boolean

    Output structure:
        
        Dictionary -> "squeeze", "thermal_hold", "tap", "slide",
                      "slow_slide"

        "squeeze" -> list(BoltPR2MotionObjs)

    """

    # Call the function
    loaded_data = converth5.load_data(input_file, output_file, save_to_file)

    return loaded_data


# Load bolt obj pickle file
def loadBoltObjFile(input_file):
    """
    Takes a path to a pkl file, loads, and returns the dictionary.  See
    convertH5ToBoltObjFile for the data structure returned
    """
    # File checking
    if not input_file.endswith(".pkl"):
        raise Exception("Input file: %s is not a pkl file.\n Please pass in pkl file with extention .pkl" % input_file)
    
    # Load the file 
    file_ptr = open(input_file, "r")
    loaded_data = cPickle.load(file_ptr)

    return loaded_data



# Insert adjective labels 
def insertAdjectiveLabels(input_boltPR2Motion, output_file, adjective_file, save_to_file):
    """
    Adjective label file is given as a pytable in h5 data

    Given either a path to a pkl file or the direct dictionary of 
    boltPR2MotionObj, will insert the labels
    
    if input_boltPR2Motion is a file, the function will first load the
    file before inserting adjectives

    """
    
    if isinstance(input_boltPR2Motion, str):
        data = process_adjective_labels.load_files(input_boltPR2Motion)
    else:
        data = input_boltPR2Motion

    data_with_adjectives = process_adjective_labels.populate_BoltPR2MotionObj(data, output_file, adjective_file, save_to_file)

    return data_with_adjectives



# Normalizes the given bolt_obj.  Works directly on the object
def normalize_data(bolt_obj, discard_raw_flag = True):
    """ 
    Given a BOLTPR2MotionObj 
    Normalize the data
        - Takes the mean and subtract 
        - Adds a pac_flat field

    Usage: normalize(bolt_obj, discard_flag)
        
        Discard flag is optional - default is True
        Discard flag set False will keep the original raw
        data, which makes the data twice as large
    """
    
    num_fingers = len(bolt_obj.electrodes)

    # For each finger normalize and store
    for finger in xrange(num_fingers):
        # Electrodes
        electrodes = bolt_obj.electrodes[finger]
        electrodes_mean = bolt_obj.electrodes_mean[finger]
        bolt_obj.electrodes_normalized.append(-(electrodes - np.mean(electrodes_mean, axis = 0)))

        # PDC
        pdc = bolt_obj.pdc[finger]
        pdc_mean = bolt_obj.pdc_mean[finger]
        bolt_obj.pdc_normalized.append(pdc - np.mean(pdc_mean))

        # TDC
        tdc = bolt_obj.tdc[finger]
        tdc_mean = bolt_obj.tdc_mean[finger]
        bolt_obj.tdc_normalized.append(tdc - np.mean(tdc_mean))

        # TAC
        tac = bolt_obj.tac[finger]
        tac_mean = bolt_obj.tac_mean[finger]
        bolt_obj.tac_normalized.append(-(tac - np.mean(tac_mean)))

        # PAC
        pac = bolt_obj.pac[finger]
        pac_mean = bolt_obj.pac_mean[finger]
        pac_norm = -(pac - np.mean(pac_mean, axis = 0)) 
        bolt_obj.pac_normalized.append(pac_norm)

        # Flatten PAC
        bolt_obj.pac_flat.append( pac.reshape(1,len(pac)*22)[0])
        bolt_obj.pac_flat_normalized.append( pac_norm.reshape(1, len(pac_norm)*22)[0])

    if discard_raw_flag:
        # Clear out raw values - comment out later if they want to be stored
        # Will double the amount of data stored
        del bolt_obj.pdc[:]
        del bolt_obj.electrodes[:]
        del bolt_obj.pac[:]
        del bolt_obj.tdc[:]
        del bolt_obj.tac[:]
        del bolt_obj.pac_flat[:]

    
# Pull from the BoltFeatureObj and return a vector of features and labels
def createFeatureVector(bolt_feature_obj, feature_list):
    """ 
    Given a BoltFeatureObj and a list of features to pull, returns
    the features and labels in vector form for all adjectives

    feature_list: expects a list of strings containing the name of the
                  features to extract
                  
                  Ex. ["max_pdc", "centroid"]

    Returns: Vector of features (1 x feature length)
             Features - in order of feature_list

             Ex. max_pdc = 10, centroid = [14, 10]

             returned numpy array: [10 14 10]

    """
    # Store the features with their adjectives 
    all_feature_vector = list()
    
    # Pull out the features from the object
    for feature in feature_list:
        
        feature_vector = eval('bolt_feature_obj.'+ feature)
        if feature == "electrode_polyfit":
            for idx in range(0,len(feature_vector[0])):
                all_feature_vector += [feature_vector[0][idx], feature_vector[1][idx]]
        else:
            all_feature_vector += feature_vector

    return (np.array(all_feature_vector), bolt_feature_obj.object_id)

# Function to split the data
def split_data(all_bolt_data, train_size):
    """
    Given a dictionary of all bolt objects

    Splits the objects into train and test sets

    train_size is a percentage - Ex. train_size = .9
    """

    train_bolt_data = dict()
    test_bolt_data = dict()
  
    # Calculate train size
    num_runs = len(all_bolt_data[all_bolt_data.keys()[0]])
    train_size = num_runs*train_size
    train_size = int(round(train_size))
    
    # Create list to shuffle and index into objects
    access_idx = range(num_runs)
    shuffle(access_idx)
    access_np_idx = np.array(access_idx)
    
    train_idx = np.nonzero(access_np_idx <= train_size)[0]
    test_idx = np.nonzero(access_np_idx > train_size)[0] 

    # Go through the list of motions
    for motion_name in all_bolt_data:
        motion_list = all_bolt_data[motion_name]

        train_set = [motion_list[i] for i in train_idx]
        test_set = [motion_list[i] for i in test_idx]
 
        train_bolt_data[motion_name] = train_set
        test_bolt_data[motion_name] = test_set
    
    return (train_bolt_data, test_bolt_data)

# Compute precision, recall and f1 score
def compute_statistics(predicted_label, truth_label):
    '''
    Given the predicted label from a supervised machine
    learning classifier, will return metrics about 
    how well the classifier performed.

    Recall - measure of how many true positives pulled
             from the dataset.
             Ex. if there are 20 objects that are "soft"
                 and only 15 are found, the recall is
                 15/20 = 75%

    Precision - measure of how accurate the classifier
                was (number of false negatives found)
                Ex. if the classifier returned 10 objects
                    that are soft and only 6 of them are
                    actually soft, the precision is
                    6/10 = 60%

    F1 score -  combining recall and precision into one
                score.  F1 score is the equally weighted
                mean of the two
    
    NOTE: This function makes the assumption that the
          labels are binary (1 and 0)
    '''

    # Subtract the true label from the predicted label
    label_sub = predicted_label - truth_label

    # Get total number of objects that were labeled true
    num_objects_found = float(sum(predicted_label))
    
    # Get total number of objects that should be true
    num_objects_truth = float(sum(truth_label))

    # If there is a -1 in the label_sub - that means that
    # the classifier missed labeling the object as true
    # This is used for recall
    num_objects_missed = float(np.shape(np.nonzero(label_sub == -1)[0])[0]) 

    # If there is a 1 in the label_sub - that means that
    # the classifier returned true for that object
    # when it should have been false - this make it
    # a false positive
    num_objects_wrong = float(np.shape(np.nonzero(label_sub == 1)[0])[0])

    import pdb; pdb.set_trace()
    # Calculate precision
    precision = (num_objects_found-num_objects_wrong)/num_objects_found

    # Calculate recall
    recall = (num_objects_truth-num_objects_missed)/num_objects_truth

    # Calculate fscore
    f1 = 2* ((precision * recall)/(precision + recall))

    return precision, recall, f1

 
# Create a function that given a classifier and a test
# vector, will predict the labels
def compute_adjective_probability_score(adj_classifiers, test_feature_obj, feature_list, adj_name, scaler_dict):
    '''
    The function expects the input to be a loaded classifier for a SINGLE adjective
   
    Inputs: Classifier dictionary for one single adjective for one motion
            Test feature object to be called
            output_type - the options are: 'Classify', 'Probability', 'f1'
                        - this tells the function which score it should return

    The output is a 1 x n vector (where n is the number of motions) of scores specified
    by output_type
    '''
    # Store dictionary of strings
    state_string = {test_feature_obj.DISABLED:'disabled',
                    test_feature_obj.THERMAL_HOLD:'thermal_hold',
                    test_feature_obj.SLIDE:'slide',
                    test_feature_obj.SQUEEZE:'squeeze',
                    test_feature_obj.TAP:'tap',
                    test_feature_obj.DONE:'done',
                    test_feature_obj.SLIDE_FAST:'slide_fast',
                    test_feature_obj.CENTER_GRIPPER:'center_gripper'
                    }   

    # Pull out the classifier for this motion
    motion_name = state_string[test_feature_obj.state]
    classifier_tuple = adj_classifiers[state_string[test_feature_obj.state]]
    classifier = classifier_tuple[0]
    scaler = scaler_dict[motion_name] 

    # Take the test feature objects and convert into a feature vector based on
    # the passed in list
    test_vector,obj_id = createFeatureVector(test_feature_obj, feature_list)

    # Scale the test_vector
    test_vector_scaled = scaler.transform(test_vector)

    #import pdb; pdb.set_trace()
    # Grab the probability of the adjective being TRUE (1)  This is the second value returned
    # from the svm
    prob_score = classifier.predict_proba(test_vector_scaled)[0][1]

    '''
    print "Object is: %s" % test_feature_obj.name
    print "Classified as %d" % classifier.predict(test_vector_scaled)[0]
    print "How certain is %f" % prob_score
    '''

    if  test_feature_obj.labels == None:
        return (classifier.predict(test_vector_scaled)[0], prob_score)
    else: 
        return (classifier.predict(test_vector_scaled)[0], prob_score, test_feature_obj.labels[adj_name])
         
       
# Function that pulls the motion that is best based on the dictionary of what scores each motion had
def get_best_motion(motion_score_dictionary):
    '''
    Given a dictionary of scores where the keys are the motions, return the motion and score that is best
    '''

    best_motion = ""
    best_score = -1.0

    for motion in motion_score_dictionary:
        score = motion_score_dictionary[motion]
        
        if score > best_score:
            best_score = score
            best_motion = motion

    return (best_motion, best_score)
