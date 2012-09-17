#!/usr/bin/env python
import roslib; roslib.load_manifest("hadjective_train_pipe")
import rospy
import numpy as np
import bolt_learning_utilities as utilities
import train_pipe as train
import cPickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from optparse import OptionParser
from sklearn import preprocessing
from hcluster import pdist, linkage, dendrogram
from scipy.cluster import hierarchy


def DrawDendrogram(feature_vector, obj_names, motion_name):
    distances = pdist(feature_vector)
    linkage_list = ['single', 'average', 'complete']
    Z = linkage(distances, linkage_list[1])
    render = hierarchy.dendrogram(Z,
                                  #p=51,
                                  #truncate_mode='level',
                                  #show_contracted=True,
                                  color_threshold=1.5,
                                  labels=obj_names,
                                  orientation='left',
                                  show_leaf_counts=True,
                                  leaf_font_size=10,
                                  )

    plt.title(motion_name+'_'+linkage_list[1])
    plt.show()
    #plt.savefig(motion_name+'_dendro_complete.png')

    return render


# MAIN FUNCTION
def main(bolt_motion_obj_file, bolt_feature_obj_file):
    
    # Load data
    print "\nLoading data from file\n"
    if bolt_feature_obj_file == None:
        all_data = utilities.loadBoltObjFile(bolt_motion_obj_file)
        print "Loaded data\n"

        # Fit PCA for electrodes on all the data
        print "Fitting PCA for electrode data\n"
        electrode_pca_dict = train.fit_electrodes_pca(all_data)

        # Convert motion objects into feature objects
        print "Generating feature object dictionary\n"
        all_features_obj_dict = train.BoltMotionObjToFeatureObj(all_data, electrode_pca_dict)

        # Store off feature object pkls
        cPickle.dump(all_features_obj_dict, open("all_feature_objs.pkl","w"), cPickle.HIGHEST_PROTOCOL)
        print "Feature object dictionary stored as 'all_feature_objs.pkl'\n"
    else:
        all_features_obj_dict = cPickle.load(open(bolt_feature_obj_file,"r"))
        print "Loaded data\n"

    # Specify features to be extracted
    feature_name_list = ["pdc_rise_count", "pdc_area", "pdc_max", "pac_energy", "pac_sc", "pac_sv", "pac_ss", "pac_sk", "tac_area", "tdc_exp_fit", "gripper_min", "gripper_mean", "transform_distance", "electrode_polyfit"]

    #import pdb; pdb.set_trace()
    #pass

    # Pull desired features from feature objects
    feature_vector_dict, adjective_label_dict = train.bolt_obj_2_feature_vector(all_features_obj_dict, feature_name_list)
    print "Created feature vector containing %s\n" % feature_name_list

    # Here is where the feature vectors for each motion need to be transformed from 510x50 to 1020x25. Be
    # careful to split based on finger! I believe every other column goes together. (So columns 1:2:49 are
    # finger 1 and columns 2:2:50 are finger 2) 
    

    # For each motion, run K-means on features
    #report = open("Trial_kmeans_15.txt", "w")
    num_objects = 51
    num_adjectives = 34

    # Create a list of object names as the labels for dendrogram
    obj_names = []
    for i in np.arange(len(all_features_obj_dict['tap'])):
        obj_names.append(all_features_obj_dict['tap'][i].name)


    num_clusters = 15
    for motion_name in feature_vector_dict:
        # We need to modify the run_kmeans() function so that it can take in a feature_vector argument that is
        # twice as long as the all_data argument!

        # Preprocess the feature_vector
        feature_vector = preprocessing.scale(feature_vector_dict[motion_name])

        #clusteirs, cluster_names, cluster_all_adjectives = train.run_kmeans(feature_vector, num_clusters, all_features_obj_dict[motion_name])

        Render = DrawDendrogram(feature_vector, np.array(obj_names), motion_name)
        
        import pdb; pdb.set_trace()
        pass

        '''
        # Save the kmeans results into txt files
        report.write('Motion: '+motion_name+'\n')
        for i in np.arange(num_clusters):
            size_cluster = len(clusters[i])
            report.write('Number of trials in cluster '+str(i)+': '+str(size_cluster)+'\n')
            report.write(str(cluster_names[i])+'\n\n')
        report.write('\n\n')
        '''
    
    #report.close()



    #import pdb; pdb.set_trace()
    #pass

    # All that's left now is to look at which trials were placed in which cluster =)

# Parse the command line arguments
def parse_arguments():
    
    parser = OptionParser()
    parser.add_option("-m", "--input_motion_obj_file", action="store", type="string", dest="in_motion_obj_file", default=None)
    parser.add_option("-f", "--input_feature_obj_file", action="store", type="string", dest="in_feature_obj_file", default=None)

    (options, args) = parser.parse_args()
    bolt_motion_obj_file = options.in_motion_obj_file
    bolt_feature_obj_file = options.in_feature_obj_file

    return bolt_motion_obj_file, bolt_feature_obj_file


if __name__ == "__main__":
    bolt_motion_obj_file, bolt_feature_obj_file = parse_arguments()
    main(bolt_motion_obj_file, bolt_feature_obj_file)



