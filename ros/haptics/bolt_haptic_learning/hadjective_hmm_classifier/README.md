Creating the h5 databases
=========================
The first steps are to create appropriate h5 databases that store all the information
needed for training. The scripts in 
Penn-haptics-bolt/ros/haptics/bolt_data_aggregator/bolt_data_parser/nodes/parsing_extracting
will do this job.

NOTE: since the databases are very large, I can provide the h5 file if you give me
a pointer to freespace (around 500Mb). This will make the whole process faster.

To create a database:
./parse_bad_pytables *.bag output.h5 ->initial database
./cvs_to_h5.py adjectives.cvs adjectives.h5 -> converts the adjectives (as as cvs) to a h5 file
./creat_links_in_h5 adjectives.h5 output.h5 -> assembles the adjectives per each object

In utilities.py there is the function  create_train_test_set(adjective_group, training_ratio).
That is used to split training and testing set. Training ratio is a floating point number (e.g. 2./3)

At the end of the process you will have the h5 file ready to be used for training.

Training the HMMs
=================
The main file that performs training is **train_chain_database.py**. Depending on
the parameters it will train one adjective, phase or sensors. The approriate h5
databases must be created beforhand.

This is a VERY SLOW procedure and it will likely take 1-2 days on a desktop computer.

**very important**: the script assumes that the destination folder has a subfolder called chains.
That is, if invoked with: ./train_chain_database.py db.h5 /home/joe/results, results will *already* have
a subfolder named chains.

Example execution:

To train all the adjectives, all the sensors, all the motion phases:
./train_chain_database.py /path/to/database.h5 /path/where/to/store/chains

To train the ajective *smooth*, all the sensors, all the motion phases:
./train_chain_database.py /path/to/database.h5 /path/where/to/store/chains smooth

To Train all the adjectives for phase HOLD_FOR_10_SECONDS and sensor pac:
./train_chain_database.py /path/to/database.h5 /path/where/to/store/chains HOLD_FOR_10_SECONDS pac

To Train adjective smooth for phase HOLD_FOR_10_SECONDS and sensor pac:
./train_chain_database.py /path/to/database.h5 /path/where/to/store/chains smooth HOLD_FOR_10_SECONDS pac


Creating untrained classifiers
==============================
The second step is to create the classifiers. These will not be trained yet, only
saved in <base_folder>/untrained_adjectives. The script that handles this part
is create_classifiers.py.

Training the classifiers
========================
Once the HMMs are created the SVM classifier can be trained. The script that
performs this is in train_classifier.py. It assumes that all the hmms have been
trained and the classifiers stored in <base_folder>/untrained_adjectives. 
The resulting adjectives will be stored in <base_folder>/trained_adjectives.

Testing the classifiers
=======================
The file test_classifiers.py uses the trained classifiers and an h5 database to
measure the scores. Again the database needs to have the "adjectives" group
created using the script:
Penn-haptics-bolt/ros/haptics/bolt_data_aggregator/bolt_data_parser/nodes/parsing_extracting/create_adjectives_links.py

Creating Static Features
========================
The first thing that needs to be done is the features created from all of the objects in the database.  This is done with create_static_features.py

It will take in path to database and a path to a directory where there must exist a folder datasets. THIS NEEDS TO BE CREATED BEFORE THE SCRIPT IS RUN.  

This function assumes that utilities_static_save.py has been renamed to utilities.py

The result should be files that have the features for each motion in its own separate file.  Be patient as this step takes some time.

Extracting Static Features
==========================
Once the features are stored in datasets, then calling extract_static_features.py will create a train/test set from these features for each adjective.  This depends on the h5 database's split on train-test.

currently it only supports positive splits and all negative exapmles are in both train/test.  This will be changed once the h5 database reflects the new category for negative splits

The features pulled must be specified in the utilities.py file.  The order in which the features are listed is the order that they are extracted.  The variable specifically is static_features

The files are stored in a directory adjective_phase_set.  Similar to before this folder MUST BE CREATED ahead of time.

The results are stored by adjective-motion

Loading Static Features
=======================
Currently there is a dummy function load_static_features.py that provides an example of how to load the specific features.

