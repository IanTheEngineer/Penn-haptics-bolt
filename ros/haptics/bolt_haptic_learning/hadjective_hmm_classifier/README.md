Creating the h5 databases
=========================
The first steps are to create appropriate h5 databases that store all the information
needed for training. The scripts in 
Penn-haptics-bolt/ros/haptics/bolt_data_aggregator/bolt_data_parser/nodes/parsing_extracting
will do this job.

NOTE: since the databases are very large, I can provide the h5 file if you give me
a pointer to free space (around 500Mb). This will make the whole process faster.

Training the HMMs
=================
The main file that performs training is train_chain_database.py. Depending on
the parameters it will train one adjective, phase or sensors. The approriate h5
databases must be created beforhand.

This is a VERY SLOW procedure and it will likely take 1-2 days on a desktop computer.

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