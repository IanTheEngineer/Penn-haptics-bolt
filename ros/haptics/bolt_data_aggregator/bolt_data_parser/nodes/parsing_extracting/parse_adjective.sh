#! /bin/bash

ADJECTIVES_H5=$1
DEST_FOLDER=$2


ALL_ADJECTIVES="absorbant porous hard hollow springy squishy rough thick compact scratchy elastic unpleasant plasticky meshy nice hairy compressible fibrous deformable metalic warm textured bumpy fuzzy sticky cool grainy gritty stiff solid crinkly smooth slippery thin sparse soft"
for CATEGORY in $ALL_ADJECTIVES; do
	echo "Doing adjective " $CATEGORY
	FILES=$(./list_files_attributes.py $ADJECTIVES_H5 $CATEGORY==1)
	NUM_FILES=$(echo $FILES | wc -w)
	echo Number of files: $NUM_FILES
	../parse_bag_pytables.py $FILES $DEST_FOLDER/$CATEGORY.h5
	echo
	echo
	echo
done
