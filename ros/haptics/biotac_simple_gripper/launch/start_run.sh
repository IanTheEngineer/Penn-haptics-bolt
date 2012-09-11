#!/bin/bash

# Creates output directories if they don't exist, then launches the start_run.launch

if [ -z "$1" ];
  then
    echo "Warning: No output filename provided"
fi

#if [ -z "$ROS_WORKSPACE" ]; then
#    echo "You must set to set ROS_WORKSPACE to create BOLT data dumping directories. Aborting..."
#    exit 1
#fi

#BOLT_DATA_DIR="$ROS_WORKSPACE/bolt_haptic_data"
BOLT_DATA_DIR="/removable/bolt_haptic_data"

if [ ! -d "$BOLT_DATA_DIR" ]; then
  mkdir $BOLT_DATA_DIR
fi

JSON_DIR="$BOLT_DATA_DIR/json_files"

if [ ! -d "$JSON_DIR" ]; then
  mkdir $JSON_DIR
fi

BAG_DIR="$BOLT_DATA_DIR/bag_files"

if [ ! -d "$BAG_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir $BAG_DIR
fi

if [ -z "$1" ];
  then
    LAUNCH="roslaunch biotac_simple_gripper start_run_no_record.launch filename:=$1 data_path:=$BOLT_DATA_DIR"
    echo "Warning: No output filename provided"

else
    LAUNCH="roslaunch biotac_simple_gripper start_run.launch filename:=$1 data_path:=$BOLT_DATA_DIR"
fi


eval $LAUNCH
