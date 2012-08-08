#!/usr/bin/env python

#modified from code sumbitted as an answer to an audio_common bug ticket
#by user kstrabala
#https://kforge.ros.org/audiocommon/trac/ticket/1

#Edited by Ian McMahon

import roslib; roslib.load_manifest('extract_bag_audio')
import rospy, sys
from ros import rosbag

def extract_audio(bag_path, topic_name, mp3_path):
    rospy.loginfo('Opening bag %s' % bag_path)
    bag = rosbag.Bag(bag_path)
    mp3_file = open(mp3_path, 'w')
    rospy.loginfo('Reading audio messages and saving to mp3 file: %s' % mp3_file)
    msg_count = 0
    for topic, msg, stamp in bag.read_messages(topics=[topic_name]):
        if msg._type == 'audio_common_msgs/AudioData':
            msg_count += 1
            mp3_file.write(''.join(msg.data))
    bag.close()
    mp3_file.close()
    rospy.loginfo('Done. %d audio messages written to %s'%(msg_count, mp3_path))

if __name__ == '__main__':
  rospy.init_node('audio_to_mp3')
  arguments = rospy.myargv(argv=sys.argv)
  try:
    bag_path = arguments[1]
    topic_name = arguments[2]
    mp3_path = arguments[3]
    extract_audio(bag_path, topic_name, mp3_path)
  except:
    rospy.logerr('Usage: rosrun extract_bag_audio audio_to_mp3.py <full_bagfile_path> /<topic_name> <full_mp3_file_path>')
