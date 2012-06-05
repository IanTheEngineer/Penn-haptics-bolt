#!/usr/bin/env python
import roslib; roslib.load_manifest('biotac_logger')
import rospy
import os
import rosjson

from std_msgs.msg import String
from biotac_sensors.msg import BioTacHand

class BioTacListener:

  def __init__(self):
    self.frame_count = 1;
  # Called each time there is a new message
  def biotacCallback(self,data):
    rospy.loginfo(rospy.get_name()+' FRAME ID: %d',self.frame_count)
    # Open existing file and append to it 
    fout = open(self.fileName,"a")

    # Stores the frame count into the message
    data.header.frame_id = self.frame_count

    # Uses rosjson to convert message to JSON 
    toWrite = rosjson.ros_message_to_json(data) + '\n'
    fout.write(toWrite); 
    fout.close()
       
    # Move to next frame 
    self.frame_count += 1

  #Check if directory exits & create it
  def check_dir(self, f):
    #d = os.path.dirname(f)
    if not os.path.exists(f):
      os.makedirs(f)

  # Setup the subscriber Node
  def listener(self):
    # Initialize Node 
    rospy.init_node('biotac_json_logger', anonymous=True)
    rospy.Subscriber('biotac_pub', BioTacHand, self.biotacCallback,queue_size=1000)

    # Find Node Parameter Name
    self.file_param = rospy.get_name() + '/filename'
    # Grab directory
    self.package_dir = roslib.packages.get_pkg_dir('biotac_json_logger')
    # Check for a 'data' directory
    self.check_dir( self.package_dir + '/data' )
    # Set output filename
    self.fileName =  self.package_dir + '/data/' + rospy.get_param(self.file_param,'default.json')
    # Create initial file - delete existing file with same name 
    fout = open(self.fileName,'w');
    fout.close();
 
    rospy.spin()

if __name__ == '__main__':

    bt_listener = BioTacListener()
    bt_listener.listener()
