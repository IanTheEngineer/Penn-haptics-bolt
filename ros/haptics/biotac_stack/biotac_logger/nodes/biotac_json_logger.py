#!/usr/bin/env python
# Copyright (c) 2012, University of Pennsylvania
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of Pennsylvania nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Authors: Vivian Chu (chuv@grasp.upenn.edu) 
#          Ian McMahon (imcmahon@grasp.upenn.edu)


import roslib; roslib.load_manifest('biotac_logger')
import rospy
import os
import rosjson_time

from std_msgs.msg import String
from biotac_sensors.msg import BioTacHand

class BioTacListener:

  def __init__(self):
    self.frame_count = 1;
    rospy.init_node('biotac_json_logger')
   
    # FILE WRITING SETUP 
    # Find Node Parameter Name
    self.file_param = rospy.get_name() + '/filename'
    # Grab directory
    self.package_dir = roslib.packages.get_pkg_dir('biotac_logger')
    # Check for a 'data' directory
    dir_status = self.check_dir( self.package_dir + '/data' )
    if dir_status:
      rospy.loginfo('The ''data'' directory was successfully created.')
    # Set output filename
    self.fileName =  self.package_dir + '/data/' + rospy.get_param(self.file_param,'default.json')
    # Create initial file - delete existing file with same name 
    self.fout = open(self.fileName,'w')
    self.fout.write("[\n")

    rospy.loginfo(rospy.get_name()+' Starting to Log to file %s:',self.fileName);
    
  # Called each time there is a new message
  def biotacCallback(self,data):
    #rospy.loginfo(rospy.get_name()+' FRAME ID: %d',self.frame_count)

    # Stores the frame count into the message
    data.header.frame_id = self.frame_count

    # Uses rosjson to convert message to JSON 
    toWrite = rosjson_time.ros_message_to_json(data) + '\n'
    self.fout.write(toWrite); 
       
    # Move to next frame 
    self.frame_count += 1

  #Check if directory exits & create it
  def check_dir(self, f):
    if not os.path.exists(f):
      os.makedirs(f)
      return True
    return False

  # Setup the subscriber Node
  def listener(self):
    rospy.Subscriber('biotac_pub', BioTacHand, self.biotacCallback,queue_size=1000)
    rospy.spin()

  def __del__(self):
    self.fout.write("]")
    self.fout.close()


if __name__ == '__main__':

    bt_listener = BioTacListener()
    bt_listener.listener()
