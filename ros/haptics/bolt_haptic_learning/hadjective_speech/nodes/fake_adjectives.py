#!/usr/bin/env python
import roslib; roslib.load_manifest('hadjective_speech')
import rospy
from hadjective_speech.msg import AdjList
from hadjective_speech.msg import Adj


def talker():
      pub = rospy.Publisher('hadjective_speech', AdjList)
      rospy.init_node('hadjectives')
      #while not rospy.is_shutdown():
      some_adj = AdjList()
      adj_array = [Adj("soft"), Adj("compressible"), Adj("hairy"), Adj("scratchy")]
      some_adj.adjectives = adj_array
      #rospy.loginfo(str)
      pub.publish(some_adj)
      rospy.sleep(1.0)

if __name__ == '__main__':
  try:
    talker()
  except rospy.ROSInterruptException:
    pass
