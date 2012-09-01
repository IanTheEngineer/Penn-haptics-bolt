#!/usr/bin/env python


def callback(msg):

def listener():
    rospy.init_node('hadjective_speech_node')
    rospy.Subscriber("hadj_speech", String, callback)#subscribes to hadjective_speech topic
    rospy.spin()

def main(argv):
    listener()


#http://en.wikibooks.org/wiki/Python_Programming/Input_and_output
#raw_input
if __name__ == '__main__':
    main(sys.argv[1:])

