#!/usr/bin/env python
import roslib; roslib.load_manifest('hadjective_speech')
import rospy
from hadjective_speech.msg import AdjProbDict
from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient
import os

def callback(msg):
    pass

def listener():
    rospy.Subscriber("hadj_speech", AdjProbDict, callback)#subscribes to hadjective_speech topic
    rospy.spin()

def main(argv):
    listener()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

    def set_green(self, stringy):
        return self.OKGREEN + stringy + self.ENDC

    def set_blue(self, stringy):
        return self.OKBLUE + stringy + self.ENDC

#http://en.wikibooks.org/wiki/Python_Programming/Input_and_output
#raw_input
if __name__ == '__main__':
    rospy.init_node('hadjective_speech_node')
    soundhandle = SoundClient()
    voice = 'voice_en1_mbrola'
    soundhandle.say('',voice)
    os.system("clear")
    color = bcolors()
    rospy.sleep(2)

    query = 'What is your name?'
    soundhandle.say(query,voice)
    response = raw_input( color.set_green(query) + color.set_blue("\n>> ") )

    #import pdb; pdb.set_trace()
    s = "It's a pleasure to meet you, %s. I am GRASPY, the PR2 roe-bot." % response
    soundhandle.say(s,voice)
    s = "It's a pleasure to meet you, %s. I am GRASPY, the PR2 robot." % response
    print color.set_green(s)
    rospy.sleep(5)

    iteration = []
    first_time = ['', '']
    next_time = [', again', 'other']
    iteration = [first_time, next_time]
    i=0
    while not rospy.is_shutdown():
      if i > 0:
        i = 1

      query = 'Shall we begin%s?' %iteration[i][0]
      soundhandle.say(query,voice)
      response = raw_input( color.set_green(query) + color.set_blue("\n>> ") )

      if response.lower().find('yes') is not -1 or response.lower().find('sure') is not -1:
        s = "Excellent! May I have an%s ob-ject to touch?"%iteration[i][1] # Tell me when you are ready."
        soundhandle.say(s,voice)
        s = "Excellent! May I have an%s object to touch?" %iteration[i][1] #\nTell me when you are ready."
        print color.set_green(s)
        rospy.sleep(4)
        query = "Tell me when you are ready."
        soundhandle.say(query,voice)
        response = raw_input(color.set_green(query) + color.set_blue("\n>> ") )
        i = i + 1
      else:
        s = "That's fine. I'm not going anywhere. I've got all the time in the world."
        soundhandle.say(s,voice)
        print color.set_green(s)
        rospy.sleep(10)

        #main(sys.argv[1:])

