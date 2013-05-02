#!/usr/bin/env python
import roslib; roslib.load_manifest('hadjective_speech')
import rospy
from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient
from hadjective_speech.msg import AdjList 
import os

voice = 'voice_en1_mbrola'

class HadjSpeechDemo:
  def __init__(self):
     rospy.init_node('hadjective_speech_node')
     self.soundhandle = SoundClient()
     self.voice = 'voice_en1_mbrola'
     self.soundhandle.say('',voice)
     os.system("clear")
     self.color = bcolors()
     self.main()
     

  def callback(self, msg):
      #import pdb; pdb.set_trace()
      #rospy.loginfo(rospy.get_name() + ": I heard %s" % msg.data)
      felt_msg = "This ob-ject feels"
      print_msg = "This object feels"
      for idx, adj in enumerate(msg.adjectives):
        if idx == len(msg.adjectives) - 1:
          felt_msg = felt_msg + " and " + str(adj.adj) + "."
          print_msg = print_msg + " and " + str(adj.adj) + "."
        else:
          felt_msg = felt_msg + " " + str(adj.adj) + ","
          print_msg = print_msg + " " + str(adj.adj) + ","
      self.soundhandle.say(felt_msg,self.voice)
      print self.color.set_green(print_msg)

  def listener(self):
      rospy.Subscriber("hadjective_speech", AdjList, self.callback)#subscribes to hadjective_speech topic
      rospy.spin()

  def main(self):
      rospy.sleep(2)
      query = 'What is your name?'
      self.soundhandle.say(query,self.voice)
      response = raw_input( self.color.set_green(query) + self.color.set_blue("\n>> ") )
      s = "It's a pleasure to meet you, %s. I am GRASPY, the PR2 roe-bot." % response
      self.soundhandle.say(s,self.voice)
      s = "It's a pleasure to meet you, %s. I am GRASPY, the PR2 robot." % response
      print self.color.set_green(s)
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
        self.soundhandle.say(query,self.voice)
        response = raw_input( self.color.set_green(query) + self.color.set_blue("\n>> ") )

        if response.lower().find('yes') is not -1 or response.lower().find('sure') is not -1: 
          s = "Excellent! May I have an%s ob-ject to touch?"%iteration[i][1] # Tell me when you are ready."
          self.soundhandle.say(s,self.voice)
          s = "Excellent! May I have an%s object to touch?" %iteration[i][1] #\nTell me when you are ready."
          print self.color.set_green(s)
          rospy.sleep(4)
          query = "Tell me when you are ready."
          self.soundhandle.say(query,self.voice)
          response = raw_input(self.color.set_green(query) + self.color.set_blue("\n>> ") )
          self.listener()

          i = i + 1
        else:
          s = "That's fine. I'm not going anywhere. I've got all the time in the world."
          self.soundhandle.say(s,voice)
          print self.color.set_green(s)
          rospy.sleep(10)

 


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
    demo = HadjSpeechDemo();
    demo.main()
    ''' rospy.init_node('hadjective_speech_node')
    #soundhandle = SoundClient()
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
        listener()

        i = i + 1
      else:
        s = "That's fine. I'm not going anywhere. I've got all the time in the world."
        soundhandle.say(s,voice)
        print color.set_green(s)
        rospy.sleep(10)

        #main(sys.argv[1:])'''

