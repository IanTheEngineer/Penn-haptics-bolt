/*********************************************************************
*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2012, University of Pennsylvania
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of University of Pennsylvania nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
* Author: Vivian Chu (chuv@grasp.upenn.edu)
*********************************************************************/

#include <biotac_simple_gripper/biotac_observer.h>
#include <biotac_simple_gripper/biotac_simple_gripper.h>
#include <biotac_simple_gripper/biotac_arm_controller.h>
#include <boost/thread.hpp>
#include <string>

class gripperController{
  private:
    ros::NodeHandle n;              // Ros Handle
    ros::Subscriber biotac_sub;     // biotac subscriber handle

    //======================================================================
    // Constants
    //======================================================================
    int Left;                        // These are defined in the BioTacObserver class
    int Right;
    static const int LightPressureContact = 50;         // Pressure value for light contacts
    static const int SqueezePressureContact = 400;      // Pressure value for squeezing objects
    
  public:

    //======================================================================
    // Constants
    //======================================================================
    static const double MoveGripperSlowDistance = 0.00005;  // Distance in meters
    static const double MoveGripperFastDistance = 0.0005;   // Distance moved for fast movement
    static const int MoveGripperRate = 50;                  // In Hz
    static const double SlideArmDistance = 0.05;            // In meters
    static const double GripperMaxOpenPosition = 0.08;      // Also specified in biotac_simple_gripper.h
    //======================================================================
    // Variables
    //====================================================================== 
    biotacObserver *biotac_obs;
    biotacSimpleGripper *simple_gripper;
    biotacArmController *arm_controller;
    std::string fileName;                                   // Filename to log data into

    //======================================================================
    // Gripper Constuctor
    // Instantiate all of the observer classes and setup subscribers
    //======================================================================
    gripperController()
    {
      // Initialize Biotac Observer
      biotac_obs = new biotacObserver();

      // Initialize Simple Gripper Controller
      simple_gripper = new biotacSimpleGripper();

      // Initialize Arm Controller
      arm_controller = new biotacArmController();

      // Initialize biotac subscriber
      biotac_sub = n.subscribe("biotac_pub", 1000, &biotacObserver::bioTacCB, biotac_obs);
      ROS_INFO("Subscribed to biotac_pub");

      // Initializing Left and Right
      Left = biotac_obs->Left;
      Right = biotac_obs->Right;
    }
    
    //======================================================================
    // Higher level motion to close gripper until contact is found
    // Pass in the rate at which contact is closed at and the distance the gripper moves
    // rate is in Hz, move_gripper_distance in meters
    // Velocity gripper moves is found by how much moved by what rate
    //======================================================================
    void findContact(ros::Rate rate, double move_gripper_distance)
    {
      int pressure_max = 0;

      while (pressure_max < LightPressureContact && ros::ok())
      {
        pressure_max = max(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);
        simple_gripper->closeByAmount(move_gripper_distance);
        ROS_INFO("Pressure Max is: [%d]", pressure_max);
        ros::spinOnce();
        rate.sleep();
      }
    }

    //======================================================================
    // Open gripper by the rate and position specified.
    // This is necessary to keep opening the gripper until the bioTacs do 
    // not report any pressure 
    //======================================================================
    void openUntilNoContact(ros::Rate rate, double move_gripper_distance)
    {
      int pressure_max = LightPressureContact + 50;

      while (pressure_max > LightPressureContact && ros::ok())
      {
        pressure_max = max(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);

        simple_gripper->open2Position(move_gripper_distance);
        ROS_INFO("Pressure Max is: [%d]", pressure_max);
        ros::spinOnce();
        rate.sleep();
      }
    }
   
    //====================================================================== 
    // Closes the gripper until a pressure is achieved, then opens again 
    // at the same rate the gripper closed at
    //======================================================================
    void squeeze(ros::Rate rate, double move_gripper_distance)
    {
      int pressure_max = 0;
   
      // Close 
      while (pressure_max < SqueezePressureContact && ros::ok())
      {
        pressure_max = max(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);

        simple_gripper->closeByAmount(move_gripper_distance);
        ROS_INFO("Pressure Max is: [%d]", pressure_max);
        ros::spinOnce();
        rate.sleep();
      }
    
      // Open - 10 and not 0 because the values will drift
      while (pressure_max > 10 && ros::ok())
      {
        pressure_max = max(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);

        simple_gripper->openByAmount(move_gripper_distance);
        ROS_INFO("Pressure Max is: [%d]", pressure_max);
        ros::spinOnce();
        rate.sleep();
      }
    }

    //======================================================================
    // Start Biotac Logger
    // Just a system call 
    //======================================================================
    void startLogger()
    {
      ROS_INFO("Start Logging");
      std::string command_pre("rosrun biotac_logger biotac_json_logger.py _filename:=");
      std::string command = command_pre + fileName; 
      std::cout << command << "\n";
      int success = system(command.c_str());
      if (success) ROS_INFO("Successfully Started"); 
    }

    //======================================================================
    // Destructor
    //======================================================================
    ~gripperController()
    {
    }
  };

//======================================================================
// Main flow
//======================================================================
int main(int argc, char* argv[])
{
  //====================================================================== 
  // Checks for place for data has been specified correctly 
  //======================================================================

  // Checks if the filename is given 
  if (argc < 2)
  {
    ROS_INFO("Please provide a name to store the data file in JSON form");
    exit(0);
  }

  char* filenameChar = argv[1];
  std::string filename = std::string(filenameChar);
  ROS_INFO("Writing to filename: %s", argv[1]);

  // Check if the file extention is .json
  if (std::string::npos == filename.find(".json"))
  {
    ROS_INFO("Please name the file with extention .json");
    exit(0);
  }

  //======================================================================
  // Start initializing controller
  //======================================================================
  
  ROS_INFO("Initializing simple controller"); 
  ros::init(argc, argv, "gripper_controller");
  
  //Create gripper controller
  gripperController controller;

  // Store filename in controller;
  controller.fileName = filenameChar;

  ROS_INFO("Waiting for BioTac Readings");
  //Wait for enough data to collect to normalize
  while (!controller.biotac_obs->init_complete_flag_ && ros::ok()){
    ros::spinOnce();
  }

  // Move hand every N seconds (in Hz)
  ros::Rate loop_rate(controller.MoveGripperRate);

  //======================================================================
  // Movements start from here
  //======================================================================

  //======================================================================
  // Open gripper and perform first action - find object and hold
  //======================================================================

  // Open the gripper
  ROS_INFO("Starting gripper movement");
  controller.simple_gripper->open2Position(controller.GripperMaxOpenPosition);
 
  // Move gripper to a set point in front of it
  ROS_INFO("Moving arm to start position"); 
  controller.arm_controller->moveArmToStart();

  // Start recording data
  ROS_INFO("Starting data logging");
  boost::thread testThread( boost::bind( &gripperController::startLogger, &controller));

  // Pause to allow the node to come up - 2 seconds
  ros::Rate waitNode(0.5);
  waitNode.sleep();

  ROS_INFO("Moving the gripper fast and find contact");
  // Close the gripper until contact is made 
  controller.findContact(loop_rate, controller.MoveGripperFastDistance); 

  // Open gripper slightly and find contact again
  ROS_INFO("Contact found at [%f], Opening gripper by 2cm", controller.simple_gripper->getGripperLastPosition());
  controller.openUntilNoContact(loop_rate, controller.simple_gripper->getGripperLastPosition() + 0.02);

  // Close gripper again - this time slow 
  ROS_INFO("Moving gripper slowly and find contact");
  controller.findContact(loop_rate, controller.MoveGripperSlowDistance);

  ROS_INFO("Contact found - holding for 10 seconds");
  // Hold the position for 10 seconds
  ros::Rate wait(0.1);
  wait.sleep();

  //======================================================================
  // Start motion slide down
  //======================================================================

  // Find position of arm
  controller.arm_controller->getArmTransform();
  double x = controller.arm_controller->getTransform('x');
  double y = controller.arm_controller->getTransform('y');
  double z = controller.arm_controller->getTransform('z');

  ROS_INFO("Current Arm location: X: [%f], Y: [%f], Z: [%f]", x,y,z);
  ROS_INFO("Sliding Arm down by [%f] meters", controller.SlideArmDistance);
  // Slide the arm down - currently 5 cm down
  controller.arm_controller->slide_down(x, y, z, controller.SlideArmDistance);

  ROS_INFO("Slide completed, holding for 2 seconds");
  // Wait for a small amount of time - 2 seconds
  waitNode.sleep();

  //======================================================================
  // Start motion to squeeze
  //====================================================================== 

  ROS_INFO("Starting Squeeze Motion"); 
  // Re-open gripper and find contact again - (from last position + 0.5cm)
  controller.simple_gripper->open2Position(controller.simple_gripper->getGripperLastPosition()+0.02);
  controller.findContact(loop_rate, controller.MoveGripperSlowDistance);
  
  // Squeeze goes here 
  controller.squeeze(loop_rate, controller.MoveGripperSlowDistance);

  // Controller open all
  controller.simple_gripper->open2Position(controller.GripperMaxOpenPosition);

  //======================================================================
  // Destroy logger
  //======================================================================
  int success = system("rosnode kill biotac_json_logger");
  if (success) ROS_INFO("Logger stopped");
  testThread.join();

  return 0;
}


