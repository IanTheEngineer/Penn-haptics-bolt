/********************************************************************
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
#include "std_msgs/Int8.h"
#include "std_msgs/String.h"
#include <boost/thread.hpp>
#include <string>
#include <sstream>

class gripperController{
  private:
    ros::NodeHandle n;              // Ros Handle
    ros::Subscriber biotac_sub;     // biotac subscriber handle

    //================================================================
    // Constants
    //================================================================
    int Left;                        // These are defined in the BioTacObserver class
    int Right;
    static const int LightPressureContact = 20;         // Pressure value for light contacts
    static const int SqueezePressureContact = 500;      // Pressure value for squeezing objects
    static const int RedistributePressureThreshold = 10; // Threshold of pressure between two biotacs to move arm
    static const int MaxBadPressure = 200;              // Maximum pressure between two biotacs when an object is off center 
  public:

    //================================================================
    // Constants
    //================================================================
    static const double MoveGripperSlowDistance = 0.00005;  // Distance in meters
    static const double MoveGripperFastDistance = 0.0005;   // Distance moved for fast movement
    static const int MoveGripperRate = 50;                  // In Hz
    static const double SlideArmDistance = 0.05;            // In meters
    static const double GripperMaxOpenPosition = 0.08;      // Also specified in biotac_simple_gripper.h
    static const double GripperSlowContactProportion = 0.2;     // Proportion of object complaince to move gripper for slide and hold
    static const double GripperFastContactProportion = 0.1;     // Proportion of object compliance to move gripper for fast slide
    static const double GripperThermalContactProportion = 0.5;  // Proportion of the object compliance to move gripper for thermal hold 
    static const int SlowSlideTime = 5;                     // Time to slide 5cm slow
    static const int FastSlideTime = 2;                     // Time to slide 5cm fast
    static const int LiftPressure = 250;
    static const double LiftHeight = 0.05; 
    static const int LiftTime = 5; 
    static const int DISABLED = 0;
    static const int THERMAL_HOLD = 1;
    static const int SLIDE = 2;
    static const int SQUEEZE = 3;
    static const int DONE = 4;
    static const int TAP = 5;
    static const int SLIDE_FAST = 6;
    static const int CENTER_GRIPPER = 7;

    //================================================================
    // Variables
    //================================================================ 
    biotacObserver *biotac_obs;
    biotacSimpleGripper *simple_gripper;
    biotacArmController *arm_controller;
    std::string fileName;                                   // Filename to log data into
    std::string filePath;
    int state;
    std::string detail_state;
    ros::Publisher state_pub;
    ros::Publisher detailed_state_pub;
    double gripper_initial_contact_position;                         // Store distance where gripper first finds the object
    double gripper_max_squeeze_position;                            // Gripper distance when max PDC is achieved
    double gripper_slow_optimal_contact_position;                 // Position for gripper to go to for optimal contact
    double gripper_fast_optimal_contact_position;
    double gripper_thermal_optimal_contact_position;
    int tap_pressure_left;
    int tap_pressure_right;

    struct fingerContact
    {
      double position;
      int finger;
    };

    fingerContact firstContact;
    fingerContact secondContact;

    //================================================================
    // Gripper Constuctor
    // Instantiate all of the observer classes and setup subscribers
    //================================================================
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

      // Initialize the initial contact distance to 8cm first
      gripper_initial_contact_position = 0.08;
      gripper_max_squeeze_position = 0.0;   // initialze it to 0 first

      // Set controller state to DISABLED
      state = DISABLED;
      detail_state = "DISABLED";

      // Initialize publisher
      state_pub = n.advertise<std_msgs::Int8>("simple_gripper_controller_state", 10);
      detailed_state_pub = n.advertise<std_msgs::String>("simple_gripper_controller_state_detailed", 10);

      // Initialize no finger touch first
      tap_pressure_left = 0;
      tap_pressure_right = 0;
    }

    //================================================================
    // Function to move the arm a little in a direction it needs to
    // move
    //================================================================
    void centerArmIncremental(ros::Rate rate, double move_gripper_distance)
    {

      // Find position of arm
      arm_controller->getArmTransform();
      double x = arm_controller->getTransform('x');
      double y = arm_controller->getTransform('y');
      double z = arm_controller->getTransform('z');

      int pressure_min = 0; 
      int pressure_max = 0;
      bool fingerSet = false;
      int num_run = 0;
      bool not_centered = true;
      int pressure_left = 0;
      int pressure_right = 0;

      // Close until minimum pressure is found - however stop if
      // any finger has too much pressure
      while (num_run < 4 && not_centered 
             && pressure_max < 600 && ros::ok())
      { 
        simple_gripper->closeByAmount(move_gripper_distance);
       
        pressure_left = biotac_obs->pressure_normalized_[Left];
        pressure_right = biotac_obs->pressure_normalized_[Right];

        // Check pressure min and max
        pressure_min = min(pressure_left, pressure_right);
        pressure_max = max(pressure_left, pressure_right);
        
        // First touches object
        if (pressure_max > 5 && !fingerSet)
        {
          fingerSet = true;
          
          if (pressure_min > 5)
          {
            not_centered = false;
            fingerSet = false;
          }

          firstContact.position = simple_gripper->getGripperLastPosition();
          if (pressure_left > pressure_right)
          {
            firstContact.finger = Left;
          }
          else
          {
            firstContact.finger = Right;
          }
        }

        if (fingerSet)
        {
          // Open grippers 
          simple_gripper->open2Position(GripperMaxOpenPosition);
          
          // Move left case 
          if (firstContact.finger == Left)
          {
            arm_controller->move_arm_to(x,y+0.015,z,2);    
          } 
          else 
          {
            arm_controller->move_arm_to(x,y-0.015,z,2);
          }
          fingerSet = false;
          num_run++;
        } 

        //ROS_INFO("Pressure Min is: [%d]", pressure_min);
       // ROS_INFO("Pressure Max is: [%d]", pressure_max);
        ros::spinOnce();
        rate.sleep();
      }
    }



    //================================================================
    // Function that moves arm according to pressures given
    //================================================================
    void redistributePressurePosition()
    {
      // Find position of arm
      arm_controller->getArmTransform();
      double x = arm_controller->getTransform('x');
      double y = arm_controller->getTransform('y');
      double z = arm_controller->getTransform('z');
      
      double gripper_difference = (firstContact.position - secondContact.position)/2.0;
      ROS_INFO("Difference calculated is: [%f]",gripper_difference); 
      ROS_INFO("First contact position is: [%f], second contact: [%f]", firstContact.position, secondContact.position);
      // Move left case 
      if (firstContact.finger == Left)
      {
        arm_controller->move_arm_to(x,y+gripper_difference,z,2);    
      } 
      else 
      {
        arm_controller->move_arm_to(x,y-gripper_difference,z,2);
      } 
    } 

    //================================================================
    // Function that moves arm according to pressures given
    //================================================================
    void redistributePressure()
    {
      // Find position of arm
      arm_controller->getArmTransform();
      double x = arm_controller->getTransform('x');
      double y = arm_controller->getTransform('y');
      double z = arm_controller->getTransform('z');
      
      int pressure_difference = tap_pressure_left-tap_pressure_right;
      double movement = (GripperMaxOpenPosition/2.0)*(abs(pressure_difference)/MaxBadPressure); 

      // Move left case 
      if (pressure_difference > RedistributePressureThreshold)
      {
        arm_controller->move_arm_to(x,y+movement,z,2);    
      } 
      else if(pressure_difference < -RedistributePressureThreshold)
      {
        arm_controller->move_arm_to(x,y-movement,z,2);
      } 
      else
      {
        ROS_INFO("Pressure normalized!");
      }
    } 

    //================================================================
    // Higher level motion to close gripper until contact is found
    // Pass in the rate at which contact is closed at and the 
    // distance the gripper moves rate is in Hz, 
    // move_gripper_distance in meters
    // Velocity gripper moves is found by how much moved by what rate
    //================================================================
    void findContact(ros::Rate rate, double move_gripper_distance)
    {
      int pressure_min = 0; 
      int pressure_max = 0;
      bool contact_found = false;
      bool fingerSet = true;
      int no_motion_counter = 0;
      int previous_pressure_max = 0;

      // Close until minimum pressure is found - however stop if
      // any finger has too much pressure
      while (pressure_min < LightPressureContact && ros::ok()
             && pressure_max < 600  && no_motion_counter < 250)
      {
        previous_pressure_max = pressure_max;
        
        // Checks if pressure has been "stuck" 
        if (abs(previous_pressure_max-pressure_max) < 1)
          no_motion_counter++;

        // First touches object
        if (pressure_max > 5 && fingerSet)
        {
          firstContact.position = simple_gripper->getGripperLastPosition();
          if (biotac_obs->pressure_normalized_[Left] > biotac_obs->pressure_normalized_[Right])
          {
            firstContact.finger = Left;
          }
          else
          {
            firstContact.finger = Right;
          }
          fingerSet = false;
        }

        // Second finger touches object
        if (pressure_min > 10)
        {
          secondContact.position = simple_gripper->getGripperLastPosition();
          if (firstContact.position == Left)
          {
            secondContact.finger = Right;
          }
          else
          {
            secondContact.finger = Left;
          }
        }

        // Set distance for object width 
        if (!contact_found && pressure_min > 10){
          gripper_initial_contact_position = simple_gripper->getGripperLastPosition();
          contact_found = true;
        }
      
        // Store last pressure felt by each finger 
        tap_pressure_left = biotac_obs->pressure_normalized_[Left];
        tap_pressure_right = biotac_obs->pressure_normalized_[Right];

        // Check pressure min and max
        pressure_min = min(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);
        pressure_max = max(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);
        simple_gripper->closeByAmount(move_gripper_distance);
        //ROS_INFO("Pressure Min is: [%d]", pressure_min);
       // ROS_INFO("Pressure Max is: [%d]", pressure_max);
        ros::spinOnce();
        rate.sleep();
      }
    }

    //================================================================
    // Close gripper to specified pressure
    //================================================================
    void closeToPressure(ros::Rate rate, int desired_pressure, 
                         double move_gripper_distance)
    {
      double current_gripper_position = simple_gripper->getGripperLastPosition();
      int pressure_max = 0;
      int pressure_min = 0;

      while (pressure_min < desired_pressure 
             && pressure_max < 500
             && ros::ok()
             && current_gripper_position > 0.0)
      {
        // Move the gripper by the specified amount
        simple_gripper->closeByAmount(move_gripper_distance);
        
        // Find and update if gripper moved
        current_gripper_position = simple_gripper->getGripperLastPosition();
      
        // Get pressure
        pressure_max = max(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);
        pressure_min = min(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);

        // Wait set time and check again 
        ros::spinOnce();
        rate.sleep();
      }
    }

    //================================================================
    // Close the gripper at the specified rate to the distance
    // specified
    //================================================================
    void closeToPosition(ros::Rate rate, double move_gripper_distance, 
                         double gripper_position)
    {
      // Get location of gripper currently 
      double current_gripper_position = simple_gripper->getGripperLastPosition();
      int pressure_max = 0;

      // Continue until position is achieved, ros cancel, or if
      // the pressure approaches something dangerous
      while (current_gripper_position > gripper_position 
            && pressure_max < 500 
            && current_gripper_position > 0.0 
            && ros::ok())
      {
        // Move the gripper by the specified amount
        simple_gripper->closeByAmount(move_gripper_distance);
        
        // Find and update if gripper moved
        current_gripper_position = simple_gripper->getGripperLastPosition();
      
        // Get pressure
        pressure_max = max(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);

        // Wait set time and check again 
        ros::spinOnce();
        rate.sleep();
      }
    }
       

    //================================================================
    // Open gripper by the rate and position specified.
    // This is necessary to keep opening the gripper until 
    // the bioTacs do not report any pressure 
    //================================================================
    void openUntilNoContact(ros::Rate rate, double gripper_position)
    {
      int pressure_max = LightPressureContact + 50;

      while (pressure_max > LightPressureContact && ros::ok())
      {
        pressure_max = max(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);

        simple_gripper->open2Position(gripper_position);
        //ROS_INFO("Pressure Max is: [%d]", pressure_max);
        ros::spinOnce();
        rate.sleep();
      }
    }
   
    //================================================================ 
    // Closes the gripper until a pressure is achieved, 
    // then opens again at the same rate the gripper closed at
    //================================================================
    void squeeze(ros::Rate rate, double move_gripper_distance)
    {
      int previous_pressure_max = 0; 
      int pressure_max = 0;
      int no_motion_counter = 0;

      // Close 
      while (pressure_max < SqueezePressureContact && ros::ok()
             && no_motion_counter < 250 
	           && simple_gripper->getGripperLastPosition() > 0.0)
      {
        previous_pressure_max = pressure_max;
        pressure_max = max(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);

        // Checks if pressure has been "stuck" 
        if (abs(previous_pressure_max-pressure_max) < 1)
          no_motion_counter++;

        simple_gripper->closeByAmount(move_gripper_distance);
        //ROS_INFO("Pressure Max is: [%d]", pressure_max);
        ros::spinOnce();
        rate.sleep();
      }
   
      // Store the gripper position when max pressure is achieved
      gripper_max_squeeze_position = simple_gripper->getGripperLastPosition();

      // Open - 10 and not 0 because the values will drift
      while (pressure_max > 10 && ros::ok() 
	     && simple_gripper->getGripperLastPosition() < 0.08)
      {
        pressure_max = max(biotac_obs->pressure_normalized_[Left], biotac_obs->pressure_normalized_[Right]);

        simple_gripper->openByAmount(move_gripper_distance);
        //ROS_INFO("Pressure Max is: [%d]", pressure_max);
        ros::spinOnce();
        rate.sleep();
      }
    }

    //================================================================ 
    // Compute position to move gripper for optimal contact during
    // hold and slide.  Currently it is a proportion of the total
    // compliance of the object (distance sensors sink into the object
    //================================================================ 
    void computeOptimalSensorContactLocation()
    {
      // Find Distance object compressed
      double object_compliance_distance = gripper_initial_contact_position - 
                                          gripper_max_squeeze_position;

      ROS_INFO("Object initial size is [%f], Squeeze size is [%f], Compliance Distance is [%f]", gripper_initial_contact_position, gripper_max_squeeze_position, object_compliance_distance);

      // Find proportion of the distance compressed
      double gripper_contact_distance = GripperSlowContactProportion * object_compliance_distance;
      double gripper_fast_contact_distance = GripperFastContactProportion * object_compliance_distance;
      double gripper_thermal_contact_distance = GripperThermalContactProportion * object_compliance_distance;

      ROS_INFO("Distance to move into object slow is [%f]", gripper_contact_distance);
      ROS_INFO("Distance to move into object fast is [%f]", gripper_fast_contact_distance);
      ROS_INFO("Distance to move into object thermal is [%f]", gripper_thermal_contact_distance); 

      // Find the position the gripper should move to 
      gripper_slow_optimal_contact_position = gripper_initial_contact_position - gripper_contact_distance;
      gripper_fast_optimal_contact_position = gripper_initial_contact_position - gripper_fast_contact_distance; 
      gripper_thermal_optimal_contact_position = gripper_initial_contact_position - gripper_thermal_contact_distance;
      
      ROS_INFO("Gripper optimal slow contact position is [%f]", gripper_slow_optimal_contact_position);
      ROS_INFO("Gripper optimal fast contact position is [%f]", gripper_fast_optimal_contact_position);
      ROS_INFO("Gripper optimal thermal contact position is [%f]", gripper_thermal_optimal_contact_position);
    }

    //================================================================
    // Start Biotac Logger
    // Just a system call 
    //================================================================
    void startLogger()
    {
      ROS_INFO("Start Logging");
      std::stringstream command;
      command << "rosrun pr2_arm_state_aggregator pr2_biotac_sub.py _filename:="<<fileName<<" _data_path:="<<filePath;
      std::cout << command << "\n";
      int success = system(command.str().c_str());
      if (success) ROS_INFO("Successfully Started"); 
    }

    //================================================================
    // Publish the state continuously
    //================================================================
    void publishState()
    {
      ros::Rate loop_rate(100);
      ROS_INFO("Start publishing state");
      while (ros::ok() && state != DONE)
      { 
        std_msgs::Int8 msg;
        std_msgs::String msgDetail;
        msg.data = state;
        msgDetail.data = detail_state;
        state_pub.publish(msg);
        detailed_state_pub.publish(msgDetail);
        ros::spinOnce();
        loop_rate.sleep();
      }
    
      // Publish the final state (DONE)
      std_msgs::Int8 msg;
      msg.data = state;
      state_pub.publish(msg);
    } 

    //================================================================
    // Destructor
    //================================================================
    ~gripperController()
    {
      delete biotac_obs;
      delete simple_gripper;
    }
  };

//================================================================
// Main flow
//================================================================
int main(int argc, char* argv[])
{
  //================================================================ 
  // Checks for place for data has been specified correctly 
  //================================================================

  // Checks if the filename is given 
  if (argc < 2)
  {
    ROS_INFO("Please provide a name and path to store the data file in JSON form");
    exit(0);
  }

  char* filepathChar = argv[1];
  std::string filepath = std::string(filepathChar);
  ROS_INFO("Writing to file path: %s", argv[1]);
  char* filenameChar = argv[2];
  std::string filename = std::string(filenameChar);
  ROS_INFO("Writing to filename: %s", argv[2]);

  // Check if the file extention is .json
  if (std::string::npos == filename.find(".json"))
  {
    ROS_INFO("Please name the file with extention .json");
    exit(0);
  }

  //================================================================
  // Start initializing controller
  //================================================================
  
  ROS_INFO("Initializing simple controller"); 
  ros::init(argc, argv, "gripper_controller");
  
  //Create gripper controller
  gripperController controller;

  // Store filename in controller;
  controller.filePath = filepathChar;
  controller.fileName = filenameChar;

  // Start thread to publish controller state
  ROS_INFO("Starting controller state publisher");
  boost::thread statePubThread( boost::bind( &gripperController::publishState, &controller));

  ROS_INFO("Waiting for BioTac Readings");
  //Wait for enough data to collect to normalize
  while (controller.biotac_obs->init_flag_ && ros::ok()){
    ros::spinOnce();
  }

  // Move hand every N seconds (in Hz)
  ros::Rate loop_rate(controller.MoveGripperRate);
  
  //================================================================
  // Movements start from here
  //================================================================

  //================================================================
  // Open gripper and move to start position, start logger
  //================================================================

  // Open the gripper
  ROS_INFO("Starting gripper movement");
  controller.detail_state = "OPEN_GRIPPER_MAX";
  controller.simple_gripper->open2Position(controller.GripperMaxOpenPosition);
 
  // Move gripper to a set point in front of it
  ROS_INFO("Moving arm to start position");
  controller.detail_state = "MOVE_ARM_START_POSITION";
  controller.arm_controller->moveArmToStart();

  // Start recording data
  ROS_INFO("Starting data logging");
  boost::thread loggingThread( boost::bind( &gripperController::startLogger, &controller));

  // Pause to allow the node to come up - 2 seconds
  ros::Rate waitNode(0.2);
  waitNode.sleep();

  //================================================================
  // Redistribute pressure - move arm in direction
  //================================================================
  controller.state = controller.CENTER_GRIPPER;
  ROS_INFO("Centering the Gripper");

  /*for (int i = 0; i < 2; i++) 
  {
    ROS_INFO("Find contact");
    controller.findContact(loop_rate, controller.MoveGripperFastDistance);
    ROS_INFO("Open gripper");
    controller.simple_gripper->open2Position(controller.GripperMaxOpenPosition);
    ROS_INFO("Redistribute Pressure Position"); 
    controller.redistributePressurePosition();
  }*/

  controller.centerArmIncremental(loop_rate, controller.MoveGripperFastDistance);
  controller.simple_gripper->open2Position(controller.GripperMaxOpenPosition);

  //================================================================
  // Fast Tap - to first find contact with object
  //================================================================
  controller.state = controller.TAP;
  ROS_INFO("State set to [%d]", controller.state);

  ROS_INFO("Moving the gripper fast and find contact");
  controller.detail_state = "MOVE_GRIPPER_FAST_CLOSE";
  // Close the gripper until contact is made 
  controller.findContact(loop_rate, controller.MoveGripperFastDistance); 
 
  // Open gripper slightly fast
  ROS_INFO("Contact found at [%f], Opening gripper by 2cm", controller.gripper_initial_contact_position);
  controller.detail_state = "OPEN_GRIPPER_BY_2CM_FAST";
  controller.openUntilNoContact(loop_rate, controller.gripper_initial_contact_position + 0.02);
  
  /*//================================================================
  // Close the gripper on the object to a specified pressure
  // and lift the object 
  //================================================================
  
  // Close gripper
  controller.state = controller.LIFT; 
  ROS_INFO("Closing gripper to lift");

  ROS_INFO("Moving the gripper fast and to the pressure [%d]", controller.LiftPressure);
  controller.detail_state = "MOVE_GRIPPER_SLOW_CLOSE";
  controller.closeToPressure(loop_rate, controller.LiftPressure, controller.MoveGripperSlowDistance);

  ROS_INFO("Lifting the Object");
  controller.detail_state = "LIFT_OBJECT";
  
  // Get current arm location  
  controller.arm_controller->getArmTransform();
  double x = controller.arm_controller->getTransform('x');
  double y = controller.arm_controller->getTransform('y');
  double z = controller.arm_controller->getTransform('z');

  ROS_INFO("Current Arm location: X: [%f], Y: [%f], Z: [%f]", x,y,z);
  controller.arm_controller->move_arm_to(x,y,z+controller.LiftHeight, controller.LiftTime);

  ROS_INFO("Putting down Object");
  controller.detail_state = "PUT_DOWN_OBJECT";

  controller.arm_controller->move_arm_to(x,y,z, controller.LiftTime);

  // Open gripper slightly fast
  ROS_INFO("Object on the table, reopening gripper to [%f]", controller.gripper_initial_contact_position);
  controller.detail_state = "OPEN_GRIPPER_BY_2CM_FAST";
  controller.openUntilNoContact(loop_rate, controller.gripper_initial_contact_position + 0.02);
*/
  //================================================================
  // Start motion to squeeze
  //================================================================ 
  controller.state = controller.SQUEEZE;
  ROS_INFO("State set to [%d]", controller.state);

  // Find contact again - (from contact position + 0.5cm)  
  ROS_INFO("Starting Squeeze Motion");
  controller.detail_state = "FIND_CONTACT_CLOSE_GRIPPER_SLOW";
  controller.findContact(loop_rate, controller.MoveGripperSlowDistance);
  
  // Squeeze goes here
  controller.detail_state = "SQUEEZE_SET_PRESSURE_SLOW";
  controller.squeeze(loop_rate, controller.MoveGripperSlowDistance);

  // Open gripper slightly fast
  ROS_INFO("Object distance found [%f], Opening gripper by 2cm", controller.gripper_max_squeeze_position);
  controller.detail_state = "OPEN_GRIPPER_BY_2CM_FAST";
  controller.openUntilNoContact(loop_rate, controller.gripper_initial_contact_position + 0.02);

  // Wait a second to allow the finger temperature to normalize
  ros::Duration(1.0).sleep(); 

  //================================================================
  // Compute Position to move gripper for optimal contact
  //================================================================
  ROS_INFO("Computing Optimal Gripper Contact Position");
  controller.computeOptimalSensorContactLocation();

  //================================================================
  // Renormalize Sensors
  //================================================================
  controller.biotac_obs->renormalize();
   
  //================================================================
  // Thermal Hold
  //================================================================
  controller.state = controller.THERMAL_HOLD;
  ROS_INFO("State set to [%d]", controller.state);

  // Move gripper to optimal contact position  
  ROS_INFO("Moving gripper slowly to position: [%f]", controller.gripper_thermal_optimal_contact_position);
  controller.detail_state = "CLOSE_GRIPPER_SLOW_TO_POSITION";
  controller.closeToPosition(loop_rate, controller.MoveGripperSlowDistance,
                             controller.gripper_thermal_optimal_contact_position);

  ROS_INFO("Contact found - holding for 10 seconds");
  controller.detail_state = "HOLD_FOR_10_SECONDS";
  // Hold the position for 10 seconds
  ros::Rate wait(0.1);
  wait.sleep();

  // Open gripper slightly fast
  ROS_INFO("Opening gripper by 2cm");
  controller.detail_state = "OPEN_GRIPPER_BY_2CM_FAST";
  controller.openUntilNoContact(loop_rate, controller.gripper_initial_contact_position + 0.02);
  //================================================================
  // Move arm back up 5cm to slide  
  //================================================================
  ROS_INFO("Moving arm up 5cm"); 
  controller.detail_state = "MOVE_UP_5CM";

  controller.arm_controller->getArmTransform();
  double x = controller.arm_controller->getTransform('x');
  double y = controller.arm_controller->getTransform('y');
  double z = controller.arm_controller->getTransform('z') + 0.07;

  ROS_INFO("Arm location will move to: X: [%f], Y: [%f], Z: [%f]", x,y,z);
  controller.detail_state = "MOVE_UP_5CM";
  ROS_INFO("Moving Arm up by 5 cm");
  controller.arm_controller->move_arm_to(x,y,z, 2);

  //================================================================
  // Start motion slide down
  //================================================================
  controller.state = controller.SLIDE;
  ROS_INFO("State set to [%d]", controller.state);

  // Move gripper to optimal contact position  
  ROS_INFO("Moving gripper slowly to position: [%f]", controller.gripper_slow_optimal_contact_position);
  controller.detail_state = "CLOSE_GRIPPER_SLOW_TO_POSITION";
  controller.closeToPosition(loop_rate, controller.MoveGripperSlowDistance,
                             controller.gripper_slow_optimal_contact_position);

  ROS_INFO("Contact found - starting slide motion");
  
  ros::Rate slide_rate(1); 
  // Find position of arm
  //controller.arm_controller->getArmTransform();
  //x = controller.arm_controller->getTransform('x');
  //y = controller.arm_controller->getTransform('y');
  //z = controller.arm_controller->getTransform('z');

  ROS_INFO("Current Arm location: X: [%f], Y: [%f], Z: [%f]", x,y,z);
  controller.detail_state = "SLIDE_5CM";
  ROS_INFO("Sliding Arm down by [%f] meters", controller.SlideArmDistance);
  // Slide the arm down - currently 5 cm down
  controller.arm_controller->slide_down(x, y, z, controller.SlideArmDistance, controller.SlowSlideTime);
  //controller.slide(slide_rate, 0.05);
  
  ROS_INFO("Slide completed, holding for 5 seconds");
  controller.detail_state = "SLIDE_DONE_WAIT_5";
  // Wait for a small amount of time - 5 seconds
  waitNode.sleep();

  controller.detail_state = "OPEN_GRIPPER_FAST_2CM";
  // Re-open gripper and find contact again - (from last position + 0.5cm)
  controller.simple_gripper->open2Position(controller.gripper_initial_contact_position+0.02);

  //================================================================
  // Move arm back up 5cm to slide again 
  //================================================================
  ROS_INFO("Moving arm back up 5cm"); 
  controller.detail_state = "MOVE_UP_5CM";

  //controller.arm_controller->getArmTransform();
  //x = controller.arm_controller->getTransform('x');
  //y = controller.arm_controller->getTransform('y');
  //z = controller.arm_controller->getTransform('z');

  ROS_INFO("Arm location will move to: X: [%f], Y: [%f], Z: [%f]", x,y,z);
  controller.detail_state = "MOVE_UP_5CM";
  ROS_INFO("Moving Arm up by 5 cm");
  controller.arm_controller->move_arm_to(x,y,z, 2);

  //================================================================
  // Slide down fast
  //================================================================
  controller.state = controller.SLIDE_FAST;
  ROS_INFO("State set to [%d]", controller.state);

  // Move gripper to optimal contact position  
  ROS_INFO("Moving gripper slowly to position: [%f]", controller.gripper_fast_optimal_contact_position);
  controller.detail_state = "CLOSE_GRIPPER_SLOW_TO_POSITION";
  controller.closeToPosition(loop_rate, controller.MoveGripperSlowDistance,
                             controller.gripper_fast_optimal_contact_position);
  ROS_INFO("Contact found");

  // Get arm position again
  controller.arm_controller->getArmTransform();
  x = controller.arm_controller->getTransform('x');
  y = controller.arm_controller->getTransform('y');
  z = controller.arm_controller->getTransform('z');
 
  ROS_INFO("Current Arm location: X: [%f], Y: [%f], Z: [%f]", x,y,z);
  controller.detail_state = "MOVE_DOWN_5CM";
  ROS_INFO("Sliding Arm down by [%f] meters", controller.SlideArmDistance);
   
  // Slide the arm down - currently 5 cm down
  controller.arm_controller->slide_down(x, y, z, controller.SlideArmDistance, controller.FastSlideTime);
  //controller.slide(slide_rate, 0.05);
  
  ROS_INFO("Slide completed, holding for 5 seconds");
  controller.detail_state = "SLIDE_DONE_WAIT_5";
  // Wait for a small amount of time - 5 seconds
  waitNode.sleep();

  controller.detail_state = "OPEN_GRIPPER_FAST_2CM";
  // Re-open gripper and find contact again - (from last position + 0.5cm)
  controller.simple_gripper->open2Position(controller.gripper_initial_contact_position+0.02);

  //================================================================
  // Reset hand back to normal
  //================================================================
  // Controller open all
  controller.detail_state = "OPEN_GRIPPER_FAST_MAX";
  controller.simple_gripper->open2Position(controller.GripperMaxOpenPosition);
  controller.arm_controller->moveArmToStart();

  //================================================================
  // Destroy logger
  //================================================================
  int success = system("rosnode kill pr2_biotac_logger");
  if (success) ROS_INFO("Logger stopped");
  loggingThread.join();

  //================================================================
  // Stop state publisher
  //================================================================
  controller.state = controller.DONE;
  ROS_INFO("State set to [%d]", controller.state);
  statePubThread.join();

  return 0;
}


