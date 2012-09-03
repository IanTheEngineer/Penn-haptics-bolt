import roslib
roslib.load_manifest("move_arm_in_position")
import rospy

from pr2_control_utilities import PR2MoveArm
from pr2_control_utilities import ControllerManagerClient
from pr2_control_utilities import PR2BaseMover
from tabletop_actions.object_detector import GenericDetector, FindClusterBoundingBox2Response
from octomap_filters.srv import FilterDefine, FilterDefineRequest
from std_srvs.srv import Empty

class MoveToHaptics(object):
    """Class to detect an object and move the arm so that the object will be
    in between the gripper.
    """
    def __init__(self, whicharm = "left_arm",
                 octomap_filters_service="/create_filter",
                 padding = 0.1):
        """
        whicharm: a string, right_arm or left_arm.
        padding: a float 
        """
        self.whicharm = whicharm
        self.padding = padding
        
        #Don't use the object recognition-detection, otherwise it will populate
        #the collision map
        self.detector = GenericDetector(detector_service = None,
                                        collision_processing = None
                                        )
        self.planner = PR2MoveArm()
        
        rospy.loginfo("Waiting for %s", octomap_filters_service)
        rospy.wait_for_service(octomap_filters_service)
        self.filter_srv = rospy.ServiceProxy(octomap_filters_service,
                                             FilterDefine)

        self.manager = ControllerManagerClient()
        if self.whicharm.startswith("right"):
            self.manager.switch_controllers(["r_arm_controller"],
                                            ["r_arm_cart_imped_controller"], 
                                            )
        else:
            self.manager.switch_controllers(["l_arm_controller"],
                                            ["l_arm_cart_imped_controller"], 
                                            )
        
        haptics_service_name = "start_haptic_exploration"
        rospy.loginfo("Waiting for service %s", haptics_service_name)
        self.haptics_service = rospy.ServiceProxy(haptics_service_name, Empty)
        
        self.move_base = PR2BaseMover(listener = self.planner.tf_listener,
                                      use_safety_dist=True)
        
        rospy.loginfo("%s is ready", self.__class__.__name__)
    
    def detect_and_filter(self):
        """Detect and object, selects the biggest cluster and removes the points
        from the map.
        
        Returns the object's pointcloud and bounding box on success, None otherwise
        """
        
        det = self.detector.segment_only()
        if det is None:
            rospy.logwarn("Problems while segmenting")
            return None
        
        
        cluster = self.detector.find_biggest_cluster(det.detection.clusters)
        box = self.detector.detect_bounding_box(cluster)
        assert isinstance(box, FindClusterBoundingBox2Response)
        xmin, xmax = self.detector.get_min_max_box(box, self.padding)
        
        #now filtering the octomap
        req = FilterDefineRequest()
        req.name = "object"
        req.operation = req.CREATE
        
        req.min.header.frame_id = box.pose.header.frame_id
        req.min.point.x = xmin[0]
        req.min.point.y = xmin[1]
        req.min.point.z = xmin[2]
        
        req.max.header.frame_id = box.pose.header.frame_id
        req.max.point.x = xmax[0]
        req.max.point.y = xmax[1]
        req.max.point.z = xmax[2]        
        
        try:
            self.filter_srv(req)
        except Exception, e:
            rospy.logerr("Error while calling the filtering service: %s", e)
            return None
        
        self.planner.update_planning_scene()
        return cluster, box
        
    
    def move_arm_to_pre_haptics(self,
                                pre_touch_difference = (-0.1,0,0)):
        """Moves the arm so that an object will roughly be between the fingers.
        It first moves to an approach pose, then it will actually move to the pose
        """
        joint_mover = self.planner.joint_mover
        if self.whicharm.startswith("right"):
            open_gripper = joint_mover.open_right_gripper
            move_arm_planning = self.planner.move_right_arm_with_ik
            move_arm_non_planning = self.planner.move_right_arm_non_collision
        else:
            open_gripper = joint_mover.open_left_gripper
            move_arm_planning = self.planner.move_left_arm_with_ik
            move_arm_non_planning = self.planner.move_left_arm_non_collision
        
        _, box = self.detect_and_filter()
        assert isinstance(box, FindClusterBoundingBox2Response)
        
        #TODO Here I should convert the pose to somenthing like /base_link, instead of
        #assuming it
        box_pose = [box.pose.pose.position.x,
                    box.pose.pose.position.y,
                    box.pose.pose.position.z
                   ]
        gripper_accounted_pose = [box_pose[0] - 0.18 - box.box_dims.x/2.,
                                  box_pose[1] + 0,
                                  box_pose[2] - 0.01
                                  ]
        
        orientation = [0,0,0,1]
        frame_id = box.pose.header.frame_id
        
        open_gripper()
        
        pre_touch_pose = gripper_accounted_pose[:]        
        pre_touch_pose[0] += pre_touch_difference[0]
        pre_touch_pose[1] += pre_touch_difference[1]
        pre_touch_pose[2] += pre_touch_difference[2]
        
        rospy.loginfo("Moving to a pre-touch position")
        if not move_arm_planning(pre_touch_pose, orientation, frame_id, 10):
            rospy.logerr("Could not move to pre-touch position!")
            return False
            
        touch_pose = gripper_accounted_pose[:]
        touch_pose[0] += 0.0
        rospy.loginfo("Moving to a touch position")
        if not move_arm_non_planning(touch_pose, orientation, frame_id, 3):
            rospy.logerr("Could not move to pre-touch position!")
            return False
        
        return True
    
    def execute_haptics(self):
        """Loads the right controllers, calls the service, re-loads the old controllers"""
        
        if self.whicharm.startswith("right"):
            self.manager.switch_controllers(["r_arm_cart_imped_controller"], 
                                            ["r_arm_controller"]                                            
                                            )
        else:
            self.manager.switch_controllers(["l_arm_cart_imped_controller"], 
                                            ["l_arm_controller"]
                                            )
        try:
            self.haptics_service.call()
        except rospy.ServiceException, e:
            rospy.logwarn("Got an exception %s, probably the service took too long",
                          e)
        
        if self.whicharm.startswith("right"):
            self.manager.switch_controllers(["r_arm_controller"],
                                            ["r_arm_cart_imped_controller"], 
                                            )
        else:
            self.manager.switch_controllers(["l_arm_controller"],
                                            ["l_arm_cart_imped_controller"], 
                                            )        
            
    def move_to_ideal_position(self, box,
                               ideal_pose = (0.80, 0.15),
                               ):
        box_pose = [box.pose.pose.position.x,
                    box.pose.pose.position.y,
                   ]                               
        
        displacement = (-(ideal_pose[0] - box_pose[0]),
                        -(ideal_pose[1] - box_pose[1]),
                        0)
        self.move_base.drive_to_displacement(displacement,
                                             frame_id=box.pose.header.frame_id)
        