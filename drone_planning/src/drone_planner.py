#!/usr/bin/env python
import rospy
import copy
import sys
import json
from collections import deque, namedtuple
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import TwistWithCovariance, TransformStamped, Pose, Vector3, PoseStamped
from std_msgs.msg import Bool, Float32
from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
import moveit_commander
from pesat_msgs.msg import Notification, ImageTargetInfo, CameraShot, CameraUpdate
from pesat_msgs.srv import PredictionMaps, PositionPrediction, CameraRegistration, GoalState, PositionRequest
import helper_pkg.utils as utils
from helper_pkg.utils import Constants
import helper_pkg.PredictionManagement as pm

noti = namedtuple("Notification", ["serial_number", "time", "likelihood", "x", "y", "dx", "dy"])


class Section(object):
    def __init__(self):
        super(Section, self).__init__()
        self._form = None
        self._score = None
        self._location = None

    @property
    def form(self):
        return self._form

    @form.setter
    def form(self, value):
        self._form = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        self._location = value


class SectionAlgorithm(pm.PredictionAlgorithm):
    DRONE_POSITIONS = 1
    TARGET_POSITIONS = 1
    INPUT_TYPE = Constants.InputOutputParameterType.pose
    OUTPUT_TYPE = Constants.InputOutputParameterType.pose
    MAP_PRESENCE = Constants.MapPresenceParameter.yes

    def __init__(self):
        super(SectionAlgorithm, self).__init__()
        self._plan = []

    def find_traces(self, section):
        return []

    def divide_to_subsections(self, section, traces):
        return []

    def score_subsections(self, subsections):
        return []

    def divide_and_score(self, section):
        traces = self.find_traces(section)
        subsections = self.divide_to_subsections(section, traces)
        scores = self.score_subsections(subsections)
        return scores

    def create_main_section(self, world_map):
        return Section()

    def order_sections(self, scores):
        return []

    def create_plan(self, ordered_sections):
        return []

    def score_section(self, section):
        unproccesed_sections = [section]
        scores = []
        while len(unproccesed_sections) > 0:
            section = scores.pop()
            subsections = self.divide_and_score(section)
            if len(subsections) == 1:
                scores.append(subsections[0])
            else:
                unproccesed_sections.extend(subsections)
        return scores

    def main(self, map):
        section = self.create_main_section(map)
        scores = self.score_section(section)
        ordered_sections = self.order_sections(scores)
        plan = self.create_plan(ordered_sections)
        return plan

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        drone_position = drone_positions[0]
        # get time
        # get current position
        # from plan compute position in given time
        # return position
        p = Pose()
        return p


class PlannerBase(pm.PredictionManagement):
    def __init__(self):
        rospy.init_node('planner', anonymous=False)
        super(PlannerBase, self).__init__()
        target_configuration = rospy.get_param("target_configuration")
        logic_configuration = rospy.get_param("logic_configuration")
        self._prediction_systems = [SectionAlgorithm()]
        self._predicted_positions = [[]]
        self._last_prediction_update_time = [utils.Math.rounding(rospy.Time.now().to_sec())]
        self._default_system = 0
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self.tfBuffer)
        opponent_service_name = target_configuration["localization"]["position_in_time_service"]
        self.load_opponent_position_service(opponent_service_name)
        position_in_time_service_name = logic_configuration["planning"]["position_in_time_service"]
        self._target_position_offset = 0
        self._drone_position_offset = 1
        self._s = rospy.Service(position_in_time_service_name, PositionRequest, self.next_position_callback)

    def next_position_callback(self, data):
        rounded_time = utils.Math.rounding(data.header.stamp.to_sec())
        self.set_default_prediction_alg(data.algorithm_index)
        position, _ = self.get_position_in_time(rounded_time, data.refresh, center=data.center)
        self._prediction_localization.set_recommended_prediction_alg()
        return [position]

    def compute_recommended_prediction_alg(self):
        return 0

    def prepare_kwargs(self, time, drone_positions, target_positions, world_map, input_map_type):
        return {}

    def get_position_from_history(self, time):
        rounded_time = utils.Math.rounding(time)
        try:
            rospy_time = rospy.Time.from_sec(rounded_time)
            trans_map_drone = self.tfBuffer.lookup_transform(self._map_frame, self._drone_base_link_frame,
                                                             rospy_time)
            trans_drone_camera = self.tfBuffer.lookup_transform(self._drone_base_link_frame,
                                                                self._camera_base_link_frame, rospy_time)
            explicit_quat = [trans_map_drone.transform.rotation.x, trans_map_drone.transform.rotation.y,
                             trans_map_drone.transform.rotation.z, trans_map_drone.transform.rotation.w]
            (_, _, drone_yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
            explicit_quat = [trans_drone_camera.transform.rotation.x, trans_drone_camera.transform.rotation.y,
                             trans_drone_camera.transform.rotation.z, trans_drone_camera.transform.rotation.w]
            (_, camera_pitch, camera_yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy_time
            pose_stamped.header.frame_id = self._map_frame
            pose_stamped.pose.position.x = trans_map_drone.transform.translation.x
            pose_stamped.pose.position.y = trans_map_drone.transform.translation.y
            pose_stamped.pose.position.z = trans_map_drone.transform.translation.z
            pose_stamped.pose.orientation.x = camera_yaw
            pose_stamped.pose.orientation.y = camera_pitch
            pose_stamped.pose.orientation.z = drone_yaw
            return pose_stamped
        except (
                tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as excs:
            rospy.loginfo(str(excs))
            return None

    def get_main_type(self):
        return self._drone_type

    def opponent_service_for_drone_positions_necessary(self):
        return False

    def opponent_service_for_target_positions_necessary(self):
        return True

    def update(self):
        pass


if __name__ == '__main__':
    planner = PlannerBase()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        planner.update()
        rate.sleep()
