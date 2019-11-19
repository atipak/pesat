#!/usr/bin/env python
import rospy
import tf
import tf2_ros
from collections import namedtuple
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32
from pesat_msgs.srv import PredictionMaps, PositionPrediction, GoalState, PositionRequest
from pesat_msgs.msg import ImageTargetInfo
import helper_pkg.utils as utils
import helper_pkg.PredictionManagement as pm
from algorithms.drone_neural import DeepPrediction
from algorithms.drone_reactive import ReactivePrediction


class PredictionLocalization(pm.PredictionManagement):
    def __init__(self):
        rospy.init_node('track_system', anonymous=False)
        super(PredictionLocalization, self).__init__()
        target_configuration = rospy.get_param("target_configuration")
        drone_configuration = rospy.get_param("drone_configuration")
        logic_configuration = rospy.get_param("logic_configuration")
        self._prediction_systems = [ReactivePrediction(), DeepPrediction(1)]
        self.prepare_structures()
        self._default_system = 0
        enum = namedtuple("State", ["searching", "tracking"])
        self._states = enum(0, 1)
        self._state = self._states.tracking
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self._drone_base_link_frame = drone_configuration["properties"]["base_link"]
        self._camera_base_link_frame = drone_configuration["camera"]["base_link"]
        opponent_service_name = target_configuration["localization"]["position_in_time_service"]
        self.load_target_position_service(opponent_service_name)
        self._target_information = None
        self._target_position_offset = 0
        self._drone_position_offset = 1
        self._target_under_supervision = True
        self._target_under_supervision_for_lasttime = rospy.Time.now()
        self._target_without_supervision_for_lasttime = rospy.Time.now()
        self._track_timeout = logic_configuration["tracking"]["timeout"]
        self._planning_timeout = logic_configuration["planning"]["timeout"]
        self._target_earliest_time = rospy.Time.now().to_sec()
        self._target_latest_time = rospy.Time.now().to_sec()
        self._items_count = 0
        # earliest
        rospy.Subscriber(target_configuration["localization"]["earliest_time_topic"], Float32,
                         self.callback_target_earliest_time)
        # latest
        rospy.Subscriber(target_configuration["localization"]["latest_time_topic"], Float32,
                         self.callback_target_latest_time)
        # items count
        rospy.Subscriber(target_configuration["localization"]["items_count_topic"], Float32,
                         self.callback_target_items_count)
        rospy.Subscriber(target_configuration["localization"]["target_information"], ImageTargetInfo,
                         self.callback_target_information)
        self._s = rospy.Service(logic_configuration["tracking"]["position_in_time_service"], PositionRequest,
                                self.next_position_callback)

    def callback_target_earliest_time(self, data):
        self._target_earliest_time = data

    def callback_target_latest_time(self, data):
        self._target_latest_time = data

    def callback_target_items_count(self, data):
        self._items_count = data

    def get_topic_bounds(self):
        return self._target_earliest_time, self._target_latest_time, self._items_count

    def compute_recommended_prediction_alg(self):
        change_state = self.change_between_tracking_and_planning()
        if self._state == self._states.searching and change_state:
            self._state = self._states.tracking
        elif self._state == self._states.tracking and change_state:
            self._state = self._states.searching
        return 0

    def change_between_tracking_and_planning(self):
        if self._target_under_supervision:
            if self._state == self._states.searching:
                if abs(
                        rospy.Time.now().to_sec() - self._target_without_supervision_for_lasttime.to_sec()) > self._planning_timeout:
                    return True
                else:
                    return False
            else:
                self._target_under_supervision_for_lasttime = rospy.Time.now()
                return False
        else:
            if self._state == self._states.tracking:
                if abs(
                        rospy.Time.now().to_sec() - self._target_under_supervision_for_lasttime.to_sec()) > self._track_timeout:
                    return True
            else:
                self._target_without_supervision_for_lasttime = rospy.Time.now()
                return False

    def prepare_kwargs(self, time, drone_positions, target_positions, world_map):
        kwargs = {}
        if time == utils.Math.rounding(rospy.Time.now().to_sec() + 0.5):
            try:
                trans_drone_camera = self._tfBuffer.lookup_transform(self._drone_base_link_frame,
                                                                     self._camera_base_link_frame, rospy.Time())
                explicit_quat = [trans_drone_camera.transform.rotation.x, trans_drone_camera.transform.rotation.y,
                                 trans_drone_camera.transform.rotation.z, trans_drone_camera.transform.rotation.w]
                (_, camera_pitch, camera_yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
                kwargs["camera_yaw"] = camera_yaw
                kwargs["camera_pitch"] = camera_pitch
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as excs:
                rospy.loginfo("Exception during calculating of camera position:\n" + str(excs))
                kwargs["camera_yaw"] = 0
                kwargs["camera_pitch"] = 0
            kwargs["target_information"] = self._target_information
        return kwargs

    def get_position_from_history(self, time):
        rounded_time = utils.Math.rounding(time)
        try:
            rospy_time = rospy.Time.from_sec(rounded_time)
            trans_map_drone = self._tfBuffer.lookup_transform(self._map_frame, self._drone_base_link_frame,
                                                              rospy_time)
            trans_drone_camera = self._tfBuffer.lookup_transform(self._drone_base_link_frame,
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
            return utils.DataStructures.array_to_pointcloud2(np.array([[pose_stamped.pose.position.x,
                                                                       pose_stamped.pose.position.y,
                                                                       pose_stamped.pose.position.z,
                                                                       camera_yaw, camera_pitch, drone_yaw, 0]]))
        except (
                tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as excs:
            rospy.loginfo("Unsuccessful getting position from history. " + str(excs))
            return None

    def get_main_type(self):
        return self._drone_type

    def opponent_service_for_drone_positions_necessary(self):
        return False

    def opponent_service_for_target_positions_necessary(self):
        return True

    def get_history_positions_from_time(self, count, time, map_type=False, center=None):
        rounded_time = utils.Math.rounding(time)
        positions = []
        for _ in range(count):
            if center is None and map_type and count > 1:
                middle, _ = self.get_position_in_time(rounded_time, False, center)
                center = self.get_center_of_positions(center, [middle])
            position, _ = self.get_position_in_time(rounded_time, map_type, center)
            positions.append(position)
            rounded_time -= 0.5
        return positions

    def callback_target_information(self, data):
        self._target_information = data

    def next_position_callback(self, data):
        rounded_time = utils.Math.rounding(data.header.stamp.to_sec())
        # print(rounded_time)
        self.set_default_prediction_alg(data.algorithm_index)
        position, _ = self.get_position_in_time(rounded_time, renew=data.refresh)
        self.set_recommended_prediction_alg()
        # print("track", position)
        return [position]

    def is_target_supervision_by_drone(self):
        return True

    def check_boundaries_target(self, count_needed):
        return self.topic_boundaries(count_needed, self._target_type)

    def check_boundaries_drone(self, count_needed):
        return self.tf_boundaries(count_needed, self._drone_type)

    def launch(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self._target_under_supervision = self.is_target_supervision_by_drone()
            self.update_last_position_time(rospy.Time.now().to_sec())
            rate.sleep()


if __name__ == "__main__":
    tp = PredictionLocalization()
    tp.launch()
