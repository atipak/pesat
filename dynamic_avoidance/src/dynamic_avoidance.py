#!/usr/bin/env python
import rospy
from collections import deque
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Range
import dronet_perception.msg as dronet_perception
from pesat_msgs.srv import PositionRequest
import helper_pkg.utils as utils
import helper_pkg.PredictionManagement as pm
from algorithms.dynamic_avoidance_algorithm import AvoidanceAlgorithm

# head -n $(grep -n "^0" avoidance_log_file.txt | tail -1 | grep -o -E '[0-9]+' | head -1) avoidance_log_file.txt 

class AvoidingManagment(pm.PredictionManagement):
    def __init__(self):
        rospy.init_node('dynamic_avoidance', anonymous=False)
        super(AvoidingManagment, self).__init__()
        logic_configuration = rospy.get_param("logic_configuration")
        environment_configuration = rospy.get_param("environment_configuration")
        drone_configuration = rospy.get_param("drone_configuration")
        self._low_bound = logic_configuration["avoiding"]["low_bound"]  # 0.7
        self._drone_base_link_frame = drone_configuration["properties"]["base_link"]
        self._map_frame = environment_configuration["map"]["frame"]
        self._camera_base_link_frame = drone_configuration["camera"]["base_link"]
        self._br = tf2_ros.TransformBroadcaster()
        self._prediction_systems = [AvoidanceAlgorithm()]
        self.prepare_structures()
        self._default_system = 0
        self._last_collision_predictions = deque(maxlen=10)
        self._last_sonar_predictions = deque(maxlen=10)
        self.init_dronet(drone_configuration)
        self._probability_records = deque(maxlen=10)
        self._sonar_records = deque(maxlen=10)
        self._probability_records.append(0.0)
        self._sonar_records.append(0.0)
        self._last_update = utils.Math.rounding(rospy.Time.now().to_sec())
        self._last_prediction = None
        self._last_state_variables = None
        self._speed_estemations = deque(maxlen=5)
        self._speed_estemation = 0
        self._next_estimate = 0.0
        self._collision_timeout = 0.0
        self._recommended_height_timeout = 0.0
        rospy.Subscriber(drone_configuration["sensors"]["sonar_topic"], Range, self.callback_sonar)
        self._pub_collision = rospy.Publisher(logic_configuration["avoiding"]["collision_topic"], Float32,
                                              queue_size=10)
        self._pub_recommended_altitude = rospy.Publisher(logic_configuration["avoiding"]["recommended_altitude_topic"],
                                                         Float32,
                                                         queue_size=10)
        self._s = rospy.Service(logic_configuration["avoiding"]["position_in_time_service"], PositionRequest,
                                self.next_position_callback)

    def init_dronet(self, drone_configuration):
        self.pub_dronet = rospy.Publisher(drone_configuration["dronet"]['state_change_topic'], Bool, queue_size=10)
        self.dronet_cmd_vel = None
        self.dronet_prediction = None
        rospy.Subscriber(drone_configuration["dronet"]["cnn_prediction_topic"], dronet_perception.CNN_out,
                         self.callback_dronet_prediction)
        self.pub_dronet.publish(True)

    # callbacks
    def next_position_callback(self, data):
        position = self._last_prediction
        return [position]

    def callback_dronet_prediction(self, data):
        self._last_collision_predictions.append(data.collision_prob)
        self.dronet_prediction = data

    def callback_sonar(self, data):
        self._last_sonar_predictions.append(data.range)

    def sonar_depth(self, sonar_range, drone_height):
        return drone_height - sonar_range

    def get_position_in_time_plus(self):
        pass

    def add_probability_record(self, cycle_passed):
        avg_prob = np.average(self._last_collision_predictions)
        if cycle_passed:
            self._probability_records.append(avg_prob)

    def add_sonar_record(self, cycle_passed):
        avg_prob = np.average(self._last_sonar_predictions)
        if cycle_passed:
            add_new_value = True
            try:
                trans_map_drone = self._tfBuffer.lookup_transform(self._map_frame, self._drone_base_link_frame,
                                                                  rospy.Time())
                explicit_quat = [trans_map_drone.transform.rotation.x, trans_map_drone.transform.rotation.y,
                                 trans_map_drone.transform.rotation.z, trans_map_drone.transform.rotation.w]
                (drone_roll, drone_pitch, drone_yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
                if abs(drone_roll) > 0.9 or abs(drone_pitch) > 0.9:
                    add_new_value = False
            except (
                    tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as excs:
                pass
            if add_new_value or len(self._sonar_records) == 0:
                self._sonar_records.append(avg_prob)
            else:
                self._sonar_records.append(self._sonar_records[-1])

    def step(self):
        if self._next_estimate < rospy.Time.now().to_sec() and rospy.Time.now().to_sec() - 2 > 0:
            self._next_estimate += 1
            now = self.get_position_from_history(rospy.Time().to_sec())
            last = self.get_position_from_history(now.header.stamp.to_sec() - 1)
            if now is not None and last is not None:
                diff_x = now.pose.position.x - last.pose.position.x
                diff_y = now.pose.position.x - last.pose.position.z
                diff_z = now.pose.position.x - last.pose.position.z
                dist = np.sqrt(np.square(diff_x) + np.square(diff_y) + np.square(diff_z))
                self._speed_estemations.append(dist)
                self._speed_estemation = np.sum(self._speed_estemations) / len(self._speed_estemations)
        # print("------------------------")
        self.pub_dronet.publish(True)
        if self._last_state_variables is None or (self._last_state_variables is not None and ("state" not in self._last_state_variables or ("state" in self._last_state_variables and self._last_state_variables["state"] ==1))):
            avg_prob = np.average(self._last_collision_predictions)
        else:
            avg_prob = 0.0
        self._pub_collision.publish(avg_prob)
        rounded_time = utils.Math.rounding(rospy.Time.now().to_sec())
        rounded_next_time = utils.Math.rounding(rospy.Time.now().to_sec() + 0.5)
        cycle_passed = rounded_time != self._last_update
        self.add_probability_record(cycle_passed)
        self.add_sonar_record(cycle_passed)
        self._last_prediction, self._last_state_variables = self.get_position_in_time(rounded_next_time)
        # print("self._last_prediction: {}, self._last_state_variables: {}".format(
        #     utils.DataStructures.pointcloud2_to_array(self._last_prediction), self._last_state_variables))
        if cycle_passed:
            self._last_update = rounded_time
        if self._last_state_variables is not None:
            if "recommended" in self._last_state_variables and self._last_state_variables["recommended"]:
                self._pub_recommended_altitude.publish(self._last_state_variables["recommended"])
            else:
                self._pub_recommended_altitude.publish(-1)

    def compute_recommended_prediction_alg(self):
        return 0

    def prepare_drone_positions(self, last_time, positions_count):
        position_newer = self.get_pose_stamped(rospy.Time())
        position_newer_pointcloud = utils.DataStructures.array_to_pointcloud2(np.array([[position_newer.pose.position.x,
                                                                                         position_newer.pose.position.y,
                                                                                         position_newer.pose.position.z,
                                                                                         position_newer.pose.orientation.x,
                                                                                         position_newer.pose.orientation.y,
                                                                                         position_newer.pose.orientation.z,
                                                                                         0]]),
                                                                              stamp=position_newer.header.stamp)
        position_older = position_newer
        position_older_pointcloud = position_newer_pointcloud
        sec_time = position_newer.header.stamp.to_sec() - 0.5
        position_older_try = self.get_pose_stamped(rospy.Time.from_sec(sec_time), False)
        if position_older_try is not None:
            position_older = position_older_try
            position_older_pointcloud = utils.DataStructures.array_to_pointcloud2(
                np.array([[position_older.pose.position.x,
                           position_older.pose.position.y,
                           position_older.pose.position.z,
                           position_older.pose.orientation.x,
                           position_older.pose.orientation.y,
                           position_older.pose.orientation.z,
                           0]]), stamp=rospy.Time().from_sec(sec_time))
        if position_newer is not None and position_older is not None:
            return [position_newer_pointcloud, position_older_pointcloud]
        else:
            return None

    def get_pose_stamped(self, rospy_time, verbose=True):
        try:
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
            pose_stamped.header.stamp = trans_drone_camera.header.stamp
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
            if verbose:
                rospy.loginfo("Exception by getting pose stamped: " + str(excs))
            return None

    def get_position_from_history(self, time):
        rounded_time = utils.Math.rounding(time)
        rospy_time = rospy.Time.from_sec(rounded_time)
        return self.get_pose_stamped(rospy_time)

    def prepare_kwargs(self, time, drone_positions, target_positions, world_map):
        sonar_list = []
        collision_list = []
        if drone_positions is not None:
            for i in range(0, len(drone_positions)):
                stamped_pose = drone_positions[i]
                pose_time = stamped_pose.header.stamp.to_sec()
                rounded_pose_time = utils.Math.floor_rounding(pose_time)
                # print("rounded_pose_time", rounded_pose_time)
                # print("self._last_update", self._last_update, "self._probability_records", self._probability_records)
                index = len(self._sonar_records) - 1 - int((self._last_update - rounded_pose_time) / 0.5)
                # print("index", index, "int", int((self._last_update - pose_time) / 0.5))
                if index < 0 or index >= len(self._sonar_records):
                    raise Exception(
                        "Requested time for sonar and prediction data is outside possibilities! Length of sonar records: {}, requested: {}".format(
                            len(self._sonar_records), index))
                sonar_list.append(self._sonar_records[index])
                collision_list.append(self._probability_records[index])
        kwargs = {"sensor_data": sonar_list, "collision_data": collision_list, "speed_estimate": self._speed_estemation}
        # print("self._last_state_variables", self._last_state_variables)
        if self._last_state_variables is not None:
            kwargs.update(self._last_state_variables)
        return kwargs

    def opponent_service_for_drone_positions_necessary(self):
        return False

    def opponent_service_for_target_positions_necessary(self):
        return False

    def get_main_type(self):
        return self._drone_type

    def check_boundaries_target(self, count_needed):
        return self.topic_boundaries(count_needed, self._target_type)

    def check_boundaries_drone(self, count_needed):
        return self.tf_boundaries(count_needed, self._drone_type)

    def get_topic_bounds(self):
        return self._target_earliest_time, self._target_latest_time, self._items_count


if __name__ == "__main__":
    da = AvoidingManagment()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        da.step()
        rate.sleep()
