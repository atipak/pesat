#!/usr/bin/env python
import rospy
from collections import deque, namedtuple
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose, Point, PoseStamped
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Range
import dronet_perception.msg as dronet_perception
from pesat_msgs.srv import PositionRequest
import helper_pkg.utils as utils
import helper_pkg.PredictionManagement as pm
from helper_pkg.utils import Constants


class AvoidanceAlgorithm(pm.PredictionAlgorithm):
    DRONE_POSITIONS = 2
    TARGET_POSITIONS = 0
    INPUT_TYPE = Constants.InputOutputParameterType.pose
    OUTPUT_TYPE = Constants.InputOutputParameterType.pose
    MAP_PRESENCE = Constants.MapPresenceParameter.yes

    def __init__(self):
        super(AvoidanceAlgorithm, self).__init__()
        logic_configuration = rospy.get_param("logic_configuration")
        drone_configuration = rospy.get_param("drone_configuration")
        self._prediction_algorithm_name = "Dynamic avoidance algorithm"
        self._low_bound = logic_configuration["avoiding"]["low_bound"]
        self._minimum_distance = logic_configuration["avoiding"]["minimum_distance"]
        self._max_height = logic_configuration["conditions"]["max_height"]
        self._shift_constant = 1
        self._smaller_shift_constant = 0.1
        self._drone_size = drone_configuration["properties"]["size"]
        self._sonar_change_limit = 0.3
        self._collision_change_limit = 0.1
        self._timeout = 50

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        pose = Pose()
        drone_position = drone_positions[0]
        collision_probability_list = kwargs["collision_data"]
        collision_probability = collision_probability_list[0]
        sensor_data_list = kwargs["sensor_data"]
        sensor_data = sensor_data_list[0]
        sonar_change = 0
        if "sonar_change" in kwargs:
            sonar_change = kwargs["sonar_change"]
        collision_change = 0
        if "collision_change" in kwargs:
            collision_change = kwargs["collision_change"]
        shift_z = 0
        state = 0
        if collision_probability > self._low_bound:
            state = 1  # obstacle in front of drone
        else:
            if collision_change > 0 and sonar_change == 0:
                state = 2  # drone in front of and above obstacle
            if collision_change == 0 and sonar_change > 0:
                state = 3  # drone is above obstacle
            if collision_change == 0 and sonar_change <= 0:
                state = 0  # free move
        if state == 1:
            if drone_position.pose.position.z + self._shift_constant < self._max_height:
                shift_z = self._shift_constant  # lift drone
            else:
                shift_z = 0  # maximal altitude reached
        elif state == 2 or state == 3:
            shift_z = 0  # height is ok
        elif state == 0:
            shift_z -= (drone_position.pose.position.z + 1)  # no height requiremnets
        if self._minimum_distance > sensor_data:
            shift_z = self._smaller_shift_constant  # drone is too low above obstacle
        # print(drone_position)
        #print("shift_z", shift_z, "sensor_data", sensor_data)
        #print("collision_probability", collision_probability)
        #print("state", state)
        pose.position.z = drone_position.pose.position.z + shift_z
        pose.position.x = drone_position.pose.position.x
        pose.position.y = drone_position.pose.position.y
        if "stable_point" in kwargs and kwargs["stable_point"] is not None:
            pose.position.x = kwargs["stable_point"].x
            pose.position.y = kwargs["stable_point"].y
        pose.orientation.x = drone_position.pose.orientation.x
        pose.orientation.y = drone_position.pose.orientation.y
        pose.orientation.z = drone_position.pose.orientation.z
        pose.orientation.w = drone_position.pose.orientation.w
        return pose

    def state_variables(self, drone_positions, target_positions, map, **kwargs):
        sonar_change = 0
        if "sonar_change" in kwargs:
            sonar_change = kwargs["sonar_change"]
        collision_change_time = 0
        if "collision_change" in kwargs:
            collision_change_time = kwargs["collision_change"]
        collision_probability_list = kwargs["collision_data"]
        collision_probability = collision_probability_list[0]
        older_collision_probability = collision_probability_list[1]
        sensor_data_list = kwargs["sensor_data"]
        sensor_data = sensor_data_list[0]
        older_sensor_data = sensor_data_list[1]
        stable_point = None
        if "stable_point" in kwargs:
            stable_point = kwargs["stable_point"]
        #print("kwargs", kwargs)
        # to 1
        if collision_probability > self._low_bound and stable_point is None:
            stable_point = Point()
            stable_point.x = drone_positions[0].pose.position.x
            stable_point.y = drone_positions[0].pose.position.y
        # to 2
        if collision_change_time == 0 and sonar_change == 0 and \
                older_collision_probability > self._low_bound > collision_probability and abs(
                older_collision_probability - collision_probability) > self._collision_change_limit:
            #print("To 2")
            stable_point = None
            sonar_change = 0
            collision_change_time = rospy.Time.now().to_sec()
        # to 3
        elif collision_change_time > 0 and sonar_change == 0 and abs(
                older_sensor_data - sensor_data) > self._sonar_change_limit:
            #print("To 3")
            sonar_change = older_sensor_data - sensor_data
            collision_change_time = 0
        # to 4
        elif collision_change_time == 0 and sonar_change > 0 and abs(
                older_sensor_data - sensor_data) > self._sonar_change_limit:
            #print("To 4")
            sonar_change = older_sensor_data - sensor_data
            collision_change_time = 0
        # to "5"
        elif collision_change_time > 0 and sonar_change == 0 and abs(
                collision_change_time - rospy.Time.now().to_sec()) > self._timeout:
            #print("To 5")
            collision_change_time = 0
            sonar_change = 0
        return {"sonar_change": sonar_change, "collision_change": collision_change_time, "stable_point": stable_point}


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
            self._sonar_records.append(avg_prob)

    def step(self):
        #print("------------------------")
        self.pub_dronet.publish(True)
        avg_prob = np.average(self._last_collision_predictions)
        self._pub_collision.publish(avg_prob)
        rounded_time = utils.Math.rounding(rospy.Time.now().to_sec())
        rounded_next_time = utils.Math.rounding(rospy.Time.now().to_sec() + 0.5)
        cycle_passed = rounded_time != self._last_update
        self.add_probability_record(cycle_passed)
        self.add_sonar_record(cycle_passed)
        self._last_prediction, self._last_state_variables = self.get_position_in_time(rounded_next_time)
        if cycle_passed:
            self._last_update = rounded_time
        if self._last_prediction is not None:
            pred = utils.DataStructures.array_to_pose(utils.DataStructures.pointcloud2_to_array(self._last_prediction))
            self._pub_recommended_altitude.publish(pred.pose.position.z)
        else:
            self._pub_recommended_altitude.publish(-1)

    def compute_recommended_prediction_alg(self):
        return 0

    def prepare_drone_positions(self, last_time, positions_count):
        position_newer = self.get_pose_stamped(rospy.Time())
        position_older = position_newer
        sec_time = position_newer.header.stamp.to_sec() - 0.5
        position_older_try = self.get_pose_stamped(rospy.Time.from_sec(sec_time), False)
        if position_older_try is not None:
            position_older = position_older_try
        if position_newer is not None and position_older is not None:
            return [position_newer, position_older]
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
                    raise Exception("Requested time for sonar and prediction data is outside possibilities!")
                sonar_list.append(self._sonar_records[index])
                collision_list.append(self._probability_records[index])
        kwargs = {"sensor_data": sonar_list, "collision_data": collision_list}
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


if __name__ == "__main__":
    da = AvoidingManagment()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        da.step()
        rate.sleep()
