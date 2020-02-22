#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Pose
import helper_pkg.utils as utils
from helper_pkg.utils import Constants
import helper_pkg.PredictionManagement as pm


class ReactivePrediction(pm.PredictionAlgorithm):
    DRONE_POSITIONS = 1
    TARGET_POSITIONS = 2
    INPUT_TYPE = Constants.InputOutputParameterType.pose
    OUTPUT_TYPE = Constants.InputOutputParameterType.pose
    MAP_PRESENCE = Constants.MapPresenceParameter.yes

    def __init__(self):
        super(ReactivePrediction, self).__init__()
        drone_configuration = rospy.get_param("drone_configuration")
        target_configuration = rospy.get_param("target_configuration")
        self._prediction_algorithm_name = "Reactive prediction"
        self._time_epsilon = 0.1
        self._ground_dist_limit = 0.2
        self._dronet_crop = 200
        self._target_image_height = drone_configuration["dronet"]["target_size"]["y"]
        self._target_image_width = drone_configuration["dronet"]["target_size"]["x"]
        self._drone_size = drone_configuration["properties"]["size"]
        self._hfov = drone_configuration["camera"]["hfov"]
        self._image_height = drone_configuration["camera"]["image_height"]
        self._image_width = drone_configuration["camera"]["image_width"]
        self._focal_length = self._image_width / (2.0 * np.tan(self._hfov / 2.0))
        self._vfov = np.arctan2(self._image_height / 2, self._focal_length)
        self._image_ration = self._image_width / float(self._image_height)
        self._focal_length_height = self._focal_length / self._image_ration
        self._max_target_speed = target_configuration["strategies"]["max_velocity"]
        self._max_target_speed_constant = 0.99
        self._target_speed_change_limit = 0.4
        self._low_target_speed_limit = 0.25
        self._max_distance_limit_for_random_position = 5.0
        self.set_min_max_dist(1)
        self.set_camera_angle_limits()
        self.track_restrictions()
        file_map_id = int(self._obstacles_file_path.split("/")[-2].split("-")[-1])
        self.f = open("tracking_test/drone_log_file_{}.txt".format(file_map_id), "a")
        self.f.write("------------------------------\n")

    def set_min_max_dist(self, r):
        mi, ma = 0, 0
        if r == 1:
            mi, ma = 6, 18
        self._min_dist, self._max_dist = mi, ma

    def set_camera_angle_limits(self):
        vertical_ratio = self._target_image_height / self._dronet_crop
        self._min_dist_alt_angle = np.pi / 2 - self._vfov / vertical_ratio
        self._max_dist_alt_angle = np.pi / 2 - self._vfov / (2 * vertical_ratio)
        horizontal_ratio = self._target_image_width / self._dronet_crop
        self._yaw_camera_limit = self._hfov / (2 * horizontal_ratio)

    def track_restrictions(self):
        self._min_dist, self._max_dist = 4, 30

    def recommended_dist(self, target_speed):
        diff_dist = self._max_dist - self._min_dist
        if target_speed < self._low_target_speed_limit:
            speed_ratio = 0
        else:
            speed_ratio = np.clip(target_speed / self._max_target_speed, 0, 1)
        return diff_dist * speed_ratio + self._min_dist

    def get_altitude_limits(self, recommended_dist):
        min_altitude = 3
        max_altitude = 15
        max_v = recommended_dist * np.cos(self._min_dist_alt_angle)
        # print("Min_dist_alt, max_dist_alt", self._min_dist_alt_angle, self._max_dist_alt_angle)
        min_v = recommended_dist * np.cos(self._max_dist_alt_angle)
        max_v = np.clip(max_v, min_altitude, max_altitude)
        min_v = np.clip(min_v, min_altitude, max_v)
        return min_v, max_v

    def recommended_altitude(self, recommended_dist):
        min_v, max_v = self.get_altitude_limits(recommended_dist)
        ratio = (recommended_dist - self._min_dist) / (self._max_dist - self._min_dist)
        numerator = max_v - min_v
        return ratio * numerator + min_v

    def recommended_ground_dist_alt(self, recommended_dist, alt):
        # print("Min_v, max_v", min_v, max_v)
        ground_dist = np.sqrt(np.square(recommended_dist) - np.square(alt))
        return ground_dist

    def recommended_vcamera(self, ground_dist, alt):
        return -(np.pi / 2 - np.arctan2(ground_dist, alt))

    def image_target_center_dist(self, target_information):
        # print(target_information)
        if target_information is not None and target_information.quotient > 0:
            # shift of target in image from image axis
            diffs = [self._image_width / 2 - target_information.centerX,
                     self._image_height / 2 - target_information.centerY]
            # abs values of shifts
            # abs_diffs = np.abs(diffs)
            # sign for shift of camera
            # signs = np.sign(diffs)
            # shift angle of target in image for camera
            angles = np.arctan2(diffs, [self._focal_length, self._focal_length_height])
            # print("diffs", diffs)
            # print("self._focal_length", self._focal_length, "self._focal_length_height", self._focal_length_height)
            return angles[0], angles[1]
        return 0, float("inf")

    def recommended_yaw(self, drone_next_pose, target_next_pose):
        return utils.Math.calculate_yaw_from_points(drone_next_pose.position.x, drone_next_pose.position.y,
                                                    target_next_pose.position.x, target_next_pose.position.y)

    def recommended_camera_orientation(self, ground_distance, altitude, camera_yaw, camera_pitch, target_information):
        angle_x, angle_y = self.image_target_center_dist(target_information)
        # target isn't in front of camera
        if angle_y == float("inf"):
            angle_y = self.recommended_vcamera(ground_distance, altitude)
        else:
            angle_x += camera_yaw
            angle_y += camera_pitch
        return angle_x, angle_y

    def modify_ground_dist(self, current_ground_dist, recommended_ground_dist, point):
        if abs(current_ground_dist - recommended_ground_dist) > self._ground_dist_limit:
            ratio = current_ground_dist / recommended_ground_dist
            modified_point = point * ratio
            modified_point.z = point.z
            return modified_point
        return point

    def vector_target_drone(self, current_drone_position, next_target_position):
        x_y_vector = [current_drone_position.pose.position.x - next_target_position.pose.position.x,
                      current_drone_position.pose.position.y - next_target_position.pose.position.y]
        return utils.Math.normalize(np.array(x_y_vector))

    def values_from_kwargs(self, kwargs):
        if "camera_yaw" in kwargs:
            camera_yaw = kwargs["camera_yaw"]
        else:
            camera_yaw = 0

        if "camera_pitch" in kwargs:
            camera_pitch = kwargs["camera_pitch"]
        else:
            camera_pitch = 0

        if "target_information" in kwargs:
            target_information = kwargs["target_information"]
        else:
            target_information = None
        if "recommended_altitude" in kwargs:
            recommended_altitude = kwargs["recommended_altitude"]
        else:
            recommended_altitude = -1
        return camera_yaw, camera_pitch, target_information, recommended_altitude

    def check_maximal_target_speed(self, current_target_speed):
        if abs(current_target_speed - self._max_target_speed) > self._target_speed_change_limit:
            self._max_target_speed = current_target_speed
        self._max_target_speed = self._max_target_speed * self._max_target_speed_constant + (
                1 - self._max_target_speed_constant) * current_target_speed

    def compute_target_speed(self, target_positions):
        target_next_pose = target_positions[1]
        target_last_pose = target_positions[0]
        speed_vector = np.array([target_next_pose.pose.position.x - target_last_pose.pose.position.x,
                                 target_next_pose.pose.position.y - target_last_pose.pose.position.y])
        target_predicted_speed = np.clip(np.hypot(speed_vector[0], speed_vector[1]), 0, self._max_target_speed)
        return target_predicted_speed

    def is_approching(self, drone_positions, target_positions):
        drone_point = utils.Math.Point(drone_positions[0].pose.position.x, drone_positions[0].pose.position.y)
        target_point = utils.Math.Point(target_positions[0].pose.position.x, target_positions[0].pose.position.y)
        target_next_point = utils.Math.Point(target_positions[1].pose.position.x, target_positions[1].pose.position.y)
        current_distance = utils.Math.euclidian_distance(drone_point, target_point)
        next_distance = utils.Math.euclidian_distance(drone_point, target_next_point)
        return current_distance > next_distance

    def drone_target_distance(self, drone_positions, target_positions):
        drone_point = utils.Math.Point(drone_positions[0].pose.position.x, drone_positions[0].pose.position.y)
        target_next_point = utils.Math.Point(target_positions[1].pose.position.x, target_positions[1].pose.position.y)
        return utils.Math.euclidian_distance(drone_point, target_next_point)

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        camera_yaw, camera_pitch, target_information, minimum_altitude = self.values_from_kwargs(kwargs)
        target_predicted_speed = self.compute_target_speed(target_positions)
        r_dist = self.recommended_dist(target_predicted_speed)
        distance = self.drone_target_distance(drone_positions, target_positions)
        alt = self.recommended_altitude(r_dist)
        if minimum_altitude > alt:
            alt = minimum_altitude
        pose = Pose()
        target_drone_vector = []
        if r_dist <= distance:
            target_drone_vector = self.vector_target_drone(drone_positions[0], target_positions[1])
            r_ground_dist = self.recommended_ground_dist_alt(r_dist, alt)
            next_position = target_drone_vector * r_ground_dist
            pose.position.x = next_position[0]
            pose.position.y = next_position[1]
        else:
            r_ground_dist = self.recommended_ground_dist_alt(distance, alt)
            pose.position.x = drone_positions[0].pose.position.x
            pose.position.y = drone_positions[0].pose.position.y
        pose.position.z = alt
        r_hcamera, r_vcamera = self.recommended_camera_orientation(r_ground_dist, alt, camera_yaw, camera_pitch,
                                                                   target_information)
        r_yaw = self.recommended_yaw(pose, target_positions[1].pose)
        pose.orientation.x = r_hcamera
        pose.orientation.y = r_vcamera
        pose.orientation.z = r_yaw
        print("Drone position:", drone_positions[0])
        print("Next target position:", target_positions[1])
        print("Next drone position:", pose)
        print("Distance:", r_dist, ",Ground distance:", r_ground_dist, ", altitude:", alt, ", yaw:", r_yaw)
        print("Hcamera:", r_hcamera, ",vcamera:", r_vcamera)
        print("Target speed:", target_predicted_speed, "target drone vector", target_drone_vector)
        map_point = map.map_point_from_real_coordinates(pose.position.x, pose.position.y, pose.position.z)
        if not map.is_free_for_drone(map_point):
            pose.position.x = target_positions[1].pose.position.x
            pose.position.y = target_positions[1].pose.position.y
        self.f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(drone_positions[0].pose.position.x,
                                                                                   drone_positions[0].pose.position.y,
                                                                                   drone_positions[0].pose.position.z,
                                                                                   drone_positions[
                                                                                       0].pose.orientation.x,
                                                                                   drone_positions[
                                                                                       0].pose.orientation.y,
                                                                                   drone_positions[
                                                                                       0].pose.orientation.z,
                                                                                   int(
                                                                                       target_information.quotient > 0.7),
                                                                                   target_positions[0].pose.position.x,
                                                                                   target_positions[0].pose.position.y,
                                                                                   target_positions[0].pose.position.z,
                                                                                   target_positions[
                                                                                       0].pose.orientation.x,
                                                                                   target_positions[
                                                                                       0].pose.orientation.y,
                                                                                   target_positions[
                                                                                       0].pose.orientation.z))
        return pose

    def state_variables(self, drone_positions, target_positions, map, **kwargs):
        return {}
