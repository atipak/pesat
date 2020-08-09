#!/usr/bin/env python
import rospy
import numpy as np
import os
from geometry_msgs.msg import Pose
import helper_pkg.utils as utils
from helper_pkg.utils import Constants
import helper_pkg.PredictionManagement as pm
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist


class ReactivePrediction(pm.PredictionAlgorithm):
    DRONE_POSITIONS = 1
    TARGET_POSITIONS = 2
    INPUT_TYPE = Constants.InputOutputParameterType.pointcloud
    OUTPUT_TYPE = Constants.InputOutputParameterType.pointcloud
    MAP_PRESENCE = Constants.MapPresenceParameter.yes
    CURRENT, LAST, NEXT, TARGET = 0, 1, 2, 3

    def __init__(self):
        super(ReactivePrediction, self).__init__()
        drone_configuration = rospy.get_param("drone_configuration")
        target_configuration = rospy.get_param("target_configuration")
        self._prediction_algorithm_name = "Reactive prediction"
        self._time_epsilon = 0.1
        self._ground_dist_limit = 0.2
        self._dronet_crop = 200
        self._minimal_height = 3
        self._maximal_height = 10
        self._target_image_height = drone_configuration["dronet"]["target_size"]["y"]
        self._target_image_width = drone_configuration["dronet"]["target_size"]["x"]
        self._drone_size = drone_configuration["properties"]["size"]
        self._hfov = drone_configuration["camera"]["hfov"]
        self._camera_range = drone_configuration["camera"]["camera_range"]
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
        self._adaptive_altitude = True
        self._prediction_type = self.CURRENT
        self._max_distance_limit_for_random_position = 5.0
        self.set_min_max_dist(1)
        self.set_camera_angle_limits()
        self.track_restrictions()
        file_name = os.path.split(self._obstacles_file_path)[1][7:]
        self.f = open("tracking_test/drone_log_file_{}.txt".format(file_name), "w")

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
        self._min_dist, self._max_dist = 4, 20

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
        if self._adaptive_altitude:
            min_v, max_v = self.get_altitude_limits(recommended_dist)
            ratio = (recommended_dist - self._min_dist) / (self._max_dist - self._min_dist)
            numerator = max_v - min_v
            final_alt = ratio * numerator + min_v
            if final_alt <= 9.5:
                rounded_alt = self._minimal_height
            else:
                rounded_alt = self._maximal_height
        else:
            rounded_alt = self._minimal_height
        return rounded_alt

    def recommended_ground_dist_alt(self, recommended_dist, alt):
        # print("Min_v, max_v", min_v, max_v)
        # if altitude is set to minimum value and recommended_dist is set to current distance, it is possible that
        # the condition is satisfied and therefore we could get NaN. We set recommended dist to alt + 1
        if recommended_dist < alt:
            recommended_dist = alt + 1
        ground_dist = np.sqrt(np.square(recommended_dist) - np.square(alt))
        return ground_dist

    def recommended_vcamera(self, ground_dist, alt):
        return np.pi / 2 - np.arctan2(ground_dist, alt)

    def image_target_center_dist(self, target_information):
        # print(target_information)
        if target_information is not None and target_information.quotient > 0:
            # shift of target in image from image axis
            diffs = [self._image_width / 2 - target_information.centerX,
                     -1 * (self._image_height / 2 - target_information.centerY)]
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

    def recommended_yaw(self, drone_next_pose, next_target_position_xy):
        return utils.Math.calculate_yaw_from_points(drone_next_pose.position.x, drone_next_pose.position.y,
                                                    next_target_position_xy[0], next_target_position_xy[1])

    def recommended_camera_orientation(self, ground_distance, altitude, camera_yaw, camera_pitch, target_information):
        angle_x, angle_y = self.image_target_center_dist(target_information)
        # target isn't in front of camera
        if angle_y == float("inf"):
            angle_y = self.recommended_vcamera(ground_distance, altitude)
        else:
            angle_x += camera_yaw
            angle_y += camera_pitch
        angle_y = np.clip(angle_y, 0, None)
        return angle_x, angle_y

    def modify_ground_dist(self, current_ground_dist, recommended_ground_dist, point):
        if abs(current_ground_dist - recommended_ground_dist) > self._ground_dist_limit:
            ratio = current_ground_dist / recommended_ground_dist
            modified_point = point * ratio
            modified_point.z = point.z
            return modified_point
        return point

    def vector_target_drone(self, current_drone_position_pose, next_target_position_xy):
        x_y_vector = [current_drone_position_pose.position.x - next_target_position_xy[0],
                      current_drone_position_pose.position.y - next_target_position_xy[1]]
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
        if recommended_altitude < self._minimal_height:
            recommended_altitude = self._minimal_height
        return camera_yaw, camera_pitch, target_information, recommended_altitude

    def check_maximal_target_speed(self, current_target_speed):
        if abs(current_target_speed - self._max_target_speed) > self._target_speed_change_limit:
            self._max_target_speed = current_target_speed
        self._max_target_speed = self._max_target_speed * self._max_target_speed_constant + (
                1 - self._max_target_speed_constant) * current_target_speed

    def compute_target_speed(self, current_target_position_xy, next_target_position_xy, time_difference):
        time_conditioned_max_target_speed = time_difference * self._max_target_speed
        speed_vector = np.array([next_target_position_xy[0] - current_target_position_xy[0],
                                 next_target_position_xy[1] - current_target_position_xy[1]])
        target_predicted_speed = np.clip(np.hypot(speed_vector[0], speed_vector[1]), 0,
                                         time_conditioned_max_target_speed)
        return target_predicted_speed

    def is_approching(self, drone_positions, target_positions):
        drone_point = utils.Math.Point(drone_positions[0].pose.position.x, drone_positions[0].pose.position.y)
        target_point = utils.Math.Point(target_positions[0].pose.position.x, target_positions[0].pose.position.y)
        target_next_point = utils.Math.Point(target_positions[1].pose.position.x, target_positions[1].pose.position.y)
        current_distance = utils.Math.euclidian_distance(drone_point, target_point)
        next_distance = utils.Math.euclidian_distance(drone_point, target_next_point)
        return current_distance > next_distance

    def drone_target_distance(self, drone_position_pose, next_target_position_xy):
        drone_point = utils.Math.Point(drone_position_pose.position.x, drone_position_pose.position.y)
        target_next_point = utils.Math.Point(next_target_position_xy[0], next_target_position_xy[1])
        return utils.Math.euclidian_distance(drone_point, target_next_point)

    def last_drone_current_drone_distance(self, current_drone_position_pose, last_drone_position_pose):
        current_drone_point = utils.Math.Point(current_drone_position_pose.position.x,
                                               current_drone_position_pose.position.y)
        last_drone_point = utils.Math.Point(last_drone_position_pose.position.x, last_drone_position_pose.position.y)
        return utils.Math.euclidian_distance(current_drone_point, last_drone_point)

    def find_location(self, array, max_distance=5):
        X = np.array(zip(array[:, 0], array[:, 1]))
        Z = ward(pdist(X))
        clusters = fcluster(Z, t=max_distance, criterion='distance')
        numbers, counts = np.unique(clusters, return_counts=True)
        arg = np.argmax(counts)
        num = numbers[arg]
        max_cluster_points = array[clusters == num]
        position = np.sum(max_cluster_points, dtype=np.float, axis=0) / len(max_cluster_points)
        return position
        # probs = counts.astype(np.float)/np.sum(counts)

    def parse_last_position(self, kwargs):
        if "last_predicted_position" in kwargs:
            last_predicted_pose = Pose()
            last_predicted_pose.position.x = kwargs["last_predicted_position"][0, 0]
            last_predicted_pose.position.y = kwargs["last_predicted_position"][0, 1]
            last_predicted_pose.position.z = kwargs["last_predicted_position"][0, 2]
            last_predicted_pose.orientation.x = kwargs["last_predicted_position"][0, 3]
            last_predicted_pose.orientation.y = kwargs["last_predicted_position"][0, 4]
            last_predicted_pose.orientation.z = kwargs["last_predicted_position"][0, 5]
            last_type_prediction = kwargs["last_type"]
            return last_predicted_pose, last_type_prediction
        return None, self.CURRENT

    def init_transformation(self, target_positions, drone_positions, map, kwargs):
        """
        TRANSFORMATION INIT DATA
        """
        time_difference = np.abs(target_positions[0].header.stamp.to_sec() - target_positions[1].header.stamp.to_sec())
        next_target_positions_array = utils.DataStructures.pointcloud2_to_array(target_positions[1])
        current_target_positions_array = utils.DataStructures.pointcloud2_to_array(target_positions[0])
        current_drone_positions_array = utils.DataStructures.pointcloud2_to_array(drone_positions[0])
        next_target_position_xy = self.find_location(next_target_positions_array)
        current_target_position_xy = self.find_location(current_target_positions_array)
        current_drone_position_pose = utils.DataStructures.array_to_pose(current_drone_positions_array)
        camera_yaw, camera_pitch, target_information, minimum_altitude = self.values_from_kwargs(kwargs)
        last_predicted_pose = None
        map_next_target_position = map.map_point_from_real_coordinates(next_target_position_xy[0],
                                                                       next_target_position_xy[1], 1)
        # is last predicted position
        last_seen = False
        last_predicted_pose, last_type_prediction = self.parse_last_position(kwargs)
        if last_predicted_pose is not None:
            coordinates = map.faster_rectangle_ray_tracing_3d(last_predicted_pose, self._vfov, self._hfov,
                                                              self._camera_range)
            mask = coordinates[:, 0] == map_next_target_position.x
            if np.count_nonzero(coordinates[mask, 1] == map_next_target_position.y):
                last_seen = True
        # is target visible from current pose
        see = False
        coordinates = map.faster_rectangle_ray_tracing_3d(current_drone_position_pose, self._vfov, self._hfov,
                                                          self._camera_range)
        mask = coordinates[:, 0] == map_next_target_position.x
        if np.count_nonzero(coordinates[mask, 1] == map_next_target_position.y):
            see = True
        if last_predicted_pose is not None:
            last_current_position_distance = [
                np.abs(current_drone_position_pose.position.x - last_predicted_pose.position.x) + np.abs(
                    current_drone_position_pose.position.y - last_predicted_pose.position.y) +
                np.abs(current_drone_position_pose.position.z - last_predicted_pose.position.z),
                np.abs(current_drone_position_pose.orientation.z - last_predicted_pose.orientation.z)]
        else:
            last_current_position_distance = [0, 0]
        preparation = False
        if "final_prediction" in kwargs:
            preparation = kwargs["final_prediction"]
        return time_difference, next_target_position_xy, current_target_position_xy, \
               current_drone_position_pose, camera_yaw, camera_pitch, target_information, \
               minimum_altitude, last_predicted_pose, last_seen, see, last_type_prediction, \
               map_next_target_position, last_current_position_distance, preparation

    def data_for_calculation(self, current_target_position_xy, next_target_position_xy, time_difference,
                             current_drone_position_pose, minimum_altitude):
        """
           DATA FOR CALCULATION OF NEXT POSITION
        """
        target_predicted_speed = self.compute_target_speed(current_target_position_xy, next_target_position_xy,
                                                           time_difference)
        r_dist = self.recommended_dist(target_predicted_speed)
        distance = self.drone_target_distance(current_drone_position_pose, next_target_position_xy)
        alt = self.recommended_altitude(r_dist)
        if minimum_altitude > alt:
            alt = minimum_altitude
        print("Target speed:", target_predicted_speed)
        return r_dist, distance, alt

    def compute_parameters(self, r_dist, distance, target_information, see, last_type_prediction, last_r_dist,
                           last_distance, last_current_position_distance, preparation):
        """
        CONDITIONS
        """
        target_is_nearer_than_wanted = r_dist > distance
        last_position_is_nearer_than_wanted = last_r_dist > last_distance
        target_in_image = target_information is not None and target_information.quotient > 0
        stay_on_place = see and target_is_nearer_than_wanted
        print("tintw {}, lpintw {}, tii {}, sop {}".format(target_is_nearer_than_wanted,
                                                           last_position_is_nearer_than_wanted, target_in_image,
                                                           stay_on_place))

        """
        COMPUTE PARAMETERS BASED ON CONDITIONS
        """
        if preparation:
            return self.CURRENT, last_type_prediction
        if target_in_image:
            if stay_on_place:
                next_position_for_drone = self.CURRENT
            else:
                next_position_for_drone = self.NEXT
        else:
            print("conditions", last_type_prediction, last_current_position_distance)
            if last_type_prediction == self.TARGET and (
                    last_current_position_distance[0] > 1 or last_current_position_distance[1] > 0.2):
                next_position_for_drone = self.LAST
            else:
                next_position_for_drone = self.TARGET
        true_prediction = next_position_for_drone
        if last_type_prediction == self.TARGET and \
                (next_position_for_drone == self.TARGET or next_position_for_drone == self.LAST) or \
                (last_type_prediction == self.NEXT and next_position_for_drone == self.NEXT) or \
                (last_type_prediction == self.CURRENT and next_position_for_drone == self.CURRENT):
            # next_position_for_drone = self.LAST
            pass
        if true_prediction == self.NEXT:
            if last_type_prediction == self.TARGET and distance > self._max_dist:
                true_prediction = self.LAST
            elif (
                    last_type_prediction == self.NEXT or last_type_prediction == self.CURRENT) and distance > self._max_dist:
                true_prediction = self.TARGET

        return next_position_for_drone, true_prediction

    def compute_next_position(self, next_position_for_drone, last_predicted_pose, current_drone_position_pose,
                              alt, distance, next_target_position_xy, r_dist, camera_yaw, camera_pitch,
                              target_information, map, current_target_position_xy, map_next_target_position,
                              true_prediction):
        """
        COMPUTE NEW POSITION FOR DRONE BASED ON PARAMETRS
        """
        target_drone_vector = []
        pose = Pose()
        if next_position_for_drone == self.LAST:
            # print("last alt:", last_predicted_pose.position.z)
            pose = last_predicted_pose
            if true_prediction != self.LAST:
                r_ground_dist = self.recommended_ground_dist_alt(distance, alt)
                pose.orientation.x, pose.orientation.y = self.recommended_camera_orientation(r_ground_dist, alt,
                                                                                             camera_yaw, camera_pitch,
                                                                                             target_information)
                pose.orientation.z = self.recommended_yaw(pose, next_target_position_xy)
        elif next_position_for_drone == self.CURRENT or next_position_for_drone == self.NEXT:
            if next_position_for_drone == self.CURRENT:
                pose.position.x = current_drone_position_pose.position.x
                pose.position.y = current_drone_position_pose.position.y
                # print("alt", alt)
                pose.position.z = alt
                """if alt < current_drone_position_pose.position.z:
                    pose.position.z = alt
                else:
                    pose.position.z = current_drone_position_pose.position.z"""
                r_ground_dist = self.recommended_ground_dist_alt(distance, alt)
            # NEXT
            else:
                target_drone_vector = self.vector_target_drone(current_drone_position_pose, next_target_position_xy)
                r_ground_dist = self.recommended_ground_dist_alt(r_dist, alt)
                next_position = target_drone_vector * r_ground_dist
                pose.position.x = next_position[0] + next_target_position_xy[0]
                pose.position.y = next_position[1] + next_target_position_xy[1]
                pose.position.z = alt
            pose.orientation.x, pose.orientation.y = self.recommended_camera_orientation(r_ground_dist, alt,
                                                                                         camera_yaw,
                                                                                         camera_pitch,
                                                                                         target_information)
            pose.orientation.z = self.recommended_yaw(pose, next_target_position_xy)
        else:
            # TARGET
            position_map = map.find_nearest_free_position(map_next_target_position.x, map_next_target_position.y)
            position = map.map_point_from_map_coordinates(position_map[0], position_map[1])
            pose.position.x = position.real_x
            pose.position.y = position.real_y
            pose.position.z = 3
            shift_xy = next_target_position_xy - current_target_position_xy
            pose.orientation.z = utils.Math.calculate_yaw_from_points(0, 0, shift_xy[0], shift_xy[1])
            pose.orientation.x, pose.orientation.y = 0, 0.2

        print("Distance:", r_dist, ", altitude:", alt)
        print("target drone vector", target_drone_vector)
        return pose, target_drone_vector

    def check_safety(self, pose, next_target_position_xy, map):
        """
        SAFETY CHECK
        """
        safety_check = False
        map_point = map.map_point_from_real_coordinates(pose.position.x, pose.position.y, pose.position.z)
        if not map.is_free_for_drone(map_point):
            safety_check = True
            pose.position.x = next_target_position_xy[0]
            pose.position.y = next_target_position_xy[1]
        return pose, safety_check

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        try:
            if "times" in kwargs and kwargs["times"][0] == target_positions[0].header.stamp.to_sec() and \
                    kwargs["times"][1] == target_positions[1].header.stamp.to_sec() and \
                    kwargs["times"][2] == drone_positions[0].header.stamp.to_sec():
                rospy.loginfo(
                    "It was returned last predicted position for times {}.".format(kwargs["times"]))
                return kwargs["last_predicted_position"]
            else:
                rospy.loginfo(
                    "Calculation a new predicted position for times {}.".format(
                        [target_positions[0].header.stamp.to_sec(),
                         target_positions[1].header.stamp.to_sec(),
                         drone_positions[0].header.stamp.to_sec()]))
            time_difference, next_target_position_xy, current_target_position_xy, \
            current_drone_position_pose, camera_yaw, camera_pitch, target_information, \
            minimum_altitude, last_predicted_pose, last_seen, see, last_type_prediction, map_next_target_position, \
            last_current_position_distance, preparation = self.init_transformation(target_positions, drone_positions,
                                                                                   map, kwargs)
            r_dist, distance, alt = self.data_for_calculation(current_target_position_xy, next_target_position_xy,
                                                              time_difference, current_drone_position_pose,
                                                              minimum_altitude)
            if last_predicted_pose is not None:
                last_r_dist, last_distance, last_alt = self.data_for_calculation(current_target_position_xy,
                                                                                 next_target_position_xy,
                                                                                 time_difference,
                                                                                 last_predicted_pose, minimum_altitude)
            else:
                last_r_dist, last_distance, last_alt = r_dist, distance, alt

            next_position_for_drone, true_prediction = self.compute_parameters(r_dist, distance, target_information,
                                                                               see, last_type_prediction, last_r_dist,
                                                                               last_distance,
                                                                               last_current_position_distance,
                                                                               preparation)
            pose, target_drone_vector = self.compute_next_position(next_position_for_drone, last_predicted_pose,
                                                                   current_drone_position_pose, alt, distance,
                                                                   next_target_position_xy, r_dist, camera_yaw,
                                                                   camera_pitch, target_information, map,
                                                                   current_target_position_xy, map_next_target_position,
                                                                   true_prediction)
            pose, safety_check = self.check_safety(pose, next_target_position_xy, map)
            if safety_check:
                self._prediction_type = self.TARGET
            else:
                if true_prediction == self.LAST:
                    self._prediction_type = self.TARGET
                else:
                    self._prediction_type = true_prediction
            """
            LOGGING
            """
            print("Drone position: {}, {}, {}, {}, {}, {}", current_drone_position_pose.position.x,
                  current_drone_position_pose.position.y, current_drone_position_pose.position.z,
                  current_drone_position_pose.orientation.x, current_drone_position_pose.orientation.y,
                  current_drone_position_pose.orientation.z)
            rospy.loginfo("Next target position: {}, {}".format(next_target_position_xy[0], next_target_position_xy[1]))
            rospy.loginfo(
                "Next drone position: {}, {}, {}, {}, {}, {}".format(pose.position.x, pose.position.y, pose.position.z,
                                                                     pose.orientation.x, pose.orientation.y,
                                                                     pose.orientation.z))
            rospy.loginfo(
                "Type of prediction {}, {}, {}".format(["CURRENT", "LAST", "NEXT", "TARGET"][next_position_for_drone],
                                                       preparation,
                                                       ["CURRENT", "LAST", "NEXT", "TARGET"][last_type_prediction]))
            if last_predicted_pose is not None:
                rospy.loginfo(
                    "Last predicted drone position: {}, {}, {}, {}, {}, {}".format(last_predicted_pose.position.x,
                                                                                   last_predicted_pose.position.y,
                                                                                   last_predicted_pose.position.z,
                                                                                   last_predicted_pose.orientation.x,
                                                                                   last_predicted_pose.orientation.y,
                                                                                   last_predicted_pose.orientation.z))

            # drone: x (1), y (2), z(3), camera: horizontal(4), vertical(5), drone yaw(6), target quotioent (7), target: x (8), y (9), z (10), roll (11), pitch (12), yaw (13)
            # predicted new position: x (14), y (15), z (16), camera: horizontal (17), vertical (18), drone yaw (19), time (20)
            self.f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
                current_drone_position_pose.position.x, current_drone_position_pose.position.y,
                current_drone_position_pose.position.z,
                current_drone_position_pose.orientation.x, current_drone_position_pose.orientation.y,
                current_drone_position_pose.orientation.z,
                int(target_information.quotient > 0.7), next_target_position_xy[0], next_target_position_xy[1],
                next_target_position_xy[2],
                next_target_position_xy[3], next_target_position_xy[4], next_target_position_xy[5],
                pose.position.x, pose.position.y, pose.position.z, pose.orientation.x,
                pose.orientation.y, pose.orientation.z, rospy.Time.now().to_sec()))
            array = np.array([[pose.position.x, pose.position.y, pose.position.z, pose.orientation.x,
                               pose.orientation.y, pose.orientation.z]])
            return array
        except Exception as e:
            rospy.logwarn(str(e))
            if "last_predicted_position" in kwargs:
                return kwargs["last_predicted_position"]
            else:
                return utils.DataStructures.pointcloud2_to_array(drone_positions[0])

    def state_variables(self, data, drone_positions, target_positions, map, **kwargs):
        kw = {}
        kw["times"] = [target_positions[0].header.stamp.to_sec(), target_positions[1].header.stamp.to_sec(),
                       drone_positions[0].header.stamp.to_sec()]
        kw["last_predicted_position"] = data
        kw["last_type"] = self._prediction_type
        return kw

    def restart_algorithm(self):
        self._prediction_type = self.CURRENT
