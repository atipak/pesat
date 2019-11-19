#!/usr/bin/env python
import rospy
import tf as tf_ros
import numpy as np
from geometry_msgs.msg import Pose
import helper_pkg.utils as utils
from helper_pkg.utils import Constants
import helper_pkg.PredictionManagement as pm


class PredictionNaive(pm.PredictionAlgorithm):
    DRONE_POSITIONS = 0
    TARGET_POSITIONS = 3
    INPUT_TYPE = Constants.InputOutputParameterType.pose
    OUTPUT_TYPE = Constants.InputOutputParameterType.pose
    MAP_PRESENCE = Constants.MapPresenceParameter.yes

    def __init__(self):
        super(PredictionNaive, self).__init__()
        _env_configuration = rospy.get_param("environment_configuration")
        target_configuration = rospy.get_param("target_configuration")
        self._map_frame = _env_configuration["map"]["frame"]
        self._target_size = target_configuration["properties"]["target_size"]
        self._prediction_algorithm_name = "Prediction naive"
        self._max_distance_limit_for_random_position = 5.0

    def calculate_speed_and_direction(self, target_positions):
        older_poses_diff = np.array([target_positions[1].pose.position.x - target_positions[0].pose.position.x,
                                     target_positions[1].pose.position.y - target_positions[0].pose.position.y])
        newer_poses_diff = np.array([target_positions[2].pose.position.x - target_positions[1].pose.position.x,
                                     target_positions[2].pose.position.y - target_positions[1].pose.position.y])
        direction = np.average(np.array([older_poses_diff, newer_poses_diff]), 0)
        speed = np.hypot(*direction)
        # print("dp", direction, speed)
        return speed, direction

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        pose = Pose()
        original_pose = target_positions[2].pose
        speed, direction = self.calculate_speed_and_direction(target_positions)
        pose.position.x = original_pose.position.x + direction[0]
        pose.position.y = original_pose.position.y + direction[1]
        map_point = map.map_point_from_real_coordinates(pose.position.x, pose.position.y, pose.position.z)
        current_limit = 5.0
        while not map.is_free_for_target(map_point):
            if current_limit > self._max_distance_limit_for_random_position:
                return None
            free_map_point = map.random_free_place_for_target_in_neighbour(map_point, current_limit)
            # print(map.map)
            if free_map_point is not None:
                real_point = map.get_coordination_from_map_indices(free_map_point)
                pose.position.x = real_point[0]
                pose.position.y = real_point[1]
                break
            current_limit += 1
        yaw = utils.Math.calculate_yaw_from_points(original_pose.position.x, original_pose.position.y,
                                                   pose.position.x, pose.position.y)
        qt = tf_ros.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        pose.orientation.x = qt[0]
        pose.orientation.y = qt[1]
        pose.orientation.z = qt[2]
        pose.orientation.w = qt[3]
        return pose

    def state_variables(self, drone_positions, target_positions, map, **kwargs):
        return {}
