#!/usr/bin/env python
import rospy
import tf as tf_ros
import numpy as np
from geometry_msgs.msg import Pose
import helper_pkg.utils as utils
from helper_pkg.utils import Constants
import helper_pkg.PredictionManagement as pm
import time


class PredictionWithParticles(pm.PredictionAlgorithm):
    DRONE_POSITIONS = 0
    TARGET_POSITIONS = 3
    INPUT_TYPE = Constants.InputOutputParameterType.pointcloud
    OUTPUT_TYPE = Constants.InputOutputParameterType.pointcloud
    MAP_PRESENCE = Constants.MapPresenceParameter.yes

    def __init__(self):
        super(PredictionWithParticles, self).__init__()
        _env_configuration = rospy.get_param("environment_configuration")
        target_configuration = rospy.get_param("target_configuration")
        drone_configuration = rospy.get_param("drone_configuration")
        self._target_size = target_configuration["properties"]["target_size"]
        self._drone_speed = drone_configuration["control"]["video_max_horizontal_speed"]
        self._max_target_speed = target_configuration["strategies"]["max_velocity"]
        self._prediction_algorithm_name = "Prediction particles"
        self.cov = 2
        self.estimations = 10

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        samples_count = self.get_samples_count(drone_positions, target_positions, **kwargs)
        observation_map = self.observation_map(**kwargs)
        t_position = utils.DataStructures.pointcloud2_to_array(target_positions[2])
        t_1_position = utils.DataStructures.pointcloud2_to_array(target_positions[1])
        t_2_position = utils.DataStructures.pointcloud2_to_array(target_positions[0])
        x_index = Constants.PointCloudNames.X
        y_index = Constants.PointCloudNames.Y
        pred_t = (t_position[:, Constants.PointCloudNames.PREDECESSOR]).astype(np.int32)
        pred_t_1 = (t_1_position[pred_t, Constants.PointCloudNames.PREDECESSOR]).astype(np.int32)
        older_speeds = np.array([t_1_position[pred_t, x_index] - t_2_position[pred_t_1, x_index],
                                 t_1_position[pred_t, y_index] - t_2_position[pred_t_1, y_index]]).T
        newer_speeds = np.array([t_position[:, x_index] - t_1_position[pred_t, x_index],
                                 t_position[:, y_index] - t_1_position[pred_t, y_index]]).T
        accelerations = (newer_speeds - older_speeds)
        increment = np.clip((newer_speeds + accelerations),
                            (-self._max_target_speed, -self._max_target_speed),
                            (self._max_target_speed, self._max_target_speed))
        #print("t", t_position)
        #print("t1", t_1_position)
        #print("t2", t_2_position)
        #print("os", older_speeds)
        #print("ns", newer_speeds)
        #print("ac", accelerations)
        positions = np.array([t_position[:, x_index], t_position[:, y_index]]).T + increment
        cov_matrix = [[self.cov, 0], [0, self.cov]]
        new_particles = np.zeros((len(positions) * self.estimations, 7))
        probs_list = np.zeros((len(positions) * self.estimations))
        for i in range(len(positions)):
            new_particles[i * self.estimations:(i + 1) * self.estimations, :2] = np.random.multivariate_normal(
                positions[i], cov_matrix, self.estimations)
            yaw = utils.Math.calculate_yaw_from_points(0, 0, (newer_speeds[i] + accelerations[i])[0],
                                                       (newer_speeds[i] + accelerations[i])[1])
            new_particles[i * self.estimations:(i + 1) * self.estimations, 5:7] = [yaw, i]
            map_x, map_y = map.get_index_on_map(new_particles[i * self.estimations:(i + 1) * self.estimations, 0],
                                                new_particles[i * self.estimations:(i + 1) * self.estimations, 1])
            if observation_map is not None:
                probs_list[i * self.estimations:(i + 1) * self.estimations] = observation_map[map_x, map_y]
            else:
                probs_list[i * self.estimations:(i + 1) * self.estimations] = map.target_obstacle_map[map_x, map_y]
        s = np.sum(probs_list)
        if s > 0:
            probs_list /= s
        else:
            probs_list = None
        new_particles_indices = np.random.choice(np.arange(0, len(new_particles)), samples_count, p=probs_list)
        new_particles = np.array(new_particles)[new_particles_indices]
        #print("np", new_particles)
        return new_particles

    def state_variables(self, drone_positions, target_positions, map, **kwargs):
        return {}
