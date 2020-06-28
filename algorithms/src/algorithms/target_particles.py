#!/usr/bin/env python
import rospy
import os
import tf as tf_ros
import numpy as np
from geometry_msgs.msg import Pose
import helper_pkg.utils as utils
from helper_pkg.utils import Constants
import helper_pkg.PredictionManagement as pm
import time
from scipy.stats import multivariate_normal, describe
import matplotlib.pyplot as plt
from scipy import stats


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
        self.cov = self._max_target_speed / 2.0
        self.estimations = 10
        self._min_location_jittering = 2
        self._min_shift_jittering = 2
        self._slack = self.estimations - self._min_location_jittering - self._min_shift_jittering
        self._default_ratio = 0.995
        self._seen_ratio = 0.3
        file_name = os.path.split(self._obstacles_file_path)[1][7:]
        self.f = open("searching_test/target_prediction_log_file_{}.txt".format(file_name), "w")

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        start = time.time()
        base_start = start
        samples_count = self.get_samples_count(drone_positions, target_positions, **kwargs)
        observation_map = self.observation_map(**kwargs)
        ratio = self._default_ratio
        if "seen" in kwargs:
            if kwargs["seen"]:
                ratio = self._seen_ratio
        print("Time 0 {}".format(time.time() - start))
        start = time.time()
        increment, clipped_newer_speeds, maximal_newer_speeds, newer_max_target_speed, \
        newer_speeds, accelerations, t_position, weight_index, positions = self.calculate_speeds_accelerations_increments(
            target_positions)
        print("Time 1 {}".format(time.time() - start))
        start = time.time()
        probs_list, new_particles = self.calculate_new_particles_with_probs(increment, clipped_newer_speeds,
                                                                            maximal_newer_speeds,
                                                                            newer_max_target_speed,
                                                                            newer_speeds, accelerations,
                                                                            observation_map, t_position, weight_index,
                                                                            map, positions)
        print("Time 2 {}".format(time.time() - start))
        start = time.time()
        chosen_particles = self.choose_from_new_particles_result_set(probs_list, observation_map, samples_count, ratio,
                                                                     new_particles, positions, t_position, weight_index,
                                                                     map)
        print("Time 3 {}".format(time.time() - start))
        """
        pose = utils.DataStructures.array_to_pose(chosen_particles)
        print("============================")
        print(pose)
        print("----------------------------")        
        print(np.median(observation_map[observation_map>0]), np.max(observation_map[observation_map>0]), 
        np.min(observation_map[observation_map>0]))
        """
        rospy.loginfo("Target prediction proceeded:  {}".format(time.time() - base_start))
        self.log(ratio)
        return chosen_particles

    def log(self, ratio):
        ar = []
        ar.append(rospy.Time.now().to_sec())
        if ratio == self._default_ratio:
            ar.append(0)
        else:
            ar.append(1)
        self.f.write("{}\n".format(self.convert_array_to_string(ar, ",")))

    def convert_array_to_string(self, array, delimiter=";"):
        str_array = [str(a) for a in array]
        s = delimiter.join(str_array)
        return s

    def choose_from_new_particles_result_set(self, probs_list, observation_map, samples_count, ratio, new_particles,
                                             positions, t_position, weight_index, map):
        s = np.sum(probs_list)
        nonnegative = np.count_nonzero(probs_list >= 0) == len(probs_list)
        if s > 0 and nonnegative:
            probs_list /= s
        else:
            probs_list = None
        if observation_map is None:
            count = samples_count
        else:
            count = int(np.ceil(samples_count * ratio))
        new_particles_indices = np.random.choice(np.arange(0, len(new_particles)), count, p=probs_list)
        chosen_particles = np.empty((samples_count, len(utils.DataStructures.point_cloud_name())))
        chosen_particles[:count] = np.array(new_particles)[new_particles_indices]
        if observation_map is not None:
            utils.Plotting.plot_probability(observation_map)
            new_chosen_xy = np.random.choice(len(observation_map) * len(observation_map[0]),
                                             samples_count - count, p=observation_map.reshape(-1))
            # print(observation_map.reshape(-1)[new_chosen_xy], new_chosen_xy)
            new_chosen_x, new_chosen_y = map.get_coordination_from_map_indices_vectors(
                new_chosen_xy % len(observation_map[0]), new_chosen_xy / len(observation_map[0]))
            # print("zip", zip(new_chosen_x, new_chosen_y))
            last_particle = np.random.randint(0, len(positions), samples_count - count)
            yaw = utils.Math.calculate_yaw_from_points_vectors(0, 0, new_chosen_x - t_position[last_particle, 0],
                                                               new_chosen_y - t_position[last_particle, 1])
            chosen_particles[count:, 0] = new_chosen_x
            chosen_particles[count:, 1] = new_chosen_y
            chosen_particles[count:, 5] = yaw
            chosen_particles[count:, 6] = last_particle
            chosen_particles[count:, weight_index] = observation_map.reshape(-1)[new_chosen_xy]
        """
        #map_x, map_y = map.get_index_on_map(chosen_particles[:, 0], chosen_particles[:, 1])
        #p = observation_map.reshape(-1)
        self.f.write("{},{}:".format(rospy.Time.now().to_sec(), ratio))
        for i in range(len(map_x)):
            self.f.write(str(chosen_particles[i, 0]) + "," + str(chosen_particles[i, 1]))
            if i + 1 < len(map_x):
                self.f.write(",")
            if i == count:
                print("----------------------------------")
            # print("p: {}. {}. {}".format(chosen_particles[i, 0], chosen_particles[i, 1],
            #                             observation_map[map_y[i], map_x[i]]))
        self.f.write("\n")"""
        # print("np", chosen_particles)
        # utils.Plotting.plot_points(chosen_particles, map, "Particles", False)
        # print("average {}, maximal {}, obs max {}".format(np.average(observation_map[map_y[:], map_x[:]]), np.max(observation_map[map_y[:], map_x[:]]), np.max(observation_map)))
        # print("hist {}".format(np.histogram(observation_map.reshape(-1), 2)))
        return chosen_particles

    def calculate_speeds_accelerations_increments(self, target_positions):
        t_position = utils.DataStructures.pointcloud2_to_array(target_positions[2])
        t_1_position = utils.DataStructures.pointcloud2_to_array(target_positions[1])
        t_2_position = utils.DataStructures.pointcloud2_to_array(target_positions[0])
        x_index = Constants.PointCloudNames.X
        y_index = Constants.PointCloudNames.Y
        weight_index = Constants.PointCloudNames.WEIGHT
        pred_t = (t_position[:, Constants.PointCloudNames.PREDECESSOR]).astype(np.int32)
        pred_t_1 = (t_1_position[pred_t, Constants.PointCloudNames.PREDECESSOR]).astype(np.int32)
        older_speeds = np.array([t_1_position[pred_t, x_index] - t_2_position[pred_t_1, x_index],
                                 t_1_position[pred_t, y_index] - t_2_position[pred_t_1, y_index]]).T
        newer_speeds = np.array([t_position[:, x_index] - t_1_position[pred_t, x_index],
                                 t_position[:, y_index] - t_1_position[pred_t, y_index]]).T
        normalized_older_speeds = np.abs(utils.Math.normalize_vectorized(older_speeds))
        normalized_newer_speeds = np.abs(utils.Math.normalize_vectorized(newer_speeds))
        newer_time_difference = np.abs(
            target_positions[2].header.stamp.to_sec() - target_positions[1].header.stamp.to_sec())
        older_time_difference = np.abs(
            target_positions[0].header.stamp.to_sec() - target_positions[1].header.stamp.to_sec())
        older_max_target_speed = self._max_target_speed * older_time_difference
        newer_max_target_speed = self._max_target_speed * newer_time_difference
        maximal_older_speeds = normalized_older_speeds * older_max_target_speed
        minimal_older_speeds = normalized_older_speeds * -older_max_target_speed
        maximal_newer_speeds = normalized_newer_speeds * newer_max_target_speed
        minimal_newer_speeds = normalized_newer_speeds * -newer_max_target_speed
        """
        print("maximal speed newer {}, older {}. Time difference newer {}, older {}".format(maximal_newer_speeds,
                                                                                            maximal_older_speeds,
                                                                                            newer_time_difference,
                                                                                            older_time_difference))
                                                                                            """
        clipped_older_speeds = np.clip(older_speeds, minimal_older_speeds, maximal_older_speeds)
        clipped_newer_speeds = np.clip(newer_speeds, minimal_newer_speeds, maximal_newer_speeds)
        accelerations = (clipped_newer_speeds - clipped_older_speeds)
        increment = clipped_newer_speeds + accelerations
        normalized_increment = np.abs(utils.Math.normalize_vectorized(increment))
        maximal_increment = normalized_increment * newer_max_target_speed
        minimal_increment = normalized_increment * -newer_max_target_speed
        increment = np.clip(increment, minimal_increment, maximal_increment)
        positions = np.array([t_position[:, x_index], t_position[:, y_index]]).T + increment
        """
        print("t", t_position)
        print("t1", t_1_position)
        print("t2", t_2_position)
        print("os", clipped_older_speeds)
        print("ns", clipped_newer_speeds)
        print("ac", accelerations)
        """
        return increment, clipped_newer_speeds, maximal_newer_speeds, newer_max_target_speed, newer_speeds, accelerations, t_position, weight_index, positions

    def calculate_new_particles_with_probs(self, increment, clipped_newer_speeds, maximal_newer_speeds,
                                           newer_max_target_speed, newer_speeds, accelerations, observation_map,
                                           t_position, weight_index, map, positions):
        particles_probability = t_position[:, weight_index]
        particles_probability /= np.sum(particles_probability)
        chosen_indices = np.random.choice(range(len(positions)), len(positions), p=particles_probability)
        new_particles = np.zeros((len(positions) * self.estimations, len(utils.DataStructures.point_cloud_name())))
        abs_increment = np.abs(increment)
        halfed_vallues = abs_increment / 2.0
        bins_count = 5  # there is 5 * 5 bins in both dimensions
        ret = stats.binned_statistic_2d(halfed_vallues[:, 0], halfed_vallues[:, 1], halfed_vallues[:, 0], 'count',
                                        bins=bins_count, expand_binnumbers=True)
        # speed for indices / maximal speed
        speed_ratio = np.nan_to_num(
            np.hypot(clipped_newer_speeds[chosen_indices, 0], clipped_newer_speeds[chosen_indices, 1]) / np.hypot(
                maximal_newer_speeds[chosen_indices, 0], maximal_newer_speeds[chosen_indices, 1]))
        speed_hypo = np.hypot(clipped_newer_speeds[chosen_indices, 0], clipped_newer_speeds[chosen_indices, 1]) * 1 / 3
        shift_jittering = (np.abs(speed_ratio).astype(np.int32) * self._slack) + self._min_shift_jittering
        location_shifts = utils.Math.uniform_circle_vectorized(speed_hypo, self.estimations)
        velocity_shifts = np.repeat(increment[chosen_indices], self.estimations, axis=0).reshape(-1, self.estimations,
                                                                                                 2)
        for i in range(bins_count):
            for j in range(bins_count):
                count = int(ret.statistic[i, j])
                if count > 0:
                    cov_matrix = [[ret.x_edge[i], 0],
                                  [0, ret.y_edge[j]]]
                    mask = np.logical_and(ret.binnumber[0] == (i + 1), ret.binnumber[1] == (j + 1))
                    add_factor = np.random.multivariate_normal([0, 0], cov_matrix, self.estimations * count).reshape(
                        (count, self.estimations, 2))
                    velocity_shifts[mask] += add_factor
        shifts = np.zeros((len(chosen_indices), self.estimations, 2))
        r = np.arange(shifts.shape[1])
        mask = shift_jittering[:, None] > r
        shifts[mask] = location_shifts[mask]
        shifts[~mask] = velocity_shifts[~mask]
        normalized_shifts = np.abs(utils.Math.normalize_generalized(shifts))
        maximal_shifts = normalized_shifts * newer_max_target_speed
        minimal_shifts = normalized_shifts * -newer_max_target_speed
        shifts = np.clip(shifts, minimal_shifts, maximal_shifts)
        repeated_positions = np.repeat(positions[chosen_indices], self.estimations, axis=0).reshape(-1,
                                                                                                    self.estimations,
                                                                                                    2)
        clipped_shifted_positions = np.clip(repeated_positions + shifts,
                                            (-map.real_height / 2.0, -map.real_width / 2.0),
                                            (map.real_height / 2.0, map.real_width / 2.0))
        new_particles[:, :2] = clipped_shifted_positions.reshape((-1, 2))
        yaw = utils.Math.calculate_yaw_from_points_vectors(0, 0, (newer_speeds[chosen_indices] + accelerations[
            chosen_indices])[:, 0], (newer_speeds[chosen_indices] + accelerations[chosen_indices])[:, 1])
        new_particles[:, 5] = np.repeat(yaw, 10)
        new_particles[:, 6] = np.repeat(chosen_indices, 10)
        map_x, map_y = map.get_index_on_map(new_particles[:, 0], new_particles[:, 1])
        if observation_map is not None:
            probs_list = observation_map[map_y, map_x]
            new_particles[:, weight_index] = np.max(np.array(
                [np.repeat(t_position[chosen_indices, weight_index], self.estimations),
                 observation_map[map_y, map_x]]), axis=0)
        else:
            probs_list = map.target_obstacle_map[map_y, map_x]
            new_particles[:, weight_index] = np.max(np.array(
                [np.repeat(t_position[chosen_indices, weight_index], self.estimations),
                 map.target_obstacle_map[map_y, map_x]]), axis=0)
        return probs_list, new_particles

    def older_version_of_calculation(self, chosen_indices, increment, clipped_newer_speeds,
                                     maximal_newer_speeds, newer_max_target_speed, new_particles,
                                     newer_speeds, accelerations, observation_map, probs_list,
                                     t_position, weight_index, map):
        for i in chosen_indices:
            start = time.time()
            cov = np.clip(np.abs(increment[i]) / 2.0, 0.0001, None)
            cov_matrix = [[cov[0], 0],
                          [0, cov[1]]]
            speed_ratio = np.nan_to_num(np.hypot(clipped_newer_speeds[i, 0], clipped_newer_speeds[i, 1]) / np.hypot(
                maximal_newer_speeds[i, 0], maximal_newer_speeds[i, 1]))
            speed_hypo = np.hypot(clipped_newer_speeds[i, 0], clipped_newer_speeds[i, 1]) * 1 / 3
            shift_jittering = int(
                (np.abs(speed_ratio)) * self._slack) + self._min_shift_jittering
            location_jittering = self.estimations - shift_jittering
            location_shifts = utils.Math.uniform_circle(speed_hypo, location_jittering)
            a.append(time.time() - start)
            start = time.time()
            velocity_shifts = np.random.multivariate_normal(increment[i], cov_matrix, shift_jittering)
            shifts = np.zeros((self.estimations, 2))
            shifts[:location_jittering] = location_shifts
            shifts[location_jittering:] = velocity_shifts
            normalized_shifts = np.abs(utils.Math.normalize_vectorized(shifts))
            maximal_shifts = normalized_shifts * newer_max_target_speed
            minimal_shifts = normalized_shifts * -newer_max_target_speed
            shifts = np.clip(shifts, minimal_shifts, maximal_shifts)
            new_particles[i * self.estimations:(i + 1) * self.estimations, :2] = np.clip(positions[i] + shifts, (
                -map.real_height / 2.0, -map.real_width / 2.0), (map.real_height / 2.0, map.real_width / 2.0))
            yaw = utils.Math.calculate_yaw_from_points(0, 0, (newer_speeds[i] + accelerations[i])[0],
                                                       (newer_speeds[i] + accelerations[i])[1])
            new_particles[i * self.estimations:(i + 1) * self.estimations, 5:7] = [yaw, i]
            map_x, map_y = map.get_index_on_map(new_particles[i * self.estimations:(i + 1) * self.estimations, 0],
                                                new_particles[i * self.estimations:(i + 1) * self.estimations, 1])
            b.append(time.time() - start)
            start = time.time()
            if observation_map is not None:
                probs_list[i * self.estimations:(i + 1) * self.estimations] = observation_map[map_y, map_x]
                new_particles[i * self.estimations:(i + 1) * self.estimations, weight_index] = \
                    np.max(np.array([np.repeat(t_position[i, weight_index], len(observation_map[map_y, map_x])),
                                     observation_map[map_y, map_x]]), axis=0)
            else:
                probs_list[i * self.estimations:(i + 1) * self.estimations] = map.target_obstacle_map[map_y, map_x]
                new_particles[i * self.estimations:(i + 1) * self.estimations, weight_index] = \
                    np.max(np.array(
                        [np.repeat(t_position[i, weight_index], len(map_x)), map.target_obstacle_map[map_y, map_x]]),
                        axis=0)
            c.append(time.time() - start)
            start = time.time()

    def state_variables(self, data, drone_positions, target_positions, map, **kwargs):
        return {}
