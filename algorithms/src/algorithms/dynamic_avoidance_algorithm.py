#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Pose, Point
import helper_pkg.utils as utils
import helper_pkg.PredictionManagement as pm
from helper_pkg.utils import Constants



class AvoidanceAlgorithm(pm.PredictionAlgorithm):
    DRONE_POSITIONS = 2
    TARGET_POSITIONS = 0
    INPUT_TYPE = Constants.InputOutputParameterType.pointcloud
    OUTPUT_TYPE = Constants.InputOutputParameterType.pointcloud
    MAP_PRESENCE = Constants.MapPresenceParameter.yes

    def __init__(self):
        super(AvoidanceAlgorithm, self).__init__()
        logic_configuration = rospy.get_param("logic_configuration")
        drone_configuration = rospy.get_param("drone_configuration")
        self._prediction_algorithm_name = "Dynamic avoidance algorithm"
        self._low_bound = logic_configuration["avoiding"]["low_bound"]
        self._minimum_distance = logic_configuration["avoiding"]["minimum_distance"]
        self._max_height = logic_configuration["conditions"]["max_height"]
        maximal_speed = drone_configuration["control"]["video_max_horizontal_speed"]
        self._shift_constant = 1
        self._smaller_shift_constant = 0.1
        self._drone_size = drone_configuration["properties"]["size"]
        self._sonar_change_limit = 0.3
        self._collision_change_limit = 0.1
        self._timeout = 50
        self.f = open("avoidance_test/log_file_{}_{}.txt".format("colour", "big"), "a")
        self.f.write("------------------------------\n")

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        pose = Pose()
        drone_position = drone_positions[0]
        drone_position = utils.DataStructures.pointcloud2_to_array(drone_position)
        collision_probability_list = kwargs["collision_data"]
        collision_probability = collision_probability_list[0]
        last_collision_probability = collision_probability_list[1]
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
        if "state" in kwargs:
            state = kwargs["state"]
        if state == 1:
            state = 1  # obstacle in front of drone
            self.f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(state, rospy.Time.now().to_sec(),
                                                                   drone_position[0, 0],
                                                                   drone_position[0, 1],
                                                                   drone_position[0, 2],
                                                                   collision_probability,
                                                                   collision_change, sonar_change))
        elif state == 2:
            state = 2  # drone in front of and above obstacle
            self.f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(state, rospy.Time.now().to_sec(),
                                                                   drone_position[0, 0],
                                                                   drone_position[0, 1],
                                                                   drone_position[0, 2],
                                                                   collision_probability,
                                                                   collision_change, sonar_change))
        elif state == 3:
            state = 3  # drone is above obstacle
            self.f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(state, rospy.Time.now().to_sec(),
                                                                   drone_position[0, 0],
                                                                   drone_position[0, 1],
                                                                   drone_position[0, 2],
                                                                   collision_probability,
                                                                   collision_change, sonar_change))
        elif state == 0 or state == 5 or state == 4:
            state = 0  # free move
            self.f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(state, rospy.Time.now().to_sec(),
                                                                   drone_position[0, 0],
                                                                   drone_position[0, 1],
                                                                   drone_position[0, 2],
                                                                   collision_probability,
                                                                   collision_change, sonar_change))
        if state == 1:
            if drone_position[0, 2] + self._shift_constant < self._max_height:
                shift_z = self._shift_constant  # lift drone
            else:
                shift_z = 0  # maximal altitude reached
        elif state == 2 or state == 3:
            shift_z = 0  # height is ok
        elif state == 0:
            shift_z -= (drone_position[0, 2] + 1)  # no height requiremnets
        if self._minimum_distance > sensor_data:
            shift_z = self._smaller_shift_constant  # drone is too low above obstacle
        # print(drone_position)
        # print("shift_z", shift_z, "sensor_data", sensor_data)
        # print("collision_probability", collision_probability)
        # print("state", state)
        pose.position.z = drone_position[0, 2] + shift_z
        pose.position.x = drone_position[0, 0]
        pose.position.y = drone_position[0, 1]
        pose.orientation.z = drone_position[0, 5]
        if "stable_point" in kwargs and kwargs["stable_point"] is not None:
            # print("Stable point: {}".format(kwargs["stable_point"]))
            pose.position.x = kwargs["stable_point"].x
            pose.position.y = kwargs["stable_point"].y
            pose.orientation.z = kwargs["stable_point"].z
        pose.orientation.x = 0
        pose.orientation.y = 0
        # print("pose: {}".format(pose))
        ar = np.array([[pose.position.x, pose.position.y, pose.position.z, 0, 0, pose.orientation.z, 0]])
        return ar

    def state_variables(self, data, drone_positions, target_positions, map, **kwargs):
        drone_position = utils.DataStructures.pointcloud2_to_array(drone_positions[0])
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
        state = 0
        if "state" in kwargs:
            state = kwargs["state"]
        if "stable_point" in kwargs:
            stable_point = kwargs["stable_point"]
        recommended = None
        if "recommended" in kwargs:
            recommended = kwargs["recommended"]
        # print("kwargs", kwargs)
        # to 1
        if collision_probability > self._low_bound and state != 1 and (state == 2 or state == 0):
            print("To 1")
            if stable_point is None or (abs(stable_point.x - drone_position[0, 0]) > 0.2 or abs(
                    stable_point.y - drone_position[0, 1]) > 0.2):
                stable_point = Point()
                stable_point.x = drone_position[0, 0]
                stable_point.y = drone_position[0, 1]
                stable_point.z = drone_position[0, 5]
            sonar_change = 0
            collision_change_time = 0
            recommended = drone_position[0, 2]
            state = 1
        # to 2
        if collision_change_time == 0 and sonar_change == 0 and \
                older_collision_probability > self._low_bound > collision_probability and abs(
            older_collision_probability - collision_probability) > self._collision_change_limit and state != 2 and state == 1:
            print("To 2")
            sonar_change = 0
            collision_change_time = rospy.Time.now().to_sec()
            state = 2
        # to 3
        elif collision_change_time > 0 and sonar_change == 0 and abs(
                older_sensor_data - sensor_data) > self._sonar_change_limit and older_sensor_data - sensor_data > 0 and state != 3 and state == 2:
            print("To 3")
            sonar_change = older_sensor_data - sensor_data
            # print("sonar data {}, {}".format(sensor_data, older_sensor_data))
            collision_change_time = 0
            state = 3
            if stable_point is None or (abs(stable_point.x - drone_position[0, 0]) > 0.2 or abs(
                    stable_point.y - drone_position[0, 1]) > 0.2):
                stable_point = Point()
                stable_point.x = drone_position[0, 0]
                stable_point.y = drone_position[0, 1]
                stable_point.z = drone_position[0, 5]
        # to 4
        elif collision_change_time == 0 and sonar_change >= 0 and abs(
                older_sensor_data - sensor_data) > self._sonar_change_limit and older_sensor_data - sensor_data < 0 and state != 0 and state != 2 and state == 3:
            if stable_point is not None and np.sqrt(
                    np.square(stable_point.x - drone_position[0, 0]) + np.square(
                        stable_point.y - drone_position[0, 1])) < 0.5:
                sonar_change = 0
                collision_change_time = rospy.Time.now().to_sec()
                state = 2
            else:
                print("To 4")
                stable_point = None
                sonar_change = older_sensor_data - sensor_data
                collision_change_time = 0
                recommended = None
                state = 0
        # to "5"
        elif collision_change_time > 0 and sonar_change == 0 and abs(
                collision_change_time - rospy.Time.now().to_sec()) > self._timeout and state != 5 and state == 2:
            print("To 5")
            stable_point = None
            collision_change_time = 0
            sonar_change = 0
            recommended = None
            state = 0
        return {"sonar_change": sonar_change, "collision_change": collision_change_time, "stable_point": stable_point,
                "recommended": recommended, "state": state}