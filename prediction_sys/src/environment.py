#!/usr/bin/env python
import rospy
import os
import cv2
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Quaternion, PoseStamped
from gazebo_msgs.msg import ModelState
from pesat_msgs.msg import ImageTargetInfo
import helper_pkg.building_blocks as bb

class RenEnviroment(object):
    def __init__(self):
        rospy.init_node('vision', anonymous=False)
        self.target_info = None
        self.target_info_time = None
        rospy.Subscriber("/target/information", ImageTargetInfo, self.callback_target_information)
        self.target = None
        self.frame_history = 8
        self.last_state = None
        self.states = None
        self.sequential_number = 0
        self.initialized = False
        self.max_shift = 12
        self.map = None
        self.maps = []
        self.width = 640
        self.battery_const = -0.02  # two percents per meter
        self.overlimit_const = -0.15  # 15 percent per meter
        self.pub_set_model = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self.side_max_shift = self.max_shift / np.sqrt(2)

    def step(self, actions):
        if not self.initialized:
            return None, -10000, True, []
        # make next step with target = next_state
        next_target_pose = self.target.make_move()
        gazebo_state = self.to_gazebo_state(next_target_pose)
        self.pub_set_model.publish(gazebo_state)
        # make next step with drone and if drone is in a forbidden state, returns done true, else return false
        # x, y, z, ro, pi, psi
        next_drone_pose = self.to_pose(actions)
        gazebo_state = self.to_gazebo_state(next_drone_pose)
        self.pub_set_model.publish(gazebo_state)
        gazebo_set_time = rospy.Time.now()
        if self.map[int(next_drone_pose.pose.position.x), int(next_drone_pose.pose.position.y)] == 255:
            # reward is given by drone's position
            d = rospy.Duration(0.1, 0)
            rospy.sleep(d)
            while gazebo_set_time >= self.target_info_time:
                d = rospy.Duration(0.1, 0)
                rospy.sleep(d)
            self.states = [next_drone_pose, next_target_pose]
            next_state = self.create_new_state(next_drone_pose, next_target_pose)
            done = False
            reward = self.create_reward()
        else:
            self.states = None
            done = True
            reward = -1
            next_state = self.last_state
        # returns next_state, reward, done
        self.last_state = next_state
        return next_state, reward, done, []

    def reset(self):
        # generate init conditions
        # create map with building_blocks
        map = bb.generate_map()
        self.map = cv2.resize(map, (map.shape[0] * 2, map.shape[1] * 2))
        self.maps.append(self.map)
        # restart enviroment with new map
        self.restart_simulation()
        # create first frame_history actions from random start
        # place for target is randomly chosen
        target_x, target_y, drone_x, drone_y = self.find_random_place()
        # drone is in position where see target and its given with perfect information
        drone_positions = [[drone_x, drone_y]]
        target_positions = [[target_x, target_y]]
        map_id = len(self.maps) - 1
        for _ in range(self.frame_history - 1):
            next_target_x, next_target_y = self.next_target_position()
            target_shift_x = next_target_x - target_x
            target_shift_y = next_target_y - target_y
            if target_shift_x != 0 or target_shift_y != 0:
                sign_x = np.sign(target_shift_x)
                sign_y = np.sign(target_shift_y)
                next_drone_x, next_drone_y = self.free_place_in_field(
                    drone_x + sign_x * int(self.side_max_shift * np.abs(target_shift_x / self.max_shift)),
                    drone_y + sign_y * int(self.side_max_shift * np.abs(target_shift_y / self.max_shift)), 0,
                    self.side_max_shift)
                if next_drone_x is None:
                    next_drone_x, next_drone_y = self.free_place_in_field(drone_x, drone_y, 0, self.side_max_shift)
                if next_drone_x is None:
                    return self.reset()
                drone_x, drone_y, target_x, target_y = next_drone_x, next_drone_y, next_target_x, next_target_y
            drone_positions.append([drone_x, drone_y])
            target_positions.append([target_x, target_y])

        return [target_positions, drone_positions, map_id], False

    def next_target_position(self):
        # TODO choose next point to reach
        return 0, 0

    def transform_state(self, state):
        tranformed_state = np.zeros([self.width, self.width, 17], dtype=np.float32)
        drone = state[1]
        target = state[0]
        pmap_center = (self.width / 2, self.width / 2)
        map_center = target[len(target) - 1]
        min_x = map_center[0] - 320
        max_x = map_center[0] + 320
        if min_x < 0:
            min_x = 0
            map_center[0] = 320
        if max_x > self.map.shape[0]:
            max_x = self.map.shape[0]
            map_center[0] = self.map.shape[0] - 320
        min_y = self.map.shape[1] - 320
        max_y = self.map.shape[1] + 320
        if min_y < 0:
            min_y = 0
            map_center[1] = 320
        if max_y > self.map.shape[1]:
            max_y = self.map.shape[1]
            map_center[1] = self.map.shape[1] - 320
        index = 0
        for s in target:
            shift_x = map_center[0] - s[0]
            shift_y = map_center[1] - s[1]
            pmap_x = pmap_center[0] + shift_x
            pmap_y = pmap_center[1] + shift_y
            for i in [-1, 0, 1]:
                x_index = pmap_x + i
                if x_index < 0:
                    x_index = 0
                if x_index > self.width:
                    x_index = self.width
                for j in [-1, 0, 1]:
                    y_index = pmap_y + j
                    if y_index < 0:
                        y_index = 0
                    if y_index > self.width:
                        y_index = self.width
                    tranformed_state[x_index, y_index, index] = 1 / 9
            index += 1
        for s in drone:
            shift_x = map_center[0] - s[0]
            shift_y = map_center[1] - s[1]
            pmap_x = pmap_center[0] + shift_x
            pmap_y = pmap_center[1] + shift_y
            for i in [-1, 0, 1]:
                x_index = pmap_x + i
                if x_index < 0:
                    x_index = 0
                if x_index > self.width:
                    x_index = self.width
                for j in [-1, 0, 1]:
                    y_index = pmap_y + j
                    if y_index < 0:
                        y_index = 0
                    if y_index > self.width:
                        y_index = self.width
                    tranformed_state[x_index, y_index, index] = 1 / 9
            index += 1
        tranformed_state[:, :, 16] = self.map[min_x:max_x, min_y, max_y]
        # get state from step/another package and creates input for network
        # state [frame_history * position of target, frame_history * position of drone, id of map]
        # returns images 640 * 640 representing position of given values
        # the center of image is given by last position of target
        return tranformed_state

    def to_gazebo_state(self, pose):
        state = ModelState()
        state.model_name = "target"
        state.pose.position.x = pose.pose.position.x
        state.pose.position.y = pose.pose.position.y
        state.pose.position.z = pose.pose.position.z
        state.pose.orientation.w = pose.pose.orientation.w
        state.pose.orientation.x = pose.pose.orientation.x
        state.pose.orientation.y = pose.pose.orientation.y
        state.pose.orientation.z = pose.pose.orientation.z
        return state

    def to_pose(self, actions):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.seq = self.sequential_number
        self.sequential_number += 1
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = actions[0]
        pose.pose.position.y = actions[1]
        pose.pose.position.z = actions[2]
        quat_tf = tf.transformations.quaternion_from_euler(actions[3], actions[4], actions[5])
        pose.pose.orientation = Quaternion(quat_tf[0], quat_tf[1], quat_tf[2], quat_tf[3])
        return pose

    def create_new_state(self, drone_pose, target_pose):
        drones_positions = self.last_state[1]
        target_positions = self.last_state[0]
        d_p = drones_positions[1:self.frame_history].append([drone_pose.pose.position.x, drone_pose.pose.position.y])
        t_p = target_positions[1:self.frame_history].append([target_pose.pose.position.x, target_pose.pose.position.y])
        map_id = len(self.maps) - 1
        return np.array([t_p, d_p, map_id])

    def create_reward(self):
        reward = self.target_info.quotient
        drone_pose = self.last_state[1][self.frame_history - 1]
        target_pose = self.last_state[0][self.frame_history - 1]
        next_drone_pose = self.states[0]
        next_target_pose = self.states[1]
        shiftx = next_drone_pose.pose.position.x - drone_pose.pose.position.x
        shifty = next_drone_pose.pose.position.y - drone_pose.pose.position.y
        hypot = np.hypot(shiftx, shifty)
        reward += hypot * self.battery_const
        if hypot > self.max_shift:
            reward += (hypot - self.max_shift) * self.overlimit_const
        return reward

    def callback_target_information(self, data):
        self.target_info_time = rospy.Time.now()
        self.target_info = data

    def restart_simulation(self):
        nodes = os.popen("rosnode list").readlines()
        gazebo_nodes = []
        for i in range(len(nodes)):
            if str(nodes[i]).find("gazebo") != -1:
                gazebo_nodes.append(nodes[i])
        for node in gazebo_nodes:
            os.system("rosnode kill " + node)
        os.system(
            "roslaunch velocity_controller bebop_cmd_vel.launch extra_localization:=false\
             world_name:=box_world world_path:=$(find inputs_generator)/worlds")

    def find_random_place(self):
        max_distance = 14
        min_distance = 4
        while True:
            x = np.random.random_integers(0, self.map.shape[0])
            y = np.random.random_integers(0, self.map.shape[1])
            if self.map[x, y] == 255:
                i, j = self.free_place_in_field(x, y, min_distance, max_distance)
                if i is not None:
                    return x, y, i, j

    def free_place_in_field(self, x, y, min_distance, max_distance):
        min_x = x - max_distance
        if min_x < 0:
            min_x = 0
        max_x = x + max_distance
        if max_x > self.map.shape[0]:
            max_x = self.map.shape[0]
        min_y = y - max_distance
        if min_y < 0:
            min_y = 0
        max_y = y + max_distance
        if max_y > self.map.shape[1]:
            max_y = self.map.shape[1]

        min_xs = x - min_distance
        if min_xs < 0:
            min_xs = 0
        max_xs = x + min_distance
        if max_xs > self.map.shape[0]:
            max_xs = self.map.shape[0]
        min_ys = y - min_distance
        if min_ys < 0:
            min_ys = 0
        max_ys = y + min_distance
        if max_ys > self.map.shape[1]:
            max_ys = self.map.shape[1]
        range_x = range(min_x, min_xs) + range(max_xs, max_x)
        range_y = range(min_y, min_ys) + range(max_ys, max_y)
        for i in range_x:
            for j in range_y:
                if self.map[i, j] == 255:
                    return i, j
        return None, None


env = RenEnviroment()
state, done = env.reset()
next_state, reward, done, _ = env.step([])
