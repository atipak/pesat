#!/usr/bin/env python
from abc import abstractmethod
from builtins import super

import rospy
from collections import namedtuple
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose, Point
from gazebo_msgs.msg import ModelState, ModelStates
from std_msgs.msg import Float64, Int64
import helper_pkg.utils as utils
from scipy.spatial import cKDTree
from pesat_msgs.srv import PointFloat, PosePlan
from helper_pkg.move_it_server import MoveitServer


class TargetMoveit(MoveitServer):
    def __init__(self):
        target_configuration = rospy.get_param("target_configuration")
        super(TargetMoveit, self).__init__(target_configuration)
        self._srv_point_height = rospy.Service(target_configuration["map_server"]["point_height_service"], PointFloat,
                                self.callback_get_height)
        self._srv_admissibility = rospy.Service(target_configuration["map_server"]["admissibility_service"], PointFloat,
                                self.callback_is_admissible)
        self._pose_planning = rospy.Service(target_configuration["map_server"]["pose_planning_service"], PosePlan,
                                self.callback_get_plan)

    def callback_get_height(self, data):
        return 0

    def callback_get_plan(self, data):
        start_pose = data.start
        end_pose = data.end
        plan = self.get_plan_from_poses(start_pose, end_pose)
        return [plan]

    def callback_is_admissible(self, data):
        is_admissible = self.is_admissible(data.point)
        if is_admissible:
            return [1]
        else:
            return [0]

    def get_target_joints(self, end_position):
        target_joints = [end_position.position.x, end_position.position.y, end_position.orientation.z]
        return target_joints


class CustomMove(object):
    def __init__(self, moveit, map):
        super(CustomMove, self).__init__()
        self.target_configuration = rospy.get_param("target_configuration")
        self.drone_configuration = rospy.get_param("drone_configuration")
        self.environment_configuration = rospy.get_param("environment_configuration")
        self._map = map
        self._map_size = max(self._map.height, self._map.width)
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self._map_frame = self.environment_configuration["map"]["frame"]
        self._drone_base_link_frame = self.drone_configuration["properties"]["base_link"]
        self.moveit = moveit
        self.minimal_velocity = 0.2

    def calculate_position(self, current_position, original_direction, original_velocity):
        if abs(original_direction.x) < 0.01 and abs(original_direction.y) < 0.01:
            direction = self.choose_random_direction()
        else:
            direction = original_direction
        if original_velocity < self.minimal_velocity:
            velocity = self.minimal_velocity
        else:
            velocity = original_velocity
        return self.next_position(current_position, direction, velocity)

    @abstractmethod
    def next_position(self, current_position, original_direction, original_velocity):
        pass

    @abstractmethod
    def reset(self):
        pass

    def choose_random_direction(self):
        x = np.random.rand() * np.random.choice([-1, 1])
        y = np.random.rand() * np.random.choice([-1, 1])
        direction = np.array([x, y])
        direction = utils.Math.normalize(direction)
        p = Point()
        p.x = direction[0]
        p.y = direction[1]
        return p

    def calculate_new_direction(self, current_position, next_position):
        p = Point()
        dir = [next_position.x - current_position.x,
               next_position.y - current_position.y]
        dir = utils.Math.normalize(dir)
        p.x = dir[0]
        p.y = dir[1]
        return p


class ReactiveMove(CustomMove):
    def __init__(self, moveit, map):
        super(ReactiveMove, self).__init__(moveit, map)

    def directed_move(self, current_position, velocity, direction):
        next_position = utils.Math.points_add(current_position, utils.Math.point_constant_mul(direction, velocity))
        while not self.moveit.is_admissible(next_position):
            direction = self.choose_random_direction()
            next_position = utils.Math.points_add(current_position, utils.Math.point_constant_mul(direction, velocity))
        self._current_direction = direction
        return next_position, direction, velocity


class AnimalsMove(ReactiveMove):
    def __init__(self, moveit, max_velocity, map):
        super(AnimalsMove, self).__init__(moveit, map)
        self._max_velocity = max_velocity
        self._max_distance = 10.0
        self._test_points = 20
        self._invisible_react_probability = 0.1
        self._visible_react_probability = 0.75

    def next_position(self, current_position, original_direction, original_velocity):
        drone_target_direction, distance = self.get_drone_target_direction(current_position)
        speed = self._max_velocity
        direction = Point()
        if self.check_free_path(current_position, drone_target_direction, distance):
            if np.random.rand() < self._visible_react_probability:
                direction = drone_target_direction
        else:
            if np.random.rand() < self._invisible_react_probability:
                direction = drone_target_direction
        return self.directed_move(current_position, speed, direction)

    def get_drone_target_direction(self, target_position):
        point = Point()
        distance = 0.0
        try:
            map_frame = self._map_frame
            drone_base_link_frame = self._drone_base_link_frame
            drone_transform = self._tfBuffer.lookup_transform(map_frame, drone_base_link_frame, rospy.Time())
            difx = target_position.x - drone_transform.transform.translation.x
            dify = target_position.y - drone_transform.transform.translation.y
            distance = np.hypot(abs(dify), abs(difx))
            dir = utils.Math.normalize(np.array([difx, dify]))
            point.x = dir[0]
            point.y = dir[1]
        except Exception as e:
            pass
        return point, distance

    def reset(self):
        pass

    def check_free_path(self, current_position, direction, distance):
        step = distance / self._test_points
        for i in range(self._test_points):
            p = utils.Math.points_add(utils.Math.point_constant_mul(direction, step * i), current_position)
            m_point = self._map.map_point_from_real_coordinates(p.x, p.y, 0.0)
            if not self._map.is_free_for_target(m_point, 0.5, min_value=255):
                return False
        return True


class DirectMove(ReactiveMove):
    def __init__(self, moveit, map):
        super(DirectMove, self).__init__(moveit, map)
        self._max_srotate_angle = np.deg2rad(self.target_configuration["strategies"]["max_srotate_angle"])
        self._slight_turn_dir_probability = self.target_configuration["strategies"]["slight_turn_dir_probability"]
        self._inverse_change_probability = self.target_configuration["strategies"]["inverse_change_probability"]

    def next_position(self, current_position, original_direction, original_velocity):
        direction = original_direction
        direction = self.slightly_rotate_direction(direction)
        direction = self.inverse_change(direction)
        return self.directed_move(current_position, original_velocity, direction)

    def slightly_rotate_direction(self, direction):
        new_direction = direction
        if self._slight_turn_dir_probability > np.random.rand():
            angle = np.random.uniform(0, self._max_srotate_angle)
            sign = np.random.choice([1, -1], 1)[0]
            angle *= sign
            arr = [new_direction.x, new_direction.y]
            dir = utils.Math.rotate_2d_vector(angle, arr)
            point = Point()
            point.x = dir[0]
            point.y = dir[1]
            new_direction = point
        return new_direction

    def inverse_change(self, direction):
        new_direction = direction
        if self._inverse_change_probability > np.random.rand():
            new_direction.x = -new_direction.x
            new_direction.y = -new_direction.y
        return new_direction


class CornersMove(CustomMove):
    def __init__(self, moveit, map):
        super(CornersMove, self).__init__(moveit, map)
        if not self._map.corners:
            self._tree = None
        else:
            self._tree = cKDTree(self._map.corners)

    def corners_in_sector_and_distance(self, position, direction, points_distance, angle, radius):
        corners_in_directions = []
        distances_of_corners = []
        if self._tree is not None:
            indicies = self._tree.query_ball_point([position], r=points_distance)
            while abs(angle) < np.pi:
                sector = utils.Math.create_sector(position, angle, direction, radius)
                for index in indicies:
                    for i in index:
                        corner = self._map.corners[i]
                        p = utils.Math.Point(corner[0], corner[1])
                        if utils.Math.is_inside_sector(p, sector):
                            corners_in_directions.append(corner)
                            distances_of_corners.append(
                                utils.Math.euclidian_distance(utils.Math.Point(position[0], position[1]), p))
                if len(corners_in_directions) > 0:
                    break
                else:
                    angle += np.deg2rad(10)
        return corners_in_directions, distances_of_corners

    def center_of_rec_from_corner(self, corner):
        rectangle_id = self._map.corners_rectangles[corner[0]][corner[1]]
        return self._map.rectangles_centers[rectangle_id]

    @abstractmethod
    def params(self, original_direction):
        pass

    @abstractmethod
    def compute_probabilities(self, distance_of_corners):
        pass

    @abstractmethod
    def get_position(self, corner, vector):
        pass

    def next_position_skeleton(self, current_position, original_direction):
        direction, points_distance, angle, radius = self.params(original_direction)
        corners_in_directions, distances_of_corners \
            = self.corners_in_sector_and_distance([current_position.x, current_position.y], [direction.x, direction.y],
                                                  points_distance, angle, radius)
        if len(corners_in_directions) > 0:
            distances_of_corners = np.array(distances_of_corners)
            probabilities = self.compute_probabilities(distances_of_corners)
            index = np.random.choice(range(len(corners_in_directions)), 1, p=probabilities)
            corner = np.array(corners_in_directions)[index][0]
            center_of_rec = self.center_of_rec_from_corner(corner)
            vector = np.array([center_of_rec[0] - corner[0], center_of_rec[1] - corner[1]])
            vector = utils.Math.normalize(vector)
            position = self.get_position(corner, vector)
            p = Point()
            p.x = position[0]
            p.y = position[1]
            if position is not None and self.moveit.is_admissible(p):
                return p, self.calculate_new_direction(current_position, p)
        return None, Point()


class ZigzagMove(CornersMove):
    def __init__(self, moveit, map):
        super(ZigzagMove, self).__init__(moveit, map)

    def params(self, original_direction):
        points_distance = 15
        angle = np.deg2rad(30)
        direction_angle = np.random.uniform(22, 90)
        direction = utils.Math.normalize(
            utils.Math.rotate_2d_vector(direction_angle, [original_direction.x, original_direction.y]))
        direction_point = Point()
        direction_point.x = direction[0]
        direction_point.y = direction[1]
        radius = 10
        return direction_point, points_distance, angle, radius

    def compute_probabilities(self, distance_of_corners):
        nearest_indices = utils.Math.two_lowest_indices(distance_of_corners)
        p = np.zeros(len(distance_of_corners))
        p[nearest_indices[1]] = 1
        return p

    def get_position(self, corner, vector):
        position = [corner[0] + vector[0], corner[1] + vector[1]]
        return position

    def next_position(self, current_position, original_direction, original_velocity):
        next_position, direction = self.next_position_skeleton(current_position, original_direction)
        return next_position, direction, original_velocity


class BehindCornersMove(CornersMove):
    def __init__(self, moveit, map):
        super(BehindCornersMove, self).__init__(moveit, map)

    def params(self, original_direction):
        points_distance = 15
        angle = np.deg2rad(22)
        radius = 10
        return original_direction, points_distance, angle, radius

    def compute_probabilities(self, distance_of_corners):
        if len(distance_of_corners) == 1:
            return [1.0]
        sum = np.sum(distance_of_corners)
        probabilities = sum - distance_of_corners
        d_factor = 1 / np.sum(probabilities)
        return [p * d_factor for p in probabilities]

    def get_position(self, corner, vector):
        positions = [[corner[0] + vector[0], corner[1] + vector[1]],
                     [corner[0] + vector[1], corner[1] - vector[0]],
                     [corner[0] - vector[1], corner[1] + vector[0]]]
        plausible_positions = []
        for pos in positions:
            p = Point()
            p.x = pos[0]
            p.y = pos[1]
            if self.moveit.is_admissible(p):
                plausible_positions.append(pos)
        if len(plausible_positions) > 0:
            index = np.random.randint(0, len(plausible_positions))
            goal_position = plausible_positions[index]
            return goal_position
        return None

    def next_position(self, current_position, original_direction, original_velocity):
        next_position, direction = self.next_position_skeleton(current_position, original_direction)
        return next_position, direction, original_velocity


class PlanningMove(CustomMove):
    def __init__(self, moveit, map):
        super(PlanningMove, self).__init__(moveit, map)

    def next_position(self, current_position, original_direction, original_velocity):
        position = self.choose_random_position()
        return position, original_direction, original_velocity

    def choose_random_position(self):
        point = Point()
        point.x = np.random.uniform(0, self._map_size)
        point.y = np.random.uniform(0, self._map_size)
        return point


class ManualMove(CustomMove):
    def __init__(self, moveit, map):
        super(ManualMove, self).__init__(moveit, map)
        self._speed_left = 0.0
        self._speed_forward = 0.0
        rospy.Subscriber("/user/target/move_forward", Float64, self.callback_move_forward)
        rospy.Subscriber("/user/target/move_left", Float64, self.callback_move_left)

    def callback_move_forward(self, data):
        self._speed_forward = data.data

    def callback_move_left(self, data):
        self._speed_left = data.data

    def next_position(self, current_position, original_direction, original_velocity):
        next_position = Point()
        next_position.x = current_position.x + self._speed_forward
        next_position.y = current_position.y + self._speed_left
        direction = Point()
        direction.x = self._speed_forward
        direction.y = self._speed_left
        velocity = np.hypot(direction.x, direction.y)
        return next_position, direction, velocity


class Target():

    def __init__(self):
        rospy.init_node('target', anonymous=False)
        target_configuration = rospy.get_param("target_configuration")
        environment_configuration = rospy.get_param("environment_configuration")
        self._max_velocity = target_configuration["strategies"]["max_velocity"]
        self._min_velocity = target_configuration["strategies"]["min_velocity"]
        self._map_frame = environment_configuration["map"]["frame"]
        self._vel_change_probability = target_configuration["strategies"]["vel_change_probability"]
        self._velocity_sigma = target_configuration["strategies"]["velocity_sigma"]
        self._target_base_link_frame = target_configuration["properties"]["base_link"]
        strategies_enum = namedtuple("Strategies", ["Manual", "Animal", "Direct", "Planning", "Cornering", "Zigzag"])
        self.moveit = TargetMoveit()
        self._obstacles_file_path = environment_configuration["map"]["obstacles_file"]
        self._map = utils.Map.get_instance(self._obstacles_file_path)
        self._strategies = strategies_enum(0, 1, 2, 3, 4, 5)
        self._strategies_list = [
            ManualMove(self.moveit, self._map),
            AnimalsMove(self.moveit, self._max_velocity, self._map),
            DirectMove(self.moveit, self._map),
            PlanningMove(self.moveit, self._map),
            BehindCornersMove(self.moveit, self._map),
            ZigzagMove(self.moveit, self._map)
        ]
        self._is_fusion_active = False
        self._strategy = self._strategies.Manual
        self._target_state = None
        self._index = -1
        self._plan = []
        self._plan_index = 0
        self._destination_position = None
        self._current_velocity = 0.2
        self._current_direction = Point()
        self._pub_set_model = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self.next_change = 0
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_target_state)
        rospy.Subscriber("/user/target/strategy", Int64, self.callback_strategy_choice)

    def pick_strategy(self):
        if self._is_fusion_active:
            if self.next_change < rospy.Time.now().to_sec():
                self.next_change = rospy.Time.now().to_sec() + np.random.rand() * 5 * 60
                index = np.random.randint(1, len(self._strategies_list))
                self.change_stategy(index)
        return self._strategies_list[self._strategy]

    def callback_target_state(self, data):
        if self._index == -1:
            for i in range(len(data.name)):
                if data.name[i] == "target":
                    self._index = i
                    break
        self._target_state = data.pose[self._index]

    def callback_strategy_choice(self, data):
        if -1 < data.data < len(self._strategies) + 1:
            if data.data == len(self._strategies):
                self._is_fusion_active = True
            else:
                self._is_fusion_active = False
                self.change_stategy(data.data)

    def change_stategy(self, index):
        if self._strategy != index:
            self._strategy = index
            self.reset()

    def reset(self):
        for i in range(len(self._strategies_list)):
            self._strategies_list[i].reset()
        self._destination_position = None

    def get_current_position(self):
        if self._target_state is not None:
            point = Point()
            point.x = self._target_state.position.x
            point.y = self._target_state.position.y
            return point
        return None

    def velocity_change(self):
        if self._vel_change_probability > np.random.rand():
            current_vel = self._current_velocity
            additive_change = np.random.normal(0, self._velocity_sigma, 1)
            current_vel += additive_change
            if current_vel > self._max_velocity:
                current_vel = self._max_velocity
            if current_vel < self._min_velocity:
                current_vel = self._min_velocity
            self._current_velocity = current_vel
        return self._current_velocity

    def publish_pose(self, move_x, move_y):
        if self._target_state is not None:
            model = ModelState()
            model.model_name = "target"
            model.reference_frame = "world"
            model.pose = self._target_state
            model.pose.position.x += move_x
            model.pose.position.y += move_y
            self._pub_set_model.publish(model)

    def move_to_position(self, position, update_f):
        current_position = self.get_current_position()
        if current_position is not None:
            while True:
                difx = self._plan[self._plan_index].position.x - current_position.x
                dify = self._plan[self._plan_index].position.y - current_position.y
                direction = (utils.Math.normalize(np.array([difx, dify]))) * self._current_velocity * update_f
                if np.all(direction > np.array([difx, dify])) or np.sum(
                        np.isclose(self.point_to_array(current_position),
                                   self.pose_to_array(self._plan[self._plan_index]))) == 2:
                    self._plan_index += 1
                    if len(self._plan) <= self._plan_index + 1:
                        break
                else:
                    break
            direction = np.clip(direction, [-np.abs(difx), -np.abs(dify)], [np.abs(difx), np.abs(dify)])
            self.publish_pose(direction[0], direction[1])

    def point_to_array(self, point):
        return np.array([point.x, point.y])

    def point_to_pose(self, point):
        pose = Pose()
        pose.position.x = point.x
        pose.position.y = point.y
        pose.orientation.w = 1
        return pose

    def pose_to_array(self, pose):
        return np.array([pose.position.x, pose.position.y])

    def main(self):
        r = 10.0
        rate = rospy.Rate(r)
        f = 1.0 / r
        while not rospy.is_shutdown():
            current_position = self.get_current_position()
            if current_position is not None:
                if self._destination_position is None:
                    velocity = self._current_velocity
                    direction = self._current_direction
                    next_position, next_direction, next_velocity = self.pick_strategy().calculate_position(
                        current_position, direction, velocity)
                    plan = self.moveit.get_plan_from_poses(self.point_to_pose(current_position),
                                                           self.point_to_pose(next_position))
                    self._plan_index = 0
                    if len(plan) > 0:
                        self._destination_position = next_position
                        self._current_velocity = next_velocity
                        self._current_direction = next_direction
                        self._plan = plan
                if self._plan is not None:
                    if len(self._plan) > self._plan_index:
                        self.move_to_position(self._plan, f)
                    else:
                        self._destination_position = None
            rate.sleep()


if __name__ == '__main__':
    target = Target()
    target.main()
