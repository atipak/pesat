#!/usr/bin/env python
import rospy
import copy
import json
import os
from collections import namedtuple
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose, Twist, Point
from tf2_geometry_msgs import PointStamped
from std_msgs.msg import Bool, Float32
from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
import moveit_commander
import sys
from pesat_msgs.srv import PredictionMaps, PositionPrediction, GoalState
import dronet_perception.msg as dronet_perception
from pesat_msgs.msg import ImageTargetInfo
import helper_pkg.utils as utils


class TargetPursuer(object):

    def __init__(self):
        super(TargetPursuer, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('track_system', anonymous=False)
        _camera_configuration = rospy.get_param("drone_bebop2")["camera"]
        _drone_configuration = rospy.get_param("drone_bebop2")["drone"]
        _track_configuration = rospy.get_param("track")
        _vision_configuration = rospy.get_param("target_localization")
        _env_configuration = rospy.get_param("environment")
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.track_timeout = _track_configuration["track_timeout"]  # seconds
        self.current_waypoint_index = 0
        self.center_target_limit = _track_configuration["center_target_limit"]  # pixels
        self.init_target_localization(_vision_configuration)
        self.init_prediction(_track_configuration)
        self.init_move_it(_track_configuration)
        self.track_restrictions()
        # TODO check constraints
        self._max_speed = 17
        self._max_target_speed = 7
        self._hfov = _camera_configuration["hfov"]
        self._image_height = _camera_configuration["image_height"]
        self._image_width = _camera_configuration["image_width"]
        self._focal_length = self._image_width / (2.0 * np.tan(self._hfov / 2.0))
        self._vfov = np.arctan2(self._image_height / 2, self._focal_length)
        self._dronet_crop = 200
        self._s = rospy.Service(_track_configuration["next_position_service"], GoalState, self.next_position_callback)
        rospy.Subscriber(_track_configuration["launch_topic"], Bool, self.init_reset_callback)
        self._launched = True
        self._is_checkpoint_reached = False
        self._current_pose = None
        self._change_state_to_searching = False
        self._map_frame = _env_configuration["map"]
        self._drone_base_link_frame = _drone_configuration["base_link"]
        self._camera_base_link_frame = _drone_configuration["camera_base_link"]
        self._target_location_frame = _vision_configuration["location_frame"]
        self._deep_save_time = 0
        self._use_deep_points = False
        self._target_image_height = _track_configuration["target_size"]["y"]
        self._target_image_width = _track_configuration["target_size"]["x"]
        ar = np.zeros((1000, 1000))
        ar.fill(255)
        self.map = utils.Map(ar)
        self.init_deep(_track_configuration)
        self.init_reactive()

    def init_target_localization(self, _vision_configuration):
        rospy.Subscriber(_vision_configuration["target_information"], ImageTargetInfo, self.callback_target_information)
        self.target_information = None
        self.last_seen = 0  # we start in search mode
        self.last_position_time = -1

    def init_prediction(self, _track_configuration):
        self.position_epsilon = _track_configuration["position_epsilon"]  # meters
        self.angle_epsilon = _track_configuration["angle_epsilon"]
        self.goal_position_epsilon = _track_configuration["goal_position_epsilon"]  # meters
        self.goal_angle_epsilon = np.pi / _track_configuration["goal_angle_epsilon_divider"]
        self.goal_angle_camera_epsilon = np.pi / _track_configuration["goal_angle_camera_epsilon_divider"]
        self.history_positions = utils.DataStructures.SliceableDeque(
            maxlen=_track_configuration["history_position_deque_len"])  # one minute per rate 20 Hz
        self.Pose = namedtuple("pose", ["x", "y", "yaw"])
        self.history_query_length = _track_configuration["history_query_length"]  # frames/dates
        self.future_positions_srv = rospy.ServiceProxy(_track_configuration['target_position_predictions_service'],
                                                       PredictionMaps)
        self.goal_positions_srv = rospy.ServiceProxy(_track_configuration['goal_drone_positions_service'],
                                                     PositionPrediction)
        self.correction_timeout = _track_configuration["correction_timeout"]  # seconds
        self.last_correction = 0  # not yet made
        self.pmap_size = _track_configuration["pmap_size"]
        self.pmap_center = (_track_configuration["pmap_center"]["x"], _track_configuration["pmap_center"]["y"])
        self.pmaps = None
        self._estimated_deep_poses = None
        self.obstacles = None
        self.planning_scene_diff_publisher = rospy.Publisher(_track_configuration['planning_scene_topic'],
                                                             PlanningScene,
                                                             queue_size=1)

    def init_move_it(self, _track_configuration):
        # moveit initialization
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.jump_threshold = _track_configuration["jump_threshold"]
        self.eef_step = _track_configuration["eef_step"]
        self.move_group = moveit_commander.MoveGroupCommander(_track_configuration["group_name"])
        # ,ns=_track_configuration["namespace"])
        self._deep_plan = None

    def init_reactive(self):
        self._estimated_reactive_poses = [Pose() for _ in range(6)]
        self._time_epsilon = 0.1
        self._ground_dist_limit = 0.2
        self._react_index = 0
        self._set_new_reactive_plan = False
        self._reactive_plan = []
        self.set_min_max_dist(1)
        self.set_camera_angle_limits()

    def init_deep(self, _track_configuration):
        self._deep_plan = []
        self._estimated_deep_poses = []
        self._deep_index = 0
        # add obstacles to planning scene
        self.obstacles_file = _track_configuration["obstacles_file"]
        self.load_obstacles()

    def track_restrictions(self):
        self._min_dist, self._max_dist = 0, 0

    def reset(self):
        self.history_positions.clear()

    # callbacks
    def callback_target_information(self, data):
        self.target_information = data
        if data.quotient > 0.7:
            self.last_seen = rospy.Time.now().to_sec()

    def next_position_callback(self, data):
        self._is_checkpoint_reached = data.last_point_reached
        return [self._current_pose, self._change_state_to_searching]

    def init_reset_callback(self, data):
        if not self._launched and data:
            self.start()
        elif not data and self._launched:
            self.stop()

    # general functions
    def start(self):
        self._launched = True

    def stop(self):
        self._launched = False

    def get_coord_on_map(self, field):
        x_map = self.pmap_center[0] + (field % self.pmap_size - self.pmap_size / 2.0)
        y_map = self.pmap_center[1] + (field / self.pmap_size - self.pmap_size / 2.0)
        return (x_map, y_map)

    def calculate_yaw_from_points(self, x_map, y_map, x_map_next, y_map_next):
        direction_x = x_map_next - x_map
        direction_y = y_map_next - y_map
        direction_length = np.sqrt(np.power(direction_x, 2) + np.power(direction_y, 2))
        return np.arccos(1 / direction_length)

    def load_obstacles(self):
        if self.obstacles is None:
            if os.path.isfile(self.obstacles_file):
                with open(self.obstacles_file, "r") as file:
                    self.obstacles = json.load(file)
                for obstacle in self.obstacles:
                    attached_object = AttachedCollisionObject()
                    attached_object.object.header.frame_id = "map"
                    attached_object.object.id = obstacle["name"]
                    pose = Pose()
                    pose.position.x = obstacle["x_pose"]
                    pose.position.y = obstacle["y_pose"]
                    pose.position.z = obstacle["z_pose"]
                    qt = tf.transformations.quaternion_from_euler(obstacle["r_orient"], obstacle["p_orient"],
                                                                  obstacle["y_orient"])
                    pose.orientation.x = qt[0]
                    pose.orientation.y = qt[1]
                    pose.orientation.z = qt[2]
                    pose.orientation.w = qt[3]
                    primitive = SolidPrimitive()
                    primitive.type = primitive.BOX
                    primitive.dimensions.resize(3)
                    primitive.dimensions[0] = obstacle["x_size"]
                    primitive.dimensions[1] = obstacle["y_size"]
                    primitive.dimensions[2] = obstacle["z_size"]
                    attached_object.object.primitives.push_back(primitive)
                    attached_object.object.primitive_poses.push_back(pose)
                    planning_scene = PlanningScene()
                    planning_scene.world.collision_objects.push_back(attached_object.object)
                    planning_scene.is_diff = True
                    self.planning_scene_diff_publisher.publish(planning_scene)

    def are_points_yaw_close(self, x_map, y_map, x_map_next, y_map_next, target_x, target_y, target_yaw):
        if np.abs(x_map - target_x) < self.position_epsilon and np.abs(y_map - target_y) < self.position_epsilon:
            if x_map != x_map_next and y_map != y_map_next:
                angle = self.calculate_yaw_from_points(x_map, y_map, x_map_next, y_map_next)
                if np.abs(angle - target_yaw) < self.goal_angle_epsilon:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False

    def pose_fitness(self, pose):
        return pose

    def transform_from_drone_to_map(self, point):
        try:
            point_stamped = PointStamped()
            point_stamped.header.stamp = rospy.Time.from_seconds(rospy.Time.now().to_sec() - 0.03)
            point_stamped.header.frame_id = self._drone_base_link_frame
            point_stamped.point.x = point.x
            point_stamped.point.y = point.y
            point_stamped.point.z = point.z
            transformed_point = self.tfBuffer.transform(point_stamped, self._map_frame)
            return transformed_point
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.loginfo("Problem with point transformation drone frame -> map frame: " + str(e))
            return None

    def add_position_to_history(self, target_x, target_y, target_yaw, position_time, buffer_length):
        rospy.loginfo(
            str(target_x) + ":" + str(target_y) + ":" + str(target_yaw) + ":" + str(position_time) + ":" + str(
                buffer_length))
        if position_time != self.last_position_time or buffer_length == 0:
            self.history_positions.append(self.Pose(target_x, target_y, target_yaw))
        else:
            self.history_positions[buffer_length - 1] = self.Pose(target_x, target_y, target_yaw)

    def add_new_pose_to_position_history(self, position_time, buffer_length, rounded_time):
        # drone sees target
        if self.target_information is not None and self.target_information.quotient > 0:
            # data from camera
            try:
                trans = self.tfBuffer.lookup_transform(self._map_frame, self._target_location_frame, rospy.Time())
                target_x = utils.Math.rounding(trans.transform.translation.x)
                target_y = utils.Math.rounding(trans.transform.translation.y)
                explicit_quat = [trans.transform.rotation.x, trans.transform.rotation.y,
                                 trans.transform.rotation.z, trans.transform.rotation.w]
                (_, _, target_yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
                # add new position
                self.add_position_to_history(target_x, target_y, target_yaw, position_time, buffer_length)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.loginfo("No target location -> map transformation!")
                return None
        # drone predicts only position of target
        else:
            # tracking timeout -> start search
            if rospy.Time.now().to_sec() - self.last_seen > self.track_timeout:
                return False
            else:
                # add predicted value
                if self.pmaps is not None:
                    (x_map, y_map, x_map_next, y_map_next) = self.current_next_predicted_poses(rounded_time)
                    angle = 0
                    if x_map != x_map_next and y_map != y_map_next:
                        angle = self.calculate_yaw_from_points(x_map, y_map, x_map_next, y_map_next)
                    self.add_position_to_history(x_map, y_map, angle, position_time, buffer_length)
                # add last position
                elif buffer_length > 0:
                    pose = self.history_positions[buffer_length - 1]
                    self.add_position_to_history(pose.x, pose.y, pose.yaw, position_time, buffer_length)
                # try to add position of drone
                else:
                    try:
                        trans = self.tfBuffer.lookup_transform(self._map_frame, self._drone_base_link_frame,
                                                               rospy.Time())
                        self.add_position_to_history(trans.transform.translation.x, trans.transform.translation.y, 0,
                                                     position_time, buffer_length)
                    except:
                        self.add_position_to_history(0, 0, 0, position_time, buffer_length)
        return True

    def image_target_center_dist(self):
        if self.target_information is not None and self.target_information.quotient > 0:
            # shift of target in image from image axis
            diffX = np.abs(self._image_width / 2 - self.target_information.centerX)
            diffY = np.abs(self._image_height / 2 - self.target_information.centerY)
            d = np.sqrt(np.square(diffX) + np.square(diffY))
            # is the distance of target from center in the image too long
            if d > self.center_target_limit:
                # sign for shift of camera
                signX = np.sign(diffX)
                signY = np.sign(diffY)
                # shift angle of target in image for camera
                angleX = signX * np.arctan2(diffX, self._focal_length)
                angleY = signY * np.arctan2(diffY, self._focal_length)
                return angleX, angleY
        return 0, 0

    def compare_react_deep_fitness(self, react_pose, deep_pose):
        return -1

    # reactive behaviour
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

    def recommended_dist(self, target_speed):
        diff_dist = self._max_dist - self._min_dist
        speed_ratio = target_speed / self._max_target_speed
        return diff_dist * speed_ratio + self._min_dist

    def get_altitude_limits(self, recommended_dist):
        min_altitude = 3
        max_altitude = 15
        max_v = recommended_dist / np.cos(self._min_dist_alt_angle)
        min_v = recommended_dist / np.cos(self._max_dist_alt_angle)
        return np.clip(min_v, min_altitude, max_altitude), np.clip(max_v, min_altitude, max_altitude)

    def recommended_altitude(self, recommended_dist, min_v, max_v):
        ratio = (recommended_dist - self._min_dist) / (self._max_dist - self._min_dist)
        numerator = max_v - min_v
        return ratio * numerator + min_v

    def recommended_ground_dist_alt(self, recommended_dist):
        min_v, max_v = self.get_altitude_limits(recommended_dist)
        alt = self.recommended_altitude(recommended_dist, min_v, max_v)
        ground_dist = np.sqrt(np.square(recommended_dist) + np.square(alt))
        return ground_dist, alt

    def recommended_vcamera(self, ground_dist, alt):
        return -(np.pi / 2 - np.arctan2(ground_dist, alt))

    def recommended_yaw_hcamera(self):
        try:
            trans_camera_drone = self.tfBuffer.lookup_transform(self._drone_base_link_frame,
                                                                self._camera_base_link_frame, rospy.Time())
            trans_drone_map = self.tfBuffer.lookup_transform(self._map_frame,
                                                             self._drone_base_link_frame, rospy.Time())
            angleX, _ = self.image_target_center_dist()
            explicit_quat = [trans_camera_drone.transform.rotation.x, trans_camera_drone.transform.rotation.y,
                             trans_camera_drone.transform.rotation.z, trans_camera_drone.transform.rotation.w]
            (_, _, camera_yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
            explicit_quat = [trans_drone_map.transform.rotation.x, trans_drone_map.transform.rotation.y,
                             trans_drone_map.transform.rotation.z, trans_drone_map.transform.rotation.w]
            (_, _, drone_yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
            yaw = drone_yaw
            angleX += camera_yaw
            if abs(angleX) > self._yaw_camera_limit:
                yaw += angleX
                angleX = 0
            return yaw, angleX
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.loginfo("No camera_case_link -> base_link -> map transformation: " + str(e))
            return 0, 0

    def recommended_speed(self, yaw, angleX, target_speed):
        point = Point()
        try:
            trans_drone_map = self.tfBuffer.lookup_transform(self._map_frame, self._drone_base_link_frame,
                                                             rospy.Time())
            explicit_quat = [trans_drone_map.transform.rotation.x, trans_drone_map.transform.rotation.y,
                             trans_drone_map.transform.rotation.z, trans_drone_map.transform.rotation.w]
            (_, _, drone_yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
            if angleX != 0:
                slide = target_speed * np.tan(angleX)
            else:
                slide = target_speed * np.tan(yaw - drone_yaw)
            point.x = target_speed
            point.y = slide
            point.z = trans_drone_map.transform.translation.z
            return point
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("No map -> camera_optical_link transformation!")
            return point

    def modify_ground_dist(self, current_ground_dist, recommended_ground_dist, point):
        if abs(current_ground_dist - recommended_ground_dist) > self._ground_dist_limit:
            ratio = current_ground_dist / recommended_ground_dist
            modified_point = point * ratio
            modified_point.z = point.z
            return modified_point
        return point

    def recommended_start_position(self, r_ground_dist):
        try:
            trans_target_drone = self.tfBuffer.lookup_transform(self._drone_base_link_frame,
                                                                self._target_location_frame, rospy.Time())
            d = np.hypot(trans_target_drone.transform.translation.x, trans_target_drone.transform.translation.y)
            p = Point()
            p.x = r_ground_dist - d
            p.y = 0
            p.z = 0
            transformed = self.transform_from_drone_to_map(p)
            return transformed
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("No target -> drone -> map transformation!")
            return PointStamped()

    def recommended_next_pose(self, current_target_speed, time):
        r_dist = self.recommended_dist(current_target_speed)
        r_ground_dist, r_alt = self.recommended_ground_dist_alt(r_dist)
        r_yaw, r_hcamera = self.recommended_yaw_hcamera()
        r_vcamera = self.recommended_vcamera(r_ground_dist, r_alt)
        r_speed = self.recommended_speed(r_yaw, r_hcamera, current_target_speed)
        r_start_position = self.recommended_start_position(r_ground_dist)
        p = Point()
        p.x = r_start_position.point.x + time * r_speed.x
        p.y = r_start_position.point.y + time * r_speed.y
        p.z = r_speed.z
        transformed_point = self.transform_from_drone_to_map(p)
        pose = Pose()
        if transformed_point is not None:
            pose.position.x = transformed_point.point.x
            pose.position.y = transformed_point.point.y
            pose.position.z = r_alt
            pose.orientation.x = r_hcamera
            pose.orientation.y = r_vcamera
            pose.orientation.z = r_yaw
            return pose
        return pose

    def check_reactive_plan(self, time, target_speed):
        if (time % 0.5) < self._time_epsilon:
            print("Check_time")
            if (time % 3) < self._time_epsilon:
                print("Replanning")
                if not self._set_new_reactive_plan:
                    self._set_new_reactive_plan = True
                    self._reactive_time = time
                    self.react_replanning(0, time, target_speed)
            else:
                self._set_new_reactive_plan = False
            next_pose_time = 0.5
            next_pose = self.recommended_next_pose(target_speed, next_pose_time)
            next_estimated_pose_time_index = int((time - self._time_epsilon) / 0.5)
            if next_estimated_pose_time_index >= len(self._estimated_reactive_poses):
                next_estimated_pose = None
            else:
                next_estimated_pose = self._estimated_reactive_poses[next_estimated_pose_time_index]
            if not self.compare_poses(next_pose, next_estimated_pose):
                print("Repplna")
                start_index = int((time % 3) / 0.5)
                self.react_replanning(start_index, time, target_speed)

    def react_replanning(self, start_index, time, target_speed):
        for t in range(start_index, 6):
            pose_time = (t + 1 - start_index) * 0.5
            self._estimated_reactive_poses[t] = self.recommended_next_pose(target_speed, pose_time)
        self._reactive_plan = self.get_plan_from_poses(0, [self._estimated_reactive_poses[5]])

    def compare_poses(self, pose1, pose2):
        if pose1 is None or pose2 is None:
            return False
        if abs(pose1.position.x - pose2.position.x) > self.position_epsilon:
            return False
        if abs(pose1.position.y - pose2.position.y) > self.position_epsilon:
            return False
        if abs(pose1.position.z - pose2.position.z) > self.position_epsilon:
            return False
        if abs(pose1.orientation.x - pose2.orientation.x) > self.angle_epsilon:
            return False
        if abs(pose1.orientation.y - pose2.orientation.y) > self.angle_epsilon:
            return False
        if abs(pose1.orientation.z - pose2.orientation.z) > self.angle_epsilon:
            return False
        return True

    def get_plan_from_poses(self, start_index, collection):
        entire_plan = []
        c_state = self.robot.get_current_state()
        for t in range(start_index, len(collection)):
            if t - 1 >= 0:
                last_pose = collection[t - 1]
                c_state.multi_dof_joint_state.transforms[0].translation.x = last_pose.position.x
                c_state.multi_dof_joint_state.transforms[0].translation.y = last_pose.position.y
                c_state.multi_dof_joint_state.transforms[0].translation.z = last_pose.position.z
                q = tf.transformations.quaternion_from_euler(0.0, 0.0, last_pose.orientation.z)
                c_state.multi_dof_joint_state.transforms[0].rotation.x = q[0]
                c_state.multi_dof_joint_state.transforms[0].rotation.y = q[1]
                c_state.multi_dof_joint_state.transforms[0].rotation.z = q[2]
                c_state.multi_dof_joint_state.transforms[0].rotation.w = q[3]
            pose = collection[t]
            q = tf.transformations.quaternion_from_euler(0.0, 0.0, pose.orientation.z)
            target_joints = [pose.position.x, pose.position.y, pose.position.z, q[0], q[1], q[2], q[3]]
            self.move_group.set_workspace([-10, -10, -10, 10, 10, 10])
            self.move_group.set_joint_value_target(target_joints)
            self.move_group.set_start_state(c_state)
            plan = self.move_group.plan()
            plan = [point.transforms for point in plan.multi_dof_joint_trajectory.points]
            entire_plan.extend(plan)
            self.move_group.clear_pose_targets()
        return entire_plan

    def next_react_pose(self, time, is_close):
        start_index = int((time % 3) / 0.5)
        if self._reactive_plan is None or self._estimated_reactive_poses is None or len(
                self._reactive_plan) <= self._react_index or len(self._estimated_reactive_poses) <= start_index:
            return None, None
        if is_close:
            self._react_index += 1
        return self._estimated_reactive_poses[start_index], self._reactive_plan[0][self._react_index]

    # probabilistic and network planning
    def next_deep_pose(self, time, is_close):
        start_index = int((time % 3) / 0.5)
        if self._deep_plan is None or self._estimated_deep_poses is None or len(self._deep_plan) <= self._deep_index or \
                len(self._estimated_deep_poses) <= start_index:
            return None, None
        if is_close:
            self._deep_index += 1
        return self._estimated_deep_poses[start_index], self._deep_plan[self._deep_index]

    def reset_positions_probability_maps(self, count):
        try:
            buffer_length = len(self.history_positions)
            drone_poses = self.get_last_drone_positions(self.history_query_length, self._deep_save_time)
            target_poses = self.history_positions[buffer_length - self.history_query_length:]
            center = (target_poses[len(target_poses) - 1].x, target_poses[len(target_poses) - 1].y)
            map = self.map.crop_map(center, self.pmap_size, self.pmap_size)
            self.pmap_center = center
            drone_maps = self.transform_poses_to_probs_maps(drone_poses, center)
            drone_move_maps = utils.DataStructures.SliceableDeque(maxlen=self.history_query_length)
            for t_map in drone_maps:
                drone_move_maps.append(t_map)
            target_maps = self.transform_poses_to_probs_maps(target_poses, center)
            target_move_maps = utils.DataStructures.SliceableDeque(maxlen=self.history_query_length)
            for t_map in target_maps:
                target_move_maps.append(t_map)
            self._estimated_deep_poses = []
            for i in range(count):
                pmap = self.future_positions_srv(target_move_maps, drone_move_maps, map)
                target_move_maps.append(pmap)
                goal_position = self.goal_positions_srv(target_move_maps)
                self._estimated_deep_poses.append(goal_position)
                explicit_quat = [goal_position.orientation.x, goal_position.orientation.y,
                                 goal_position.orientation.z, goal_position.orientation.w]
                (_, _, yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
                drone_pose = [self.Pose(goal_position.position.x, goal_position.position.y, yaw)]
                drone_move_maps.append(self.transform_poses_to_probs_maps(drone_pose, center)[0])
                self.last_correction = rospy.Time.now().to_sec()
            self.pmaps = target_move_maps
            self._deep_save_time = rospy.Time.now().to_sec()
        except rospy.ServiceException as exc:
            rospy.logwarn("Service did not process request: " + str(exc))
        return

    def transform_poses_to_probs_maps(self, positions, center):
        maps = []
        center_pmap = int(self.pmap_size / 2)
        for t in range(0, len(positions)):
            pos_x, pos_y = int(positions[t].x - center[0]), int(positions[t].y - center[1])
            pmap = np.zeros((self.pmap_size, self.pmap_size), dtype=np.float32)
            pmap[center_pmap + pos_x, center_pmap + pos_y] = 0.6
            for v in [-1, 0, 1]:
                for h in [-1, 0, 1]:
                    x_c = center_pmap + pos_x + h
                    y_c = center_pmap + pos_y + v
                    if not self.map.is_free(x_c, y_c, 254):
                        acc = 0
                        for v1 in [-1, 0, 1]:
                            for h1 in [-1, 0, 1]:
                                x_coor = x_c + h1
                                y_coor = y_c + v1
                                if -1 <= x_coor <= 1 and -1 <= y_coor <= 1 and self.map.is_free(x_coor, y_coor, 254):
                                    acc += 1
                        add_value = 0.05 / acc
                        for v1 in [-1, 0, 1]:
                            for h1 in [-1, 0, 1]:
                                x_coor = x_c + h1
                                y_coor = y_c + v1
                                if -1 <= x_coor <= 1 and -1 <= y_coor <= 1 and self.map.is_free(x_coor, y_coor, 254):
                                    pmap[x_coor, y_coor] += add_value
                    else:
                        pmap[x_c, y_c] += 0.05
            maps.append(pmap)
        return maps

    def get_last_drone_positions(self, count, last_time):
        poses = []
        try:
            for t in range(count):
                if last_time - t * 0.5 > 0:
                    lookup_time = rospy.Time.from_seconds(last_time - t * 0.5)
                else:
                    lookup_time = rospy.Time.from_seconds(last_time)
                trans = self.tfBuffer.lookup_transform(self._map_frame, self._drone_base_link_frame, lookup_time)
                explicit_quat = [trans.transform.rotation.x, trans.transform.rotation.y,
                                 trans.transform.rotation.z, trans.transform.rotation.w]
                (_, _, yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
                poses.append(self.Pose(trans.transform.translation.x, trans.transform.translation.y, yaw))
        except (
                tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            rospy.loginfo("No map -> base_link transformation!")
        return poses

    def current_next_predicted_poses(self, rounded_time):
        index = int(rounded_time / 0.5)
        if index >= len(self.pmaps):
            return (0, 0, 0, 0)
        # probability map for current time
        pmap = self.pmaps[index]
        # field where is the target with the biggest probability
        field = np.argmax(pmap)
        # coordinates
        (x_map, y_map) = self.get_coord_on_map(field)
        if index + 1 >= len(self.pmaps):
            return (x_map, y_map, x_map, y_map)
        # probability map for current time + 0,5s
        pmap_next = self.pmaps[index + 1]
        # field where is the target with the biggest probability
        field_next = np.argmax(pmap_next)
        # coordinates
        (x_map_next, y_map_next) = self.get_coord_on_map(field_next)
        return (x_map, y_map, x_map_next, y_map_next)

    def check_deep_plan(self, rounded_time, diff):
        # only if there is enough data to perform prediction
        buffer_length = len(self.history_positions)
        if buffer_length > self.history_query_length:
            # time from last predition correction
            replaning = False
            if self.pmaps is not None:
                pose = self.history_positions[buffer_length - 1]
                (x_map, y_map, x_map_next, y_map_next) = self.current_next_predicted_poses(rounded_time)
                # is target in neighbourhood from prediction
                replaning = not self.are_points_yaw_close(x_map, y_map, x_map_next, y_map_next, pose.x, pose.y,
                                                          pose.yaw)
            if self._deep_plan is None or self.pmaps is None or diff > self.correction_timeout or replaning:
                self.reset_positions_probability_maps(buffer_length)
                self._deep_plan = self.get_plan_from_poses(0, self._estimated_deep_poses)
                self._deep_index = 0

    def get_target_speed(self):
        buffer_length = len(self.history_positions)
        if buffer_length > 5:
            x_speeds = []
            y_speeds = []
            for t in range(5):
                last_pos = self.history_positions[buffer_length - t - 1]
                before_last = self.history_positions[buffer_length - t - 2]
                x_speed = (last_pos.x - before_last.x) / 0.5
                y_speed = (last_pos.y - before_last.y) / 0.5
                x_speeds.append(x_speed)
                y_speeds.append(y_speed)
            return np.array(np.hypot(np.average(x_speeds), np.average(y_speeds)))
        else:
            return np.array(0.0)

    def next_drone_pose(self, is_close, rounded_time, diff):
        # calculate time, calculate speed of target
        self._use_deep_points = False # WARNING set FALSE
        time = rospy.Time.now().to_sec()
        target_speed = self.get_target_speed()
        self.check_reactive_plan(time, target_speed)
        self.check_deep_plan(rounded_time, diff)
        deep_pose, deep_plan_pose = self.next_deep_pose(time, is_close)
        react_pose, react_plan_pose = self.next_react_pose(time, is_close)
        use_deep_point = False
        if self._use_deep_points:
            start_index = int((time % 3) / 0.5)
            predict_time = self._reactive_time + (start_index + 1) * 0.5
            pose = self.recommended_next_pose(target_speed, predict_time)
            if self.compare_react_deep_fitness(pose, deep_pose) < 0:
                self.react_replanning(start_index, time, target_speed)
                self._use_deep_points = False

                use_deep_point = False
            else:
                use_deep_point = True
        else:
            if self.compare_react_deep_fitness(react_pose, deep_pose) > 0:
                self._use_deep_points = True
                use_deep_point = True
            else:
                use_deep_point = False
        if use_deep_point:
            if deep_plan_pose is None or deep_pose is None:
                return None
            explicit_quat = [deep_plan_pose.rotation.x, deep_plan_pose.rotation.y, deep_plan_pose.rotation.z,
                             deep_plan_pose.rotation.w]
            (_, _, yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
            pose = Pose()
            pose.position.x = deep_plan_pose.translation.x
            pose.position.y = deep_plan_pose.translation.y
            pose.position.z = deep_plan_pose.translation.z
            pose.orientation.x = deep_pose.orientation.x
            pose.orientation.x = deep_pose.orientation.y
            pose.orientation.z = yaw
            return pose
        else:
            if react_plan_pose is None or react_pose is None:
                print("None")
                return None
            print("React")
            explicit_quat = [react_plan_pose.rotation.x, react_plan_pose.rotation.y, react_plan_pose.rotation.z,
                             react_plan_pose.rotation.w]
            (_, _, yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
            pose = Pose()
            pose.position.x = react_plan_pose.translation.x
            pose.position.y = react_plan_pose.translation.y
            pose.position.z = react_plan_pose.translation.z
            pose.orientation.x = react_pose.orientation.x
            pose.orientation.x = react_pose.orientation.y
            pose.orientation.z = yaw
            return pose

    def step(self):
        # time from last prediction
        diff = rospy.Time.now().to_sec() - self.last_correction
        # time is rounded up 0.5 s
        position_time = utils.Math.rounding(rospy.Time.now().to_sec())
        rounded_time = utils.Math.rounding(diff)
        buffer_length = len(self.history_positions)
        # add new data
        if not self.add_new_pose_to_position_history(position_time, buffer_length, rounded_time):
            self._current_pose = None
            self._change_state_to_searching = True
        if self._launched:
            next_pose = self.next_drone_pose(self._is_checkpoint_reached, rounded_time, diff)
            self.last_position_time = position_time
            self._current_pose = next_pose
            self._change_state_to_searching = False
        else:
            self._current_pose = None
            self._change_state_to_searching = False


if __name__ == "__main__":
    tp = TargetPursuer()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        tp.step()
        rate.sleep()
