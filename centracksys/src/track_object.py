#!/usr/bin/env python
import rospy
import copy
import json
from collections import deque, namedtuple
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Bool
from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
import moveit_commander
import sys
from pesat_msgs.srv import PredictionMaps, PositionPrediction
import dronet_perception.msg as dronet_perception
from pesat_msgs.msg import ImageTargetInfo


class TargetPursuer(object):

    def __init__(self):
        super(TargetPursuer, self).__init__()
        rospy.init_node('track_system', anonymous=False)
        self.params = rospy.get_param("track")
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.track_timeout = self.params["track_timeout"]  # seconds
        self.current_waypoint_index = 0
        self.center_target_limit = self.params["center_target_limit"]  # pixels
        self.max_camera_vel = self.params["max_camera_vel"]
        self.max_linear_vel = self.params["max_linear_vel"]
        self.max_angular_vel = self.params["max_angular_vel"]
        self.distance_limit = self.params["distance_limit"]
        self.yaw_limit = self.params["yaw_limit"]
        self.camera_limit = self.params["camera_limit"]
        self.camera_max_angle = self.params["camera_max_angle"]
        self.init_target_localization()
        self.init_prediction()
        self.init_move_it()
        self.init_dronet()

    def init_dronet(self):
        self.pub_dronet = rospy.Publisher(self.params['state_change_topic'], Bool, queue_size=10)
        self.dronet_cmd_vel = None
        self.dronet_prediction = None
        rospy.Subscriber(self.params["dronet_cmd_vel_topic"], Twist, self.callback_dronet_vel)
        rospy.Subscriber(self.params["cnn_prediction_topic"], dronet_perception.CNN_out,
                         self.callback_dronet_prediction)

    def init_target_localization(self):
        rospy.Subscriber(self.params["target_information_topic"], ImageTargetInfo, self.callback_target_information)
        self.target_information = None
        self.target_seen_time = 0
        self.image_width = self.params["image_width"]
        self.image_height = self.params["image_height"]
        self.last_seen = 0  # we start in search mode
        self.last_position_time = -1

    def init_prediction(self):
        self.position_epsilon = self.params["position_epsilon"]  # meters
        self.angle_epsilon = self.params["angle_epsilon"]
        self.goal_position_epsilon = self.params["goal_position_epsilon"]  # meters
        self.goal_angle_epsilon = np.pi / self.params["goal_angle_epsilon_divider"]
        self.goal_angle_camera_epsilon = np.pi / self.params["goal_angle_camera_epsilon_divider"]
        self.history_positions = deque(maxlen=self.params["history_position_deque_len"])  # one minute per rate 20 Hz
        self.Pose = namedtuple("pose", ["x", "y", "yaw"])
        self.history_query_length = self.params["history_query_length"]  # frames/dates
        self.future_positions_srv = rospy.ServiceProxy(self.params['target_position_predictions_service'],
                                                       PredictionMaps)
        self.goal_positions_srv = rospy.ServiceProxy(self.params['goal_drone_positions_service'], PositionPrediction)
        self.correction_timeout = self.params["correction_timeout"]  # seconds
        self.last_correction = 0  # not yet made
        self.pmap_size = self.params["pmap_size"]
        self.pmap_center = (self.params["pmap_center"]["x"], self.params["pmap_center"]["y"])
        self.pmaps = None
        self.goal_positions = None
        self.obstacles = None
        self.obstacles_file = self.params["obstacles_file"]
        self.planning_scene_diff_publisher = rospy.Publisher(self.params['planning_scene_topic'], PlanningScene,
                                                             queue_size=1)

    def init_move_it(self):
        # moveit initialization
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = self.params["group_name"]
        namespace = self.params["namespace"]
        self.jump_threshold = self.params["jump_threshold"]
        self.eef_step = self.params["eef_step"]
        self.move_group = moveit_commander.MoveGroupCommander(group_name, ns=namespace)
        self.plan = None

    def reset(self):
        self.history_positions.clear()

    def rounding(self, value):
        diff = value - int(value)
        if diff < 0.25:
            r = 0.0
        elif diff < 0.75:
            r = 0.5
        else:
            r = 1
        return int(value) + r

    # callbacks
    def callback_target_information(self, data):
        self.target_information = data
        if data:
            self.target_seen_time = rospy.Time().now()

    def callback_dronet_vel(self, data):
        self.dronet_cmd_vel = data

    def callback_dronet_prediction(self, data):
        self.dronet_prediction = data

    def get_coord_on_map(self, field):
        x_map = self.pmap_center[0] + (field % self.pmap_size - self.pmap_size / 2.0)
        y_map = self.pmap_center[1] + (field / self.pmap_size - self.pmap_size / 2.0)
        return (x_map, y_map)

    def get_predicted_yaw(self, x_map, y_map, x_map_next, y_map_next):
        direction_x = x_map_next - x_map
        direction_y = y_map_next - y_map
        direction_length = np.sqrt(np.power(direction_x, 2) + np.power(direction_y, 2))
        return np.arccos(1 / direction_length)

    def need_replanning(self, x_map, y_map, x_map_next, y_map_next, target_x, target_y, target_yaw):
        if np.abs(x_map - target_x) < self.position_epsilon and np.abs(y_map - target_y) < self.position_epsilon:
            if x_map != x_map_next and y_map != y_map_next:
                angle = self.get_predicted_yaw(x_map, y_map, x_map_next, y_map_next)
                if np.abs(angle - target_yaw) < self.goal_angle_epsilon:
                    return False
                else:
                    return True
            else:
                return False
        else:
            return True

    def need_new_path(self, goal_position):
        # check if values aren't outside of limit
        origin_x = self.goal_positions.pose.position.x
        origin_y = self.goal_positions.pose.position.y
        origin_z = self.goal_positions.pose.position.z
        (origin_roll, origin_pitch, origin_yaw) = tf.transformations.euler_from_quaternion(
            self.goal_positions.pose.orientation)
        new_x = goal_position.pose.position.x
        new_y = goal_position.pose.position.y
        new_z = goal_position.pose.position.z
        (new_roll, new_pitch, new_yaw) = tf.transformations.euler_from_quaternion(goal_position.pose.orientation)
        if abs(origin_x - new_x) < self.goal_position_epsilon and abs(
                origin_y - new_y) < self.goal_position_epsilon and abs(
            origin_z - new_z) < self.goal_position_epsilon and abs(
            origin_yaw - new_yaw) < self.goal_angle_epsilon and abs(
            origin_roll - new_roll) < self.goal_angle_camera_epsilon and abs(
            origin_pitch - new_pitch) < self.goal_angle_camera_epsilon:
            return False
        else:
            return True

    def set_new_plan(self):
        waypoints = []
        for p in self.goal_positions:
            pose_goal = Pose()
            pose_goal.orientation.w = 1.0
            pose_goal.position.x = p.pose.position.x
            pose_goal.position.y = p.pose.position.y
            pose_goal.position.z = p.pose.position.z
            (_, _, yaw) = tf.transformations.euler_from_quaternion(p.pose.orientation)
            quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
            pose_goal.orientation.x = quaternion[0]
            pose_goal.orientation.y = quaternion[1]
            pose_goal.orientation.z = quaternion[2]
            pose_goal.orientation.w = quaternion[3]
            waypoints.append(copy.deepcopy(pose_goal))
        # add obstacles to planning scene
        self.load_obstacles()
        # plan new path
        (self.plan, fraction) = self.move_group.compute_cartesian_path(waypoints, self.eef_step, self.jump_threshold)
        return

    def load_obstacles(self):
        if self.obstacles is None:
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
                self.planning_scene_diff_publisher.publish(planning_scene);

    def get_new_probability_maps(self, buffer_length):
        try:
            self.pmaps = self.future_positions_srv(
                self.history_positions[buffer_length - self.history_query_length:])
            self.last_correction = rospy.Time.now()
            self.pmap_center = self.history_positions[buffer_length - 1][0:2]
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        return

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

    def get_cmd_vel_dronet(self):
        if self.dronet_prediction.collision_prob > 1:
            return self.dronet_cmd_vel
        twist = Twist()
        try:
            # camera centering
            trans = self.tfBuffer.lookup_transform(self.params["camera_base_link"], self.params['camera_optical_link'],
                                                   rospy.Time())
            explicit_quat = [trans.transform.rotation.x, trans.transform.rotation.y,
                             trans.transform.rotation.z, trans.transform.rotation.w]
            (horizontal, _, vertical) = tf.transformations.euler_from_quaternion(explicit_quat)
            start_position = -np.pi / 2
            horizontal_shift = horizontal - start_position
            vertical_shift = vertical - start_position
            twist.angular.x = horizontal_shift
            twist.angular.y = vertical_shift
            # drone centering
            trans_last = self.tfBuffer.lookup_transform(self.params["map"], self.params['base_link'], rospy.Time())
            trans = self.tfBuffer.lookup_transform(self.params["map"], self.params['base_link'],
                                                   trans.header.stamp - 0.01)
            vel_x = trans_last.transform.translation.x - trans.transform.translation.x
            vel_y = trans_last.transform.translation.y - trans.transform.translation.y
            angle = np.arctan2(vel_y / vel_x)
            twist.angular.z = angle
        except:
            pass
        return Twist

    def get_cmd_vel_plan(self, drone_pose):
        # path to current waypoint
        waypoint = self.plan[self.current_waypoint_index]
        vel_x = waypoint.position.x - drone_pose.position.x
        vel_y = waypoint.position.y - drone_pose.position.y
        vel_z = waypoint.position.z - drone_pose.position.z
        explicit_quat = [drone_pose.rotation.x, drone_pose.rotation.y,
                         drone_pose.rotation.z, drone_pose.rotation.w]
        (_, _, drone_yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
        explicit_quat = [waypoint.rotation.x, waypoint.rotation.y,
                         waypoint.rotation.z, waypoint.rotation.w]
        (_, _, waypoint_yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
        vel_yaw = waypoint_yaw - drone_yaw
        vel_x = np.clip(vel_x, -self.max_linear_vel, self.max_linear_vel)
        vel_y = np.clip(vel_y, -self.max_linear_vel, self.max_linear_vel)
        vel_z = np.clip(vel_z, -self.max_linear_vel, self.max_linear_vel)
        vel_yaw = np.clip(vel_yaw, -self.max_angular_vel, self.max_angular_vel)
        if vel_x < self.distance_limit and vel_y < self.distance_limit and vel_z < self.distance_limit and \
                vel_yaw < self.yaw_limit:
            self.current_waypoint_index += 1
        if len(self.plan) > self.current_waypoint_index:
            return None
        t = Twist()
        t.linear.x = vel_x
        t.linear.y = vel_y
        t.linear.z = vel_z
        t.angular.z = vel_yaw
        return t

    def get_cmd_vel_image(self):
        if self.target_information.quotient > 0:
            # shift of target in image from image axis
            diffX = np.abs(self.image_width / 2 - self.target_information.centerX)
            diffY = np.abs(self.image_height / 2 - self.target_information.centerY)
            d = np.sqrt(np.square(diffX) + np.square(diffY))
            # is the distance of target from center in the image too long
            if d > self.center_target_limit:
                # sign for shift of camera
                signX = np.sign(diffX)
                signY = np.sign(diffY)
                # shift angle of target in image for camera
                angleX = signX * np.arctan2(diffX, self.focal_length)
                angleY = signY * np.arctan2(diffY, self.focal_length)
                twist = Twist()
                twist.angular.x = angleX
                twist.angular.y = angleY
                try:
                    trans = self.tfBuffer.lookup_transform(self.params["map"], self.params['camera_optical_link'],
                                                           rospy.Time())
                    explicit_quat = [trans.transform.rotation.x, trans.transform.rotation.y,
                                     trans.transform.rotation.z, trans.transform.rotation.w]
                    (horizontal, vertical, _) = tf.transformations.euler_from_quaternion(explicit_quat)
                    angle_limit = 0.8 * self.camera_max_angle
                    residue = self.camera_max_angle - angle_limit
                    divided_res = residue / 10.0
                    divided_lin_vel = self.max_linear_vel / 10.0
                    divided_ang_vel = self.max_angular_vel / 10.0
                    if horizontal < -angle_limit:
                        # slowing
                        surplus = -horizontal - angle_limit
                        twist.linear.x = -1 * (surplus / divided_res) * divided_lin_vel
                    if horizontal > angle_limit:
                        # speeding up
                        surplus = horizontal - angle_limit
                        twist.linear.x = (surplus / divided_res) * divided_lin_vel
                    if vertical < -angle_limit:
                        # turn right
                        surplus = -horizontal - angle_limit
                        twist.angular.z = -1 * (surplus / divided_res) * divided_ang_vel
                    if vertical > angle_limit:
                        # turn left
                        surplus = horizontal - angle_limit
                        twist.angular.z = (surplus / divided_res) * divided_ang_vel
                except (
                        tf2_ros.LookupException, tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException):
                    rospy.loginfo("No map -> camera_optical_link transformation!")
                return twist
        return Twist()

    def choose_right_vel(self, t_dronet, t_image, t_plan):
        # something in the way of drone
        if np.abs(t_dronet.linear.x) > 0:
            return t_dronet
        # target too far from center of camera optical axis
        elif np.abs(t_image.linear.x) > 0:
            twist = Twist()
            twist.linear.x = np.clip(0.7 * t_image.linear.x + 0.3 * t_plan.linear.x, -self.max_linear_vel,
                                     self.max_linear_vel)
            twist.linear.y = np.clip(t_plan.linear.y, -self.max_linear_vel, self.max_linear_vel)
            twist.linear.z = np.clip(t_plan.linear.z, -self.max_linear_vel, self.max_linear_vel)
            twist.angular.x = np.clip(t_image.angular.x, -self.max_camera_vel, self.max_camera_vel)
            twist.angular.y = np.clip(t_image.angular.y, -self.max_camera_vel, self.max_camera_vel)
            twist.angular.z = np.clip(0.7 * t_image.angular.z + 0.3 * t_plan.angular.z, -self.max_angular_vel,
                                      self.max_angular_vel)
            return twist
        # compromise between Dronet and Image targeting, target in field of view is more important
        elif t_dronet.linear.x == 0 and t_image.linear.x == 0:
            twist = Twist()
            k1 = 2
            k2 = 1
            twist.angular.x = np.clip((k1 * t_image.angular.x + k2 * t_dronet.angular.x) / (k1 + k2),
                                      -self.max_camera_vel, self.max_camera_vel)
            twist.angular.y = np.clip((k1 * t_image.angular.y + k2 * t_dronet.angular.y) / (k1 + k2),
                                      -self.max_camera_vel, self.max_camera_vel)
            twist.angular.z = np.clip(0.7 * t_dronet.angular.z + 0.3 * t_plan.angular.z, -self.max_angular_vel,
                                      self.max_angular_vel)
            twist.linear.x = np.clip(t_plan.linear.x, -self.max_linear_vel, self.max_linear_vel)
            twist.linear.y = np.clip(t_plan.linear.y, -self.max_linear_vel, self.max_linear_vel)
            twist.linear.z = np.clip(t_plan.linear.z, -self.max_linear_vel, self.max_linear_vel)
        else:
            return t_plan

    def add_position_to_history(self, target_x, target_y, target_yaw, position_time, buffer_length):
        if position_time != self.last_position_time:
            self.history_positions.append(self.Pose(target_x, target_y, target_yaw))
        else:
            self.history_positions[buffer_length - 1] = self.Pose(target_x, target_y, target_yaw)

    def step(self):
        # time from last prediction
        diff = rospy.Time.now() - self.last_correction
        # time is rounded up 0.5 s
        position_time = self.rounding(rospy.Time.now())
        rounded_time = self.rounding(diff)
        buffer_length = len(self.history_positions)
        # add new data
        if self.target_information.quotient > 0:
            # data from camera
            trans = self.tfBuffer.lookup_transform(self.params["map"], self.params['target_position'], rospy.Time())
            target_x = self.rounding(trans.transform.translation.x)
            target_y = self.rounding(trans.transform.translation.y)
            explicit_quat = [trans.transform.rotation.x, trans.transform.rotation.y,
                             trans.transform.rotation.z, trans.transform.rotation.w]
            (_, _, target_yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
            # add new position
            self.add_position_to_history(target_x, target_y, target_yaw, position_time, buffer_length)
        else:
            # tracking timeout -> start search
            if rospy.Time.now() - self.last_seen > self.track_timeout:
                return None
            else:
                # add predicted value
                if self.pmaps is not None:
                    (x_map, y_map, x_map_next, y_map_next) = self.current_next_predicted_poses(rounded_time)
                    angle = 0
                    if x_map != x_map_next and y_map != y_map_next:
                        angle = self.get_predicted_yaw(x_map, y_map, x_map_next, y_map_next)
                    self.add_position_to_history(x_map, y_map, angle, position_time, buffer_length)
                # add last position
                elif buffer_length > 0:
                    pose = self.history_positions[buffer_length - 1]
                    self.add_position_to_history(pose.x, pose.y, pose.yaw, position_time, buffer_length)
                # try to add position of drone
                else:
                    try:
                        trans = self.tfBuffer.lookup_transform(self.params["map"], self.params['base_link'],
                                                               rospy.Time())
                        self.add_position_to_history(trans.transform.translation.x, trans.transform.translation.y, 0,
                                                     position_time, buffer_length)
                    except:
                        self.add_position_to_history(0, 0, 0, position_time, buffer_length)

        # enough data to perform prediction
        if buffer_length > self.history_query_length:
            # time from last predition correction
            replaning = False
            if self.pmaps is not None:
                pose = self.history_positions[buffer_length - 1]
                (x_map, y_map, x_map_next, y_map_next) = self.current_next_predicted_poses(rounded_time)
                # is target in neighbourhood from prediction
                replaning = self.need_replanning(x_map, y_map, x_map_next, y_map_next, pose.x, pose.y, pose.yaw)
            if self.plan is None or self.pmaps is None or diff > self.correction_timeout or replaning:
                self.get_new_probability_maps(buffer_length)
                replaning = True
            # new goal position
            if replaning:
                goal_position = self.goal_positions_srv(self.pmaps)
                new_path_needed = self.need_new_path(goal_position)
                # goal position is far than it can be accepted
                if new_path_needed:
                    self.goal_positions = goal_position
                    self.set_new_plan()
        # path following
        if self.plan is not None:
            t_dronet = self.get_cmd_vel_dronet()
            t_plan = self.get_cmd_vel_plan()
            t_image = self.get_cmd_vel_image()
            return self.choose_right_vel(t_dronet, t_image, t_plan)

        self.last_position_time = position_time


if __name__ == "__main__":
    tp = TargetPursuer()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        tp.step()
        rate.sleep()
