#!/usr/bin/env python
import rospy
import copy
import json
from collections import deque, namedtuple
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import TwistWithCovariance, TransformStamped
from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
import moveit_commander
from pesat_msgs.msg import Notification, ImageTargetInfo, JointStates
from pesat_msgs.srv import PredictionMaps, PositionPrediction

noti = namedtuple("Notification", ["serial_number", "time", "likelihood", "x", "y", "dx", "dy"])
watch_point = namedtuple("WatchPoint", ["x", "y", "z", "yaw", "vertical_camera_turn"])


class Database(object):
    def __init__(self, maximum_length, maximal_time):
        self._notifications = {}
        self._last_serial_number = {}
        self._last_notification = 0
        self._earlest_notification = 0
        self._maximum_length = maximum_length
        self._maximal_time = maximal_time  # secs
        rospy.Subscriber("/cctv/notifications", Notification, self.callback_notify)

    @property
    def notifications(self):
        return self._notifications

    def callback_notify(self, data):
        notification = noti(data.serial_number, rospy.Time.now(), data.likelihood, data.x, data.y, data.velocity_x,
                            data.velocity_y)
        self.add(notification)

    def add(self, notification):
        if notification.serial_number in self._notifications:
            self._notifications[notification.serial_number] = notification
        else:
            # already more than maximal length
            if self._earlest_notification <= self._last_notification - self._maximum_length:
                self._notifications[self._earlest_notification] = None
                self._earlest_notification += 1
            else:
                if self._last_notification - self._maximum_length < 0:
                    self._earlest_notification = 0
                else:
                    self._earlest_notification += 1
            self._notifications[notification.serial_number] = notification
            self._last_notification += 1

    def database_check(self):
        i = self._earlest_notification
        while i <= self._last_notification:
            if self._notifications[i].time < rospy.Time.now() - self._maximal_time:
                self._notifications[i] = None
                i += 1
            else:
                break
        self._earlest_notification = i


class MeetingPointPlanner(object):
    def __init__(self):
        prediction_service_name = ""
        self._srv_prediction = rospy.ServiceProxy(prediction_service_name, PredictionMaps)
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self._database = Database(400, 300)
        self._meeting_point = None

    def next(self, drone_position):
        meeting_point = None
        probability_map = None
        return meeting_point, probability_map

    def interruption(self):
        return False

    def reset(self, found):
        if found:
            # clear everything
            self._meeting_point = None
        else:
            pass


class WatchPointPlanner(object):
    def __init__(self):
        drone_place_planner_srv_name = ""
        self._srv_position = rospy.ServiceProxy(drone_place_planner_srv_name, PositionPrediction)
        self._watch_points = None
        self._meeting_point = None
        self._probability_map = None
        self._last_index = -1
        self._last_distance = 0
        self._drone_speed = 17

    def next(self, drone_position, meeting_point, probability_map):
        self._meeting_point = meeting_point
        self._probability_map = probability_map
        self._watch_points = None
        return self._watch_points

    def interruption(self, index, found):
        if found:
            return True
        if index != self._last_index:
            self._last_distance = np.sqrt(np.square(self._meeting_point.x - self._watch_points[index].x) + np.square(
                self._meeting_point.y - self._watch_points[index].y) + np.square(
                self._meeting_point.z - self._watch_points[index].z))
        if self._last_distance * self._drone_speed > self._meeting_point.time:
            return True
        return False

    def reset(self):
        self._watch_points = None
        self._meeting_point = None
        self._probability_map = None
        self._last_index = -1
        self._last_distance = 0


class MovePlanner(object):
    def __init__(self):
        target_information_topic = ""
        cmd_vel_topic = ""
        desired_pos_topic = ""
        current_pos_topic = ""
        current_vel_topic = ""
        update_velocities_topic = ""
        rospy.Subscriber(target_information_topic, ImageTargetInfo, self.callback_target_information)
        rospy.Subscriber(cmd_vel_topic, TwistWithCovariance, self.callback_velocities)
        rospy.Subscriber(update_velocities_topic, JointStates, self.callback_update)
        self._pub_cmd_vel = rospy.Publisher(cmd_vel_topic, TwistWithCovariance, queue_size=10)
        self._pub_desired_pos = rospy.Publisher(desired_pos_topic, JointStates, queue_size=10)
        self._pub_current_pos = rospy.Publisher(current_pos_topic, JointStates, queue_size=10)
        self._pub_current_vel = rospy.Publisher(current_vel_topic, JointStates, queue_size=10)
        self._target_information = None
        self._current_velocities = None
        self._seen = False
        self._max_height = 22
        self._min_height = 5
        self._hfov = 1.7
        self._image_height = 480
        self._image_width = 856
        self._focal_length = self._image_width / (2.0 * np.tan(self._hfov / 2.0))
        self._max_vertical_angle = np.arctan2(self._image_height / 2, self._focal_length)
        self._index = 0
        self._watch_points = None
        self._vel_update = False
        self._approching_result = False
        self._is_approching = False

    # callbacks
    def callback_target_information(self, data):
        self._target_information = data

    def callback_velocities(self, data):
        self._current_velocities = data

    def callback_update(self, data):
        if self._vel_update:
            twist = TwistWithCovariance()
            twist.twist.linear.x = data.values[0]
            twist.twist.linear.y = data.values[1]
            twist.twist.linear.z = data.values[2]
            twist.twist.angular.x = data.values[3]
            twist.twist.angular.y = data.values[4]
            twist.twist.angular.z = data.values[5]
            self._pub_cmd_vel.publish(twist)

    def next(self, watch_points):
        self._watch_points = watch_points
        self._index = 0
        self._seen = False
        return True

    def make_move(self, drone_position, horizontal_camera_turn, verical_camera_turn):
        if self._watch_points is not None:
            # current position
            x_drone = drone_position.transform.translation.x
            y_drone = drone_position.transform.translation.y
            z_drone = drone_position.transform.translation.z
            (_, _, yaw_drone) = tf.transformations.euler_from_quaternion(drone_position.transform.rotation)
            # goal position
            x_point = self._watch_points[self._index].x
            y_point = self._watch_points[self._index].y
            z_point = self._watch_points[self._index].z
            yaw_point = self._watch_points[self._index].yaw
            horizontal_camera_turn_point = 0.0
            verical_camera_turn_point = self._watch_points[self._index].verical_camera_turn
            current_positions = [x_drone, y_drone, z_drone, horizontal_camera_turn, verical_camera_turn, yaw_drone]
            desired_positions = [x_point, y_point, z_point, horizontal_camera_turn_point,
                                 verical_camera_turn_point, yaw_point]
            current_velocities = [self._current_velocities.twist.linear.x,
                                  self._current_velocities.twist.linear.y,
                                  self._current_velocities.twist.linear.z,
                                  self._current_velocities.twist.angular.x,
                                  self._current_velocities.twist.angular.y,
                                  self._current_velocities.twist.angular.z]
            self._pub_desired_pos.publish(desired_positions)
            self._pub_current_pos.publish(current_positions)
            self._pub_current_vel.publish(current_velocities)
            is_close = np.isclose(desired_positions, current_positions)
            if np.nonzero(is_close) == 6:
                self._index += 1
                if self._index >= len(self._watch_points):
                    self._index = 0
                    self._watch_points = None
        else:
            pass
        if self._target_information.quotient > 0:
            pass

    def interruption(self):
        if self._target_information.quotient > 0:
            self._seen = True
            return True
        if self._watch_points is None:
            self._seen = False
            return True
        return False

    @property
    def was_seen(self):
        return self._seen

    def approching(self):
        pass


class Planner():
    def __init__(self):
        self._meeting_planner = MeetingPointPlanner()
        self._watch_planner = WatchPointPlanner()
        self._move_planner = MovePlanner()
        self._state_enum_class = namedtuple("EnumState", ["meeting_point", "watch_point", "moving"])
        self._state_enum = self._state_enum_class(1, 2, 3)
        self._state = self._state_enum.meeting_point
        self._meeting_point = None
        self._watch_points = None
        self._probability_map = None

    def step(self):
        while not rospy.is_shutdown():
            try:
                trans = self.tfBuffer.lookup_transform("map", 'bebop2/base_link', rospy.Time())
                self.save_change_state(trans)
                # do some stuffs
            except:
                interrupted = True

    def save_change_state(self, drone_position):
        def change_state_zero(interrupt, true_value, false_value):
            if interrupt:
                self._state = true_value
                return True
            else:
                self._state = false_value
                return False

        def change_state_one(interrupt):
            return change_state_zero(interrupt, self._state_enum.meeting_point, self._state_enum.watch_point)

        def change_state_two(interrupt):
            return change_state_zero(interrupt, self._state_enum.meeting_point, self._state)

        def change_state_three(interrupt):
            return change_state_zero(interrupt, self._state_enum.meeting_point, self._state_enum.moving)

        def change_state_four(interrupt):
            return change_state_zero(interrupt, self._state_enum.watch_point, self._state)

        if self._state == self._state_enum.meeting_point:
            self._meeting_point, self._probability_map = self._meeting_planner.next(drone_position)
            change_state_one(self._meeting_planner.interruption())
        elif self._state == self._state_enum.watch_point:
            if change_state_two(self._meeting_planner.interruption()):
                pass
            else:
                self._watch_points = self._watch_planner.next(drone_position, self._meeting_point,
                                                              self._probability_map)
                change_state_three(self._watch_planner.interruption())
        else:
            if change_state_two(self._meeting_planner.interruption()):
                pass
            elif change_state_four(self._watch_planner.interruption()):
                pass
            else:
                self._move_planner.next(self._watch_points)
                change_state_four(self._move_planner.interruption())
