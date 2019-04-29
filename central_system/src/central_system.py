#!/usr/bin/env python
import rospy
import copy
import json
from collections import deque, namedtuple
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose, Point, TwistWithCovariance, TransformStamped
from std_msgs.msg import Float32, Bool, Float64
import sys
import helper_pkg.utils as utils
from pesat_msgs.msg import JointStates
from pesat_msgs.srv import GoalState


class CenralSystem(object):
    def __init__(self):
        rospy.init_node('central_system', anonymous=False)
        _da_configuration = rospy.get_param("dynamic_avoidance")
        _pid_configuration = rospy.get_param("position_pid")
        _drone_configuration = rospy.get_param("drone_bebop2")["drone"]
        _planner_config = rospy.get_param("drone_planner")["planner"]
        _tracker_config = rospy.get_param("track")
        _params = rospy.get_param("central_system")
        _environment_configuration = rospy.get_param("environment")
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self._is_avoiding = False
        self._reset = False
        self._collision_probability = 0.0
        self._recommended_altitude = -1
        self._low_collision_limit = _params["low_collision_limit"]
        self._current_velocities = None
        self._vel_update = True
        enum = namedtuple("State", ["searching", "tracking", "emergency", "avoiding"])
        self._states = enum(0, 1, 2, 3)
        self._state = self._states.searching
        self._go_to_next = False
        self._map_frame = _environment_configuration["map"]
        self._base_link_frame = _drone_configuration["base_link"]
        self._camera_base_link_frame = _drone_configuration["camera_base_link"]
        self._max_speed = _drone_configuration["max_speed"]
        self._last_state = self._state
        self._last_go_to_next = self._go_to_next

        rospy.Subscriber(_pid_configuration["update_velocities_topic"], JointStates, self.callback_update)
        rospy.Subscriber(_da_configuration["recommended_altitude_topic"], Float32, self.callback_recommended_altitude)
        rospy.Subscriber(_da_configuration["is_avoiding_topic"], Bool, self.callback_da_avoiding)
        rospy.Subscriber(_da_configuration["collision_topic"], Float32, self.callback_da_collision)

        self._pub_da_switch = rospy.Publisher(_da_configuration["switch_topic"], Bool, queue_size=10)
        self._pub_da_alert = rospy.Publisher(_da_configuration["alert_topic"], Bool, queue_size=10)
        self._pub_cmd_vel = rospy.Publisher(_drone_configuration["cmd_vel_topic"], TwistWithCovariance, queue_size=10)
        self._pub_desired_pos = rospy.Publisher(_pid_configuration["desired_pos_topic"], JointStates, queue_size=10)
        self._pub_current_pos = rospy.Publisher(_pid_configuration["current_pos_topic"], JointStates, queue_size=10)
        self._pub_current_vel = rospy.Publisher(_pid_configuration["current_vel_topic"], JointStates, queue_size=10)
        self._pub_launch_planner = rospy.Publisher(_planner_config["launch_topic"], Bool, queue_size=10)
        self._pub_launch_tracker = rospy.Publisher(_tracker_config["launch_topic"], Bool, queue_size=10)

        # pid controllers
        # forward
        topic_prefix = "forward"
        self._forward_speed = 0.0
        self._pub_forward_position = rospy.Publisher(topic_prefix + "/position", Float64, queue_size=10)
        self._pub_forward_target_position = rospy.Publisher(topic_prefix + "/target_position", Float64, queue_size=10)
        rospy.Subscriber(topic_prefix + "/speed", Float64, self.callback_forward_speed)
        # left
        topic_prefix = "left"
        self._left_speed = 0.0
        self._pub_left_position = rospy.Publisher(topic_prefix + "/position", Float64, queue_size=10)
        self._pub_left_target_position = rospy.Publisher(topic_prefix + "/target_position", Float64, queue_size=10)
        rospy.Subscriber(topic_prefix + "/speed", Float64, self.callback_left_speed)
        # up
        topic_prefix = "up"
        self._up_speed = 0.0
        self._pub_up_position = rospy.Publisher(topic_prefix + "/position", Float64, queue_size=10)
        self._pub_up_target_position = rospy.Publisher(topic_prefix + "/target_position", Float64, queue_size=10)
        rospy.Subscriber(topic_prefix + "/speed", Float64, self.callback_up_speed)
        # hcamera
        topic_prefix = "hcamera"
        self._hcamera_speed = 0.0
        self._pub_hcamera_position = rospy.Publisher(topic_prefix + "/position", Float64, queue_size=10)
        self._pub_hcamera_target_position = rospy.Publisher(topic_prefix + "/target_position", Float64, queue_size=10)
        rospy.Subscriber(topic_prefix + "/speed", Float64, self.callback_hcamera_speed)
        # vcamera
        topic_prefix = "vcamera"
        self._vcamera_speed = 0.0
        self._pub_vcamera_position = rospy.Publisher(topic_prefix + "/position", Float64, queue_size=10)
        self._pub_vcamera_target_position = rospy.Publisher(topic_prefix + "/target_position", Float64, queue_size=10)
        rospy.Subscriber(topic_prefix + "/speed", Float64, self.callback_vcamera_speed)
        # yaw
        topic_prefix = "yaw"
        self._yaw_speed = 0.0
        self._pub_yaw_position = rospy.Publisher(topic_prefix + "/position", Float64, queue_size=10)
        self._pub_yaw_target_position = rospy.Publisher(topic_prefix + "/target_position", Float64, queue_size=10)
        rospy.Subscriber(topic_prefix + "/speed", Float64, self.callback_yaw_speed)

        # services
        # rospy.wait_for_service(_planner_config["next_position_service"])
        # rospy.wait_for_service(_tracker_config["next_position_service"])
        # rospy.wait_for_service(_da_configuration["next_position_service"])
        try:
            # self._planner_next_point_srv = rospy.ServiceProxy(_planner_config["next_position_service"], GoalState)
            self._tracker_next_point_srv = rospy.ServiceProxy(_tracker_config["next_position_service"], GoalState)
            self._da_next_point_srv = rospy.ServiceProxy(_da_configuration["next_position_service"], GoalState)
        except rospy.ServiceException as e:
            print("Service call failed: %s", e)

        # init
        self._pub_da_alert.publish(True)

    # callbacks
    def callback_da_collision(self, data):
        self._collision_probability = data

    def callback_da_avoiding(self, data):
        self._is_avoiding = data

    def callback_up_speed(self, data):
        self._up_speed = data.data

    def callback_left_speed(self, data):
        self._left_speed = data.data

    def callback_forward_speed(self, data):
        self._forward_speed = data.data

    def callback_hcamera_speed(self, data):
        self._hcamera_speed = data.data

    def callback_vcamera_speed(self, data):
        self._vcamera_speed = data.data

    def callback_yaw_speed(self, data):
        self._yaw_speed = data.data

    def callback_update(self, data):
        print(data.values)
        if self._vel_update:
            pass
            # self.publish_cmd_velocity([x for x in data.values])

    def callback_recommended_altitude(self, data):
        self._recommended_altitude = data

    def make_move(self, next_point, drone_position):
        if self._reset:
            return
        # there is a obstacle in front of drone
        if (
                self._collision_probability > self._low_collision_limit or self._is_avoiding) and self._state != self._states.avoiding:
            self._pub_da_switch.publish(True)
            self._last_state = self._state
            self._last_go_to_next = self._go_to_next
            self._state = self._states.avoiding
            self._go_to_next = False
            return
        if next_point is not None:
            return self.fly_to_next_point(next_point, drone_position)
        else:
            return False

    def fly_to_next_point(self, next_pose, drone_position):
        # current position
        x_drone = drone_position.transform.translation.x
        y_drone = drone_position.transform.translation.y
        z_drone = drone_position.transform.translation.z
        yaw_drone = drone_position.transform.rotation.z
        horizontal_camera_turn = drone_position.transform.rotation.x
        vertical_camera_turn = drone_position.transform.rotation.y
        # goal position
        x_point = next_pose.position.x
        y_point = next_pose.position.y
        if self._recommended_altitude == -1:
            z_point = next_pose.position.z
        else:
            z_point = self._recommended_altitude
        yaw_point = next_pose.orientation.z
        horizontal_camera_turn_point = next_pose.position.x
        verical_camera_turn_point = next_pose.position.y
        current_positions = [x_drone, y_drone, z_drone, horizontal_camera_turn, vertical_camera_turn, yaw_drone]
        desired_positions = [x_point, y_point, z_point, horizontal_camera_turn_point,
                             verical_camera_turn_point, yaw_point]
        if self._current_velocities is not None:
            current_velocities = [self._current_velocities.twist.linear.x,
                                  self._current_velocities.twist.linear.y,
                                  self._current_velocities.twist.linear.z,
                                  self._current_velocities.twist.angular.x,
                                  self._current_velocities.twist.angular.y,
                                  self._current_velocities.twist.angular.z]
        else:
            current_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._pub_desired_pos.publish(desired_positions)
        self._pub_current_pos.publish(current_positions)
        self._pub_current_vel.publish(current_velocities)
        is_close = np.isclose(desired_positions, current_positions)
        if np.nonzero(is_close) == 6:
            return True
        else:
            return False

    def publish_cmd_velocity(self, vel_command):
        twist = TwistWithCovariance()
        twist.twist.linear.x = vel_command[0]
        twist.twist.linear.y = vel_command[1]
        twist.twist.linear.z = vel_command[2]
        twist.twist.angular.x = vel_command[3]
        twist.twist.angular.y = vel_command[4]
        twist.twist.angular.z = vel_command[5]
        self._pub_cmd_vel.publish(twist)

    def emergency_state(self):
        goal_state_msg = GoalState()
        goal_state_msg.change_state = False
        goal_state_msg.goal_pose = TransformStamped()
        return goal_state_msg

    def is_emergency(self):
        return False

    def reset(self):
        self._reset = True
        if self._state == self._states.avoiding:
            self._pub_da_switch.publish(False)
            self._state = self._last_state
            self._go_to_next = self._last_go_to_next
        self.publish_cmd_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def step(self):
        try:
            self._state = self._states.tracking  # WARNING set tracking
            drone_position = self._tfBuffer.lookup_transform(self._map_frame, self._base_link_frame, rospy.Time())
            camera_position = self._tfBuffer.lookup_transform(self._base_link_frame, self._camera_base_link_frame,
                                                              rospy.Time())
            (_, _, yaw) = tf.transformations.euler_from_quaternion(drone_position.transform.rotation)
            (_, camera_vertical, camera_horizontal) = tf.transformations.euler_from_quaternion(
                camera_position.transform.rotation)
            drone_position.transform.rotation.x = camera_horizontal
            drone_position.transform.rotation.y = camera_vertical
            drone_position.transform.rotation.z = yaw
            if self._state == self._states.avoiding:
                if self._is_avoiding:
                    goal_state_msg = self._da_next_point_srv(drone_position, self._go_to_next)
                else:
                    self._state = self._last_state
                    self._go_to_next = self._last_go_to_next
                    if self._recommended_altitude == -1:
                        self._pub_da_switch.publish(False)
                    return
            elif self._state == self._states.searching:
                goal_state_msg = self._planner_next_point_srv(drone_position, self._go_to_next)
            elif self._state == self._states.tracking:
                goal_state_msg = self._tracker_next_point_srv(drone_position, self._go_to_next)
            else:
                goal_state_msg = self.emergency_state()
            if self.is_emergency():
                self._state = self._states.emergency
                return
            pose = goal_state_msg.goal_pose
            change_state = goal_state_msg.change_state = False  # WARNING set FALSE
            if self._state == self._states.searching and change_state:
                self._state = self._states.tracking
                self._pub_launch_tracker.publish(True)
                self._pub_launch_planner.publish(False)
            elif self._state == self._states.tracking and change_state:
                self._state = self._states.searching
                self._pub_launch_tracker.publish(False)
                self._pub_launch_planner.publish(True)
            if change_state:
                return
            if pose is not None:
                self._go_to_next = self.make_move(pose, drone_position)
            else:
                self.publish_cmd_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


        except Exception as e:
            print("Exception:", e)

    def ask_for_vel(self, current_positions, desired_positions):
        # forward
        self._pub_forward_target_position.publish(desired_positions[0])
        self._pub_forward_position.publish(current_positions[0])
        # left
        self._pub_left_target_position.publish(desired_positions[1])
        self._pub_left_position.publish(current_positions[1])
        # up
        self._pub_up_target_position.publish(desired_positions[2])
        self._pub_up_position.publish(current_positions[2])
        # horizontal camera
        self._pub_hcamera_target_position.publish(desired_positions[3])
        self._pub_hcamera_position.publish(current_positions[3])
        # vertical camera
        self._pub_vcamera_target_position.publish(desired_positions[4])
        self._pub_vcamera_position.publish(current_positions[4])
        # yaw
        self._pub_yaw_target_position.publish(desired_positions[5])
        self._pub_yaw_position.publish(current_positions[5])
        return [self._forward_speed / self._max_speed,
                self._left_speed / self._max_speed,
                self._up_speed / self._max_speed,
                self._hcamera_speed / self._max_speed,
                self._vcamera_speed / self._max_speed,
                self._yaw_speed / self._max_speed]

    def test(self):
        desired_positions = [5.0, 5.0, 5.0, 5.0, 5.0, -5.0]
        current_positions = [1.16446, 0.158, 545.0, 4.0, 5.0, -3.980]
        current_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print(self.ask_for_vel(current_positions, desired_positions))
        self._pub_desired_pos.publish(desired_positions)
        self._pub_current_pos.publish(current_positions)
        self._pub_current_vel.publish(current_velocities)


if __name__ == '__main__':
    central_system = CenralSystem()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        central_system.test()
        rate.sleep()
