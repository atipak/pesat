#!/usr/bin/env python
import rospy
import copy
import json
from collections import deque, namedtuple
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose, Point, TwistWithCovariance, TransformStamped, PoseStamped
from std_msgs.msg import Float32, Bool, Float64, Header, UInt32, Int32
import sys
import helper_pkg.utils as utils
from helper_pkg.utils import Constants
from pesat_msgs.msg import JointStates
from pesat_msgs.srv import PositionRequest


class CenralSystem(object):
    def __init__(self):
        rospy.init_node('central_system', anonymous=False)
        logic_configuration = rospy.get_param("logic_configuration")
        drone_configuration = rospy.get_param("drone_configuration")
        environment_configuration = rospy.get_param("environment_configuration")
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self._collision_probability = 0.0
        self._recommended_altitude = -1
        self._low_collision_limit = logic_configuration["avoiding"]["low_bound"]
        self._current_velocities = None
        enum = namedtuple("State", ["sat", "avoiding"])
        self._states = enum(0, 1)
        self._state = self._states.sat
        self._emergency_state = False
        self._map_frame = environment_configuration["map"]["frame"]
        self._base_link_frame = drone_configuration["properties"]["base_link"]
        self._camera_base_link_frame = drone_configuration["camera"]["base_link"]
        self._max_speed = drone_configuration["properties"]["max_speed"]
        self._last_state = self._state
        self._distance_limit = 0.01
        self._rotation_limit = np.pi / 360
        self._drone_size = drone_configuration["properties"]["size"]
        self._map_file = environment_configuration["map"]["obstacles_file"]
        self._last_drone_state = None
        self._plan = []
        self._plan_index = 0
        self._last_pose = None
        self.map = utils.Map.get_instance(self._map_file)

        rospy.Subscriber(logic_configuration["avoiding"]["recommended_altitude_topic"], Float32,
                         self.callback_recommended_altitude)
        rospy.Subscriber(logic_configuration["avoiding"]["collision_topic"], Float32, self.callback_da_collision)

        self._pub_cmd_vel = rospy.Publisher(drone_configuration["control"]["cmd_vel_topic"], TwistWithCovariance,
                                            queue_size=10)
        self._pub_desired_pos = rospy.Publisher(drone_configuration["position_pid"]["desired_pos_topic"], JointStates,
                                                queue_size=10)
        self._pub_current_pos = rospy.Publisher(drone_configuration["position_pid"]["current_pos_topic"], JointStates,
                                                queue_size=10)
        self._pub_current_vel = rospy.Publisher(drone_configuration["position_pid"]["current_vel_topic"], JointStates,
                                                queue_size=10)

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
        self._tracker_service_name = logic_configuration["tracking"]["position_in_time_service"]
        self._da_service_name = logic_configuration["avoiding"]["position_in_time_service"]
        self._tracker_next_point_srv = None
        self._da_next_point_srv = None
        _ = rospy.Service(drone_configuration["localization"]["position_in_time_service"], PositionRequest,
                          self.callback_position_in_time_service)

        # test settings
        self._tunning_state = 0

    # callbacks
    def callback_da_collision(self, data):
        self._collision_probability = data.data

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

    def callback_recommended_altitude(self, data):
        self._recommended_altitude = data.data

    def callback_position_in_time_service(self, data):
        service = self.safe_right_service_acquisition()
        response = self.get_drone_position_in_time(service, data.header.stamp.to_sec(), data.center, data.refresh)
        return [response]

    def safe_sat_service_acquisition(self):
        if self._tracker_next_point_srv is None:
            try:
                self._tracker_next_point_srv = rospy.ServiceProxy(self._tracker_service_name, PositionRequest)
            except rospy.ServiceException as e:
                self._tracker_next_point_srv = None
                rospy.loginfo_once("Service call failed: " + str(e))
        return self._tracker_next_point_srv

    def safe_dynamic_avoidance_service_acquisition(self):
        if self._da_next_point_srv is None:
            try:
                self._da_next_point_srv = rospy.ServiceProxy(self._da_service_name, PositionRequest)
            except rospy.ServiceException as e:
                self._da_next_point_srv = None
                rospy.loginfo_once("Service call failed: " + str(e))
        return self._da_next_point_srv

    def safe_right_service_acquisition(self):
        if self._state == self._states.sat:
            return self.safe_sat_service_acquisition()
        elif self._state == self._states.avoiding:
            return self.safe_dynamic_avoidance_service_acquisition()
        else:
            return None

    def fly_to_next_point(self, next_pose, drone_position):
        if drone_position is not None and next_pose is not None:
            # current position
            x_now = drone_position.position.x
            y_now = drone_position.position.y
            z_now = drone_position.position.z
            yaw_now = drone_position.orientation.z
            camera_horientation_now = drone_position.orientation.x
            camera_vorientation_now = drone_position.orientation.y
            # goal position
            x_next = next_pose.position.x
            y_next = next_pose.position.y
            z_next = next_pose.position.z
            yaw_next = next_pose.orientation.z
            camera_horientation_next = next_pose.orientation.x
            camera_vorientation_next = next_pose.orientation.y
            current_positions = [x_now, y_now, z_now, camera_horientation_now, camera_vorientation_now, yaw_now]
            desired_positions = [x_next, y_next, z_next, camera_horientation_next, camera_vorientation_next, yaw_next]
            v = self.ask_for_vel(current_positions, desired_positions)
            if v[5] > 0.1:
                vel = self.get_zero_velocity()
                vel[5] = v[5]
            else:
                vel = v
        else:
            vel = self.get_zero_velocity()
        self.publish_cmd_velocity(vel)
        print("central system, vel", vel)
        # print(current_positions)

    def check_position_for_obstacles(self, pose):
        if pose.position.z < self._recommended_altitude - 0.05:
            # print(pose, self._recommended_altitude)
            rospy.loginfo("Altitude of drone was changed into recommended one.")
            pose.position.z = self._recommended_altitude
        map_point = self.map.map_point_from_real_coordinates(pose.position.x, pose.position.y, pose.position.z)
        if self.map.is_free_for_drone(map_point):
            return pose
        else:
            rospy.loginfo("Next position got from prediction system isn't free.")
            return None

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
        goal_state_msg = PositionRequest()
        goal_state_msg.pose_stamped = PoseStamped()
        return goal_state_msg

    def is_emergency(self):
        if self._emergency_state:
            return True
        return False

    def execute_emergency_actions(self):
        self.publish_cmd_velocity(self.get_zero_velocity())
        last_drone_position = self._last_drone_state
        current_drone_position = self.get_current_drone_state()
        if abs(
                last_drone_position.position.x - current_drone_position.position.x) < 0.05 and abs(
            last_drone_position.position.y - current_drone_position.position.y) < 0.05 and abs(
            last_drone_position.position.z - current_drone_position.position.z) < 0.05 and abs(
            last_drone_position.orientation.z - current_drone_position.orientation.z) < 0.05:
            self._emergency_state = False

    def get_zero_velocity(self):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def change_state(self):
        if self._collision_probability > self._low_collision_limit and self._state != self._states.avoiding:
            self._last_state = self._state
            self._state = self._states.avoiding
            self._emergency_state = True
        elif self._collision_probability < self._low_collision_limit and self._state == self._states.avoiding:
            self._state = self._states.sat

    def get_current_drone_state(self):
        try:
            drone_position = self._tfBuffer.lookup_transform(self._map_frame, self._base_link_frame, rospy.Time())
            camera_position = self._tfBuffer.lookup_transform(self._base_link_frame, self._camera_base_link_frame,
                                                              rospy.Time())
            explicit_quat = [drone_position.transform.rotation.x, drone_position.transform.rotation.y,
                             drone_position.transform.rotation.z, drone_position.transform.rotation.w]
            (_, _, yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
            explicit_quat = [camera_position.transform.rotation.x, camera_position.transform.rotation.y,
                             camera_position.transform.rotation.z, camera_position.transform.rotation.w]
            (_, camera_vertical, camera_horizontal) = tf.transformations.euler_from_quaternion(explicit_quat)
            pose = Pose()
            pose.position.x = drone_position.transform.translation.x
            pose.position.y = drone_position.transform.translation.y
            pose.position.z = drone_position.transform.translation.z
            pose.orientation.x = camera_horizontal
            pose.orientation.y = camera_vertical
            pose.orientation.z = yaw
            self._last_drone_state = drone_position
            return pose
        except Exception as e:
            print("Exception in getting drone state:", e)
            return None

    def are_poses_similiar(self, pose_original, pose_new):
        if pose_original is None:
            return False
        if abs(pose_original.position.x - pose_new.position.x) < 0.2 and \
                abs(pose_original.position.y - pose_new.position.y) < 0.2 and \
                abs(pose_original.position.z - pose_new.position.z) < 0.2 and \
                abs(pose_original.orientation.z - pose_new.orientation.z) < 0.2:
            return True
        return False

    def loop(self):
        # print("loop")
        if self.is_emergency():
            self.execute_emergency_actions()
            return
        self.change_state()
        time = rospy.Time.now().to_sec() + 0.5
        response = self.real_case(time)
        # response = self.avoiding_test(time)
        # response = self.pid_tunning()
        # print("response", response)
        if response is None:
            self.publish_cmd_velocity(self.get_zero_velocity())
            return
        else:
            current = self.get_current_drone_state()
            if not self.are_poses_similiar(self._last_pose, response):
                if current is not None:
                    self._plan = self.map.drone_plan(current, response)
                    self._plan_index = 0
                else:
                    return
            pose = self.next_plan_point(response)
            pose = self.check_position_for_obstacles(pose)
            print("Central system, pose", pose)
            self.fly_to_next_point(pose, current)
            self._last_pose = response

    def next_plan_point(self, pose):
        current_pose = self.get_current_drone_state()
        if current_pose is None:
            return None
        if len(self._plan) > self._plan_index and self.are_poses_similiar(self._plan[self._plan_index], current_pose):
            self._plan_index += 1
        if len(self._plan) == self._plan_index:
            return current_pose
        self._plan[self._plan_index].orientation.x = pose.orientation.x
        self._plan[self._plan_index].orientation.y = pose.orientation.y
        return self._plan[self._plan_index]

    def get_drone_position_in_time(self, service, time, center, refresh):
        header = Header()
        header.stamp = rospy.Time.from_sec(time)
        try:
            response = service.call(header, -1, refresh, center)
            # print(response)
            if response.pointcloud is not None:
                return response.pointcloud
        except rospy.ServiceException as exc:
            rospy.logwarn("Service did not process request: " + str(exc))
        return None

    def cast_to_float64(self, fl):
        f = Float64()
        f.data = float(fl)
        return f

    def ask_for_vel(self, current_positions, desired_positions):
        # forward
        self._pub_forward_target_position.publish(self.cast_to_float64(desired_positions[0]))
        self._pub_forward_position.publish(self.cast_to_float64(current_positions[0]))
        # left
        self._pub_left_target_position.publish(self.cast_to_float64(desired_positions[1]))
        self._pub_left_position.publish(self.cast_to_float64(current_positions[1]))
        # up
        self._pub_up_target_position.publish(self.cast_to_float64(desired_positions[2]))
        self._pub_up_position.publish(self.cast_to_float64(current_positions[2]))
        # horizontal camera
        self._pub_hcamera_target_position.publish(self.cast_to_float64(desired_positions[3]))
        self._pub_hcamera_position.publish(self.cast_to_float64(current_positions[3]))
        # vertical camera
        self._pub_vcamera_target_position.publish(self.cast_to_float64(desired_positions[4]))
        self._pub_vcamera_position.publish(self.cast_to_float64(current_positions[4]))
        # yaw
        self._pub_yaw_target_position.publish(self.cast_to_float64(desired_positions[5]))
        self._pub_yaw_position.publish(self.cast_to_float64(current_positions[5]))
        # print("forward speed:", self._forward_speed)
        # print("left speed:", self._left_speed)
        # print("up speed:", self._up_speed)
        return [self._forward_speed / self._max_speed,
                self._left_speed / self._max_speed,
                self._up_speed / self._max_speed,
                self._hcamera_speed / self._max_speed,
                self._vcamera_speed / self._max_speed,
                self._yaw_speed / self._max_speed]

    def real_case(self, time):
        service = self.safe_right_service_acquisition()
        response = self.get_drone_position_in_time(service, time, None, False)
        if response is not None:
            response = utils.DataStructures.array_to_pose(utils.DataStructures.pointcloud2_to_array(response))
        return response

    # tests
    def test(self):
        desired_positions = [5.0, 5.0, 5.0, 5.0, 5.0, -5.0]
        drone_state = self.get_current_drone_state()
        if drone_state is None:
            return
        current_positions = [drone_state.position.x,
                             drone_state.position.y,
                             drone_state.position.z,
                             drone_state.orientation.x,
                             drone_state.orientation.y,
                             drone_state.orientation.z]
        # current_positions = [1.16446, 0.158, 545.0, 4.0, 5.0, -3.980]
        # current_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        velocity = self.ask_for_vel(current_positions, desired_positions)
        self.publish_cmd_velocity(velocity)

    def pid_tunning(self):
        drone_state = self.get_current_drone_state()
        if drone_state is not None:
            if self._tunning_state == 0 and abs(drone_state.position.z - 0.5) < 0.05:
                self._tunning_state = 1
            elif self._tunning_state == 1 and abs(drone_state.position.z - 1) < 0.05:
                self._tunning_state = 2
            elif self._tunning_state == 2 and abs(drone_state.position.x - 1) < 0.05:
                self._tunning_state = 0
            pr = PositionRequest()
            next_pose = Pose()
            next_pose.position.y = drone_state.position.y
            if self._tunning_state == 0:
                next_pose.position.z = 0.5
                next_pose.position.x = 0
            elif self._tunning_state == 1:
                next_pose.position.z = 1
                next_pose.position.x = 0
            else:
                next_pose.position.z = 1
                next_pose.position.x = 1
            next_pose.orientation.x = drone_state.orientation.x
            next_pose.orientation.y = drone_state.orientation.y
            next_pose.orientation.z = drone_state.orientation.z
            # print(next_pose)
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = self._map_frame
            pose_stamped.pose = next_pose
            pr.pose_stamped = pose_stamped
            return pr
        return None

    def avoiding_test(self, time):
        if self._state == self._states.avoiding:
            service = self.safe_right_service_acquisition()
            response = self.get_drone_position_in_time(service, time, None, False)
            response = utils.DataStructures.array_to_pose(utils.DataStructures.pointcloud2_to_array(response))
            return response
        else:
            return self.forward_direction()

    def forward_direction(self):
        drone_state = self.get_current_drone_state()
        if drone_state is not None:
            # print("pr")
            pr = PositionRequest()
            next_pose = Pose()
            next_pose.position.x = drone_state.position.x
            next_pose.position.y = drone_state.position.y
            next_pose.position.z = drone_state.position.z
            if self._recommended_altitude > 0.5:
                next_pose.position.z = self._recommended_altitude
            else:
                next_pose.position.z = 0.5
            next_pose.position.x = drone_state.position.x + 0.3
            next_pose.orientation.x = drone_state.orientation.x
            next_pose.orientation.y = drone_state.orientation.y
            next_pose.orientation.z = drone_state.orientation.z
            # print(next_pose)
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = self._map_frame
            pose_stamped.pose = next_pose
            pr.pose_stamped = pose_stamped
            return pr
        return None


if __name__ == '__main__':
    central_system = CenralSystem()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        central_system.loop()
        rate.sleep()
