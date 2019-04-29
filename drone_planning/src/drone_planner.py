#!/usr/bin/env python
import rospy
import copy
import sys
import json
from collections import deque, namedtuple
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import TwistWithCovariance, TransformStamped, Pose, Vector3
from std_msgs.msg import Bool, Float32
from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
import moveit_commander
from pesat_msgs.msg import Notification, ImageTargetInfo, JointStates, CameraShot, CameraUpdate
from pesat_msgs.srv import PredictionMaps, PositionPrediction, CameraRegistration, GoalState
import helper_pkg.utils as utils

noti = namedtuple("Notification", ["serial_number", "time", "likelihood", "x", "y", "dx", "dy"])
watch_point = namedtuple("WatchPoint", ["position", "yaw", "vertical_camera_turn", "delay"])
meet_point = namedtuple("MeetingPoint", ["center", "axis", "rotation", "time", "halfspace"])


class Database(object):
    def __init__(self, maximum_length, maximal_time):
        self._notifications = {}
        self._last_serial_number = {}
        self._last_notification = 0
        self._earlest_notification = 0
        self._maximum_length = maximum_length
        self._maximal_time = maximal_time  # secs
        _cctv_system_config = rospy.get_param("cctv")
        rospy.Subscriber(_cctv_system_config["camera_notification_topic"], Notification, self.callback_notify)

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
        self._params = rospy.get_param("drone_planner")["meet_point_planner"]
        prediction_service_name = self._params["prediction_service_name"]
        try:
            self._srv_prediction = rospy.ServiceProxy(prediction_service_name, PredictionMaps)
        except rospy.ServiceException as e:
            print("Service call failed: %s", e)
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self._database = Database(self._params["maximum_length"], self._params["maximal_time"])
        self._meeting_point = None
        self._interruption = False

    def next(self, drone_position):
        if drone_position is None:
            self._interruption = True
            return None, None
        if not self.check_drone_condition():
            # get
            pass
        meeting_point = None
        probability_map = None
        return meeting_point, probability_map

    def interruption(self, done, found):
        if not self.check_drone_condition() or done or found:
            return True
        if self._interruption:
            self._interruption = False
            return True
        return False

    def check_drone_condition(self):
        # TODO fill condition
        # accumulator, drone error, crash
        return True

    def reset(self, found):
        if found:
            # clear everything
            self._meeting_point = None
        else:
            pass


class WatchPointPlanner(object):
    def __init__(self):
        _params = rospy.get_param("drone_planner")["planner"]
        _drone_configuration = rospy.get_param("drone_bebop2")["drone"]
        # moveit
        moveit_commander.roscpp_initialize(sys.argv)
        self._robot = moveit_commander.RobotCommander()
        self._scene = moveit_commander.PlanningSceneInterface()
        group_name = _params["group_name"]
        namespace = _params["namespace"]
        self._jump_threshold = _params["jump_threshold"]
        self._eef_step = _params["eef_step"]
        self._move_group = moveit_commander.MoveGroupCommander(group_name, ns=namespace)
        self._params = rospy.get_param("drone_planner")["watch_point_planner"]
        # topic names
        drone_place_planner_srv_name = self._params["drone_place_planner_srv_name"]
        # services
        self._srv_position = rospy.ServiceProxy(drone_place_planner_srv_name, PositionPrediction)
        # topics
        # constants
        self._drone_speed = self._drone_configuration["max_speed"]
        # variables
        self._watch_points = None
        self._meeting_point = None
        self._probability_map = None
        self._last_index = -1
        self._last_distance = 0
        self._map = None
        self._done = False
        self._limit_time_exhausted = False
        self._interruption = False

    @property
    def map(self):
        return self._map

    @property
    def meeting_point_delay(self):
        return self._limit_time_exhausted

    def next(self, drone_position, meeting_point, probability_map):
        if drone_position is None or meeting_point is None or probability_map is None:
            self._interruption = True
            return None
        self._meeting_point = meeting_point
        self._probability_map = probability_map
        self._watch_points = []
        return self._watch_points

    def interruption(self, index, found, done):
        if self._interruption:
            self._interruption = False
            return True
        if done or found or self._limit_time_exhausted or index >= len(self._watch_points):
            return True
        if index != self._last_index:
            self._last_distance = np.sqrt(
                np.square(self._meeting_point.x - self._watch_points[index].position.x) + np.square(
                    self._meeting_point.y - self._watch_points[index].position.y) + np.square(
                    self._meeting_point.z - self._watch_points[index].position.z))
        # can't meet the deadline
        if self._last_distance / self._drone_speed + rospy.Time.now() > self._meeting_point.time:
            self._limit_time_exhausted = True
            return True
        return False

    def reset(self):
        self._watch_points = None
        self._meeting_point = None
        self._probability_map = None
        self._last_index = -1
        self._last_distance = 0
        self._map = None
        self._done = False

    def to_goal_position(self, goal_pose):
        self._move_group.set_pose_target(goal_pose)
        plan = self._move_group.plan()
        self._move_group.clear_pose_targets()
        return plan


class MovePlanner(object):
    def __init__(self, watch_point_planner):
        # parameters
        _params = rospy.get_param("drone_planner")["move_planner"]
        _drone_configuration = rospy.get_param("drone_bebop2")["drone"]
        _camera_configuration = rospy.get_param("drone_bebop2")["camera"]
        _target_configuration = rospy.get_param("target")
        _environment_configuration = rospy.get_param("environment")
        _da_configuration = rospy.get_param("dynamic_avoidance")
        _pid_configuration = rospy.get_param("position_pid")
        _cctv_configuration = rospy.get_param("cctv")
        _target_localization_configuration = rospy.get_param("target_localization")
        # subscribers
        rospy.Subscriber(_params["target_information"], ImageTargetInfo, self.callback_target_information)

        # publishers
        self._pub_camera_shot = rospy.Publisher(_cctv_configuration["camera_shot_topic"], CameraShot, queue_size=10)
        self._pub_camera_update = rospy.Publisher(_cctv_configuration["camera_update_topic"], CameraUpdate,
                                                  queue_size=10)

        # services
        rospy.wait_for_service(_cctv_configuration["camera_registration_service"])
        try:
            cam_registration = rospy.ServiceProxy(_cctv_configuration["camera_registration_service"],
                                                  CameraRegistration)
            self._camera_id = cam_registration(Vector3(0, 0, 0), 1, 1, 1, 0, 1)
        except rospy.ServiceException as e:
            print("Service call failed: %s", e)
        # constants
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        # TODO choose right limit for antiaproaching
        self._max_time_from_last_seen_approaching = _params["max_time_from_last_seen_approaching"]  # seconds
        self._maximal_approaching_distance = _params["maximal_approaching_distance"]  # meters
        self._max_height = _params["max_height"]
        self._min_height = _params["min_height"]
        self._hfov = _camera_configuration["hfov"]
        self._image_height = _camera_configuration["image_height"]
        self._image_width = _camera_configuration["image_width"]
        self._focal_length = self._image_width / (2.0 * np.tan(self._hfov / 2.0))
        self._max_vertical_angle = np.arctan2(self._image_height / 2, self._focal_length) #vfov
        self._watch_point_planner = watch_point_planner
        self._map_frame = _environment_configuration["map"]
        self._target_location_frame = _target_localization_configuration["location_frame"]
        # variables
        self._target_information = None
        self._seen = False
        self._index = 0
        self._watch_points = None
        self._approaching_result = False
        self._is_approaching = False
        self._found = False
        self._approaching_trajectory_save = None
        self._approaching_index_save = 0
        self._moving_start = rospy.Time.now()
        self._time_limit_exhausted = False
        self._reset = False
        self._interruption = False

    # callbacks
    def callback_target_information(self, data):
        self._target_information = data

    def next(self, watch_points):
        if watch_point is None:
            self._interruption = True
            return False
        self._watch_points = watch_points
        self._target_information = None
        self._seen = False
        self._index = 0
        self._watch_points = None
        self._approaching_result = False
        self._is_approaching = False
        self._found = False
        self._approaching_trajectory_save = None
        self._approaching_index_save = 0
        self._moving_start = rospy.Time.now()
        self._time_limit_exhausted = False
        self._reset = False
        return True

    def interruption(self):
        if self._interruption:
            self._interruption = False
            return True
        if self._time_limit_exhausted or self._approaching_result or self._watch_points is None:
            return True
        return False

    def next_point(self, drone_position, close_enough):
        if rospy.Time.now() - self._moving_start > self._watch_points[self._index].delay:
            self._time_limit_exhausted = True
            return None
            # move drone by given points
        if self._watch_points is not None:
            if close_enough:
                self._index += 1
                if self._index >= len(self._watch_points):
                    self.reset()
                    return None
            return self._watch_points[self._index]
        else:
            # two branches
            # drone is in state approaching
            if self._is_approaching:
                try:
                    trans = self._tfBuffer.lookup_transform(self._map_frame, self._target_location_frame,
                                                            rospy.Time())
                    if trans.header.stamp < rospy.Time.now() - self._max_time_from_last_seen_approaching:
                        self.antiapproaching()
                    else:
                        target = trans.transform.translation
                        drone = drone_position.transform.translation
                        distance = self.vector_distance(target, drone)
                        if distance < self._maximal_approaching_distance:
                            self._approaching_result = True
                        else:
                            self._approaching_result = False
                            self.approaching(reapproaching=True)
                except Exception as e:
                    print(e)
            # drone is in classical state
            else:
                self._found = False
        if self._target_information is not None and self._target_information.quotient > 0:
            if not self._is_approaching:
                self.approaching()
            if not self._is_approaching and self._approaching_result:
                self._found = True
        return None

    def find_closest_watch_point(self, drone_position):
        i = self._index
        last_distance = self.vector_distance_3d(drone_position.transform.translation, self._watch_points[i].position)
        while i < len(self._watch_points):
            distance = self.vector_distance_3d(drone_position.transform.translation, self._watch_points[i].position)
            if distance <= last_distance:
                last_distance = distance
            else:
                break
            i += 1
        if i == len(self._watch_points):
            return -1
        return i

    def reset(self):
        self._reset = True

    @property
    def found(self):
        return self._found

    @property
    def done(self):
        return self._reset

    @property
    def watch_point_delay(self):
        return self._time_limit_exhausted

    @property
    def current_index(self):
        if self._is_approaching:
            return self._approaching_index_save
        else:
            return self._index

    def approaching(self, reapproaching=False):
        try:
            # get position of target
            trans = self._tfBuffer.lookup_transform(self._map_frame, self._target_location_frame, rospy.Time())
            pose = Pose()
            rounded_x, rounded_y = utils.rounding(trans.transform.translation.x), utils.rounding(
                trans.transform.translation.y)
            # free place for drone around the position of target
            maximal_distance = 3
            minimal_distance = 1
            pose_x, pose_y = self._watch_point_planner.map.free_place_in_field(rounded_x, rounded_y,
                                                                               minimal_distance * 2,
                                                                               maximal_distance * 2)
            if pose_x is None:
                raise Exception(
                    "A free place around target position wasn't found. \n X: {}, Y: {}, max. distance:{}".format(
                        rounded_x, rounded_y, maximal_distance))
            pose.position.y = pose_y
            pose.position.x = pose_x
            pose.position.z = 7
            pose.orientation.w = 1
            # get plan
            plan = self._watch_point_planner.to_goal_position(pose)
            if len(plan) < 2:
                raise Exception("No suitable plan found.")
            last_position = plan[len(plan) - 1]
            (_, _, yaw_drone) = tf.transformations.euler_from_quaternion(last_position.orientation)
            # make rotation with drone around z-axis to scan neighbourhood
            for i in range(4):
                yaw_drone += np.pi / 2
                plan.trajectory.append(
                    self.create_new_trajectory_pose(last_position.position.x, last_position.position.y,
                                                    last_position.position.z, yaw_drone))
            # change trajectories
            if self._approaching_trajectory_save is not None and not reapproaching:
                raise Exception(
                    "Cannot call an approaching method if drone has already tried to approach the target.\
                     Call this method with reapproaching set on True instead")
            elif reapproaching:
                self._watch_points = plan
                self._index = 0
            else:
                self._approaching_trajectory_save = self._watch_points
                self._approaching_index_save = self._index
                self._watch_points = plan
                self._index = 0
            self._is_approaching = True
        except Exception as e:
            print("Exception was thrown: {}".format(e))

    @staticmethod
    def transform_plan(plan):
        # TODO make from plan trajectory = watch_points
        pass

    @staticmethod
    def create_new_trajectory_pose(x, y, z, yaw):
        p = Pose()
        p.position.x = x
        p.position.y = y
        p.position.z = z
        p.orientation = tf.transformations.quaternion_from_euler(0, 0, yaw)
        return p

    def antiapproaching(self):
        if self._approaching_trajectory_save is None:
            raise Exception()
        try:
            # get position of target
            trans = self._tfBuffer.lookup_transform(self._map_frame, self._target_location_frame, rospy.Time())
            update = CameraUpdate()
            update.center.x = 0
            update.center.y = 0
            update.end.x = 0
            update.end.y = 0
            update.start.x = 0
            update.start.y = 0
            update.default.x = trans.transform.translation.x
            update.default.y = trans.transform.translation.y
            update.radius = 0
            update.mi = 1
            update.sigma = 0
            update.camera_id = self._camera_id
            self._pub_camera_update.publish(update)
            trans_older = self._tfBuffer.lookup_transform(self._map_frame, self._target_location_frame,
                                                          rospy.Time() - 0.2)
            velocity_x = trans.transform.translation.x - trans_older.transform.translation.x
            velocity_y = trans.transform.translation.y - trans_older.transform.translation.y
            shot = CameraShot()
            shot.camera_id = self._camera_id
            shot.velocity_x = velocity_x
            shot.velocity_y = velocity_y
            self._pub_camera_shot.publish(shot)
        except Exception as e:
            print(e)
        self._watch_points = self._approaching_trajectory_save
        self._index = self._approaching_index_save
        self._is_approaching = False

    @staticmethod
    def vector_distance_3d(v1, v2):
        return np.sqrt(np.square(v1.x - v2.x) + np.square(v1.y - v2.y) + np.square(v1.z - v2.z))

    @staticmethod
    def vector_distance(v1, v2):
        return np.sqrt(np.square(v1.x - v2.x) + np.square(v1.y - v2.y))


class Planner():
    def __init__(self):
        rospy.init_node('planner', anonymous=False)
        _drone_configuration = rospy.get_param("drone_bebop2")["drone"]
        _environment_configuration = rospy.get_param("environment")
        _params = rospy.get_param("drone_planner")["planner"]
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self._meeting_planner = MeetingPointPlanner()
        self._watch_planner = WatchPointPlanner()
        self._move_planner = MovePlanner(self._watch_planner)
        self._state_enum_class = namedtuple("EnumState", ["meeting_point", "watch_point", "moving"])
        self._state_enum = self._state_enum_class(1, 2, 3)
        self._state = self._state_enum.meeting_point
        self._meeting_point = None
        self._watch_points = None
        self._probability_map = None
        self._map_frame = _environment_configuration["map"]
        self._base_link_frame = _drone_configuration["base_link"]
        self._s = rospy.Service(_params["next_position_service"], GoalState, self.next_position_callback)
        rospy.Subscriber(_params["launch_topic"], Bool, self.init_reset_callback)
        self._launched = False

    # callbacks
    def next_position_callback(self, data):
        goal_state_msg = GoalState()
        pose = Pose()
        point = self._move_planner.next_point(data.drone_position, data.last_point_reached)
        if point is not None:
            pose.position.x = point.position.x
            pose.position.y = point.position.y
            pose.position.z = point.position.z
            pose.orientation.z = point.yaw
            pose.orientation.y = point.vertical_camera_turn
            pose.orientation.x = 0.0
            goal_state_msg.goal_pose = pose
        else:
            goal_state_msg.goal_pose = None
        goal_state_msg.change_state = self.found
        return goal_state_msg

    def init_reset_callback(self, data):
        if not self._launched and data:
            self.start()
        elif not data and self._launched:
            self.stop()

    @property
    def found(self):
        return self._move_planner.found

    def step(self):
        try:
            trans = self._tfBuffer.lookup_transform(self._map_frame, self._base_link_frame, rospy.Time())
            self.save_change_state(trans)
        except Exception as e:
            print("Exception:", e)
            self.reset()

    def start(self):
        self._launched = True

    def stop(self):
        self._launched = False
        self.reset()

    def reset(self):
        self._meeting_planner.reset()
        self._watch_planner.reset()
        self._move_planner.reset()

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

        # state meeting planner
        if self._state == self._state_enum.meeting_point:
            self._watch_planner.reset()
            self._move_planner.reset()
            self._meeting_point, self._probability_map = self._meeting_planner.next(drone_position)
            change_state_one(self._meeting_planner.interruption(self._move_planner.done, self._move_planner.found))
        # state watch planner
        elif self._state == self._state_enum.watch_point:
            # back to state M
            if change_state_two(self._meeting_planner.interruption(self._move_planner.done, self._move_planner.found)):
                self._watch_planner.reset()
                self._move_planner.reset()
            # to state W
            else:
                self._move_planner.reset()
                self._watch_points = self._watch_planner.next(drone_position, self._meeting_point,
                                                              self._probability_map)
                change_state_three(
                    self._watch_planner.interruption(self._move_planner.current_index, self._move_planner.found,
                                                     self._move_planner.done))
        # state move planner
        else:
            # back to state M
            if change_state_two(self._meeting_planner.interruption(self._move_planner.done, self._move_planner.found)):
                self._watch_planner.reset()
                self._move_planner.reset()
            # back to state W
            elif change_state_four(
                    self._watch_planner.interruption(self._move_planner.current_index, self._move_planner.found,
                                                     self._move_planner.done)):
                self._move_planner.reset()
            # to state C
            else:
                self._move_planner.next(self._watch_points)
                change_state_four(self._move_planner.interruption())


if __name__ == '__main__':
    planner = Planner()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        planner.step()
        rate.sleep()