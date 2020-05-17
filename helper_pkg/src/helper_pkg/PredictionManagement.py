from abc import abstractmethod

import rospy
import re
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from pesat_msgs.msg import ImageTargetInfo, FloatImageStamped
from pesat_msgs.srv import PositionRequest, PointInTime
import helper_pkg.utils as utils
import tf as tf_ros
import tf2_ros
from helper_pkg.utils import Constants
from collections import deque
import traceback
from scipy.spatial import cKDTree
import cv2


class PredictionAlgorithm(object):
    DRONE_POSITIONS = 7
    TARGET_POSITIONS = 8
    INPUT_TYPE = Constants.InputOutputParameterType.map
    OUTPUT_TYPE = Constants.InputOutputParameterType.map
    MAP_PRESENCE = Constants.MapPresenceParameter.yes

    def __init__(self):
        super(PredictionAlgorithm, self).__init__()
        env_configuration = rospy.get_param("environment_configuration")
        self._map_frame = env_configuration["map"]["frame"]
        self._obstacles_file_path = env_configuration["map"]["obstacles_file"]
        self._prediction_algorithm_name = "Unknown algorithm"
        self._map_width = 680
        self._map_height = 680
        self._map_resolution = 0.5

    def get_algorithm_parameters(self):
        return [self.INPUT_TYPE,
                self.DRONE_POSITIONS,
                self.TARGET_POSITIONS,
                self.MAP_PRESENCE,
                self.OUTPUT_TYPE]

    def check_parameters(self, drone_positions, target_positions, map):
        # print("target positions", target_positions)
        # if self.INPUT_TYPE == Constants.InputOutputParameterType.pose:
        #    control_class = PoseStamped
        # elif self.INPUT_TYPE == Constants.InputOutputParameterType.pointcloud:
        #    control_class = PointCloud2
        # else:
        #    control_class = np.ndarray
        control_class = PointCloud2
        if self.TARGET_POSITIONS > 0:
            if target_positions is None or not isinstance(target_positions, list) or len(
                    target_positions) != self.TARGET_POSITIONS:
                raise AttributeError(self._prediction_algorithm_name + ": wrong target positions parameter.")
            for i in range(len(target_positions)):
                if not isinstance(target_positions[i], control_class):
                    raise AttributeError(
                        self._prediction_algorithm_name + ": target position with index {} isn't instance of class {}, but it is {}".format(
                            i, control_class, type(target_positions[i])))
        if self.DRONE_POSITIONS > 0:
            if drone_positions is None or not isinstance(drone_positions, list) or len(
                    drone_positions) != self.DRONE_POSITIONS:
                raise AttributeError(self._prediction_algorithm_name + ": wrong drone positions parameter.")
            for i in range(len(drone_positions)):
                if not isinstance(drone_positions[i], control_class):
                    raise AttributeError(
                        self._prediction_algorithm_name + ": drone position with index {} isn't instance of class {}, but it is {}".format(
                            i, control_class, type(drone_positions[i])))
        if self.MAP_PRESENCE == Constants.MapPresenceParameter.yes:
            if map is None or not isinstance(map, utils.Map):
                raise AttributeError(self._prediction_algorithm_name + ": wrong map parameter.")

    def observation_map(self, **kwargs):
        observation_map = None
        if "observation" in kwargs and kwargs["observation"] is not None:
            observation_map = kwargs["observation"]
        return observation_map

    def predict_pose(self, drone_positions=None, target_positions=None, map=None, **kwargs):
        self.check_parameters(drone_positions, target_positions, map)
        input_drone_positions = self.prepare_data_for_input(drone_positions)
        input_target_positions = self.prepare_data_for_input(target_positions)
        data = self.pose_from_parameters(input_drone_positions, input_target_positions, map, **kwargs)
        state_variables = self.state_variables(data, input_drone_positions, input_target_positions, map, **kwargs)
        output_data = self.prepare_data_for_output(data,
                                                   self.get_samples_count(drone_positions, target_positions,
                                                                          **kwargs))
        return output_data, state_variables

    def get_samples_count(self, drone_positions, target_positions, **kwargs):
        if "samples_count" in kwargs:
            return kwargs["samples_count"]
        if self.TARGET_POSITIONS > 0:
            ar = utils.DataStructures.pointcloud2_to_array(target_positions[0])
            return len(ar)
        if self.DRONE_POSITIONS > 0:
            ar = utils.DataStructures.pointcloud2_to_array(drone_positions[0])
            return len(ar)
        return 1000

    @abstractmethod
    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def state_variables(self, drone_positions, target_positions, map, **kwargs):
        raise NotImplementedError

    def prepare_data_for_input(self, data_collection):
        input_data_collection = []
        for data in data_collection:
            input_data = data
            if self.INPUT_TYPE == Constants.InputOutputParameterType.map:
                input_data = utils.DataStructures.array_to_image(utils.DataStructures.pointcloud2_to_array(data),
                                                                 self._map_width, self._map_height,
                                                                 self._map_resolution)
            if self.INPUT_TYPE == Constants.InputOutputParameterType.pose:
                input_data = utils.DataStructures.pointcloud2_to_pose_stamped(data)
            input_data_collection.append(input_data)
        return input_data_collection

    def prepare_data_for_output(self, data, samples_count):
        output_samples = data
        if self.OUTPUT_TYPE == Constants.InputOutputParameterType.map:
            output_samples = utils.DataStructures.image_to_array(data, samples_count, self._map_resolution,
                                                                 self._map_resolution, 2 * np.pi)
        if self.OUTPUT_TYPE == Constants.InputOutputParameterType.pose:
            output_samples = utils.DataStructures.pose_to_array(data, samples_count, self._map_resolution, np.pi)
        return utils.DataStructures.array_to_pointcloud2(output_samples)


class Database(object):

    def __init__(self, minimal_time):
        self.items = deque([], maxlen=minimal_time)
        self.times = deque([], maxlen=minimal_time)
        self._first_iteration = True

    def init_db(self, world_map, particles_count=2000, max_values=(3, 2, 0.5), frames_count=3,
                init_frames=None):
        rounded_time = utils.Math.rounding(rospy.Time.now().to_sec())
        if rounded_time - frames_count * 0.5 < 0:
            rounded_time = frames_count * 0.5
        if init_frames is None:
            frame = np.zeros((particles_count, len(utils.DataStructures.point_cloud_name())))
            positions = np.array(np.nonzero(world_map.target_obstacle_map))
            # print("shape", world_map.target_obstacle_map.shape)
            points = zip(positions[1, :], positions[0, :])
            tree = cKDTree(points)
            positions_indices = np.random.randint(0, len(positions[0]), particles_count)
            frame[:, 0] = positions[0, positions_indices] + np.random.uniform(0, 0.5, (particles_count))
            frame[:, 1] = positions[1, positions_indices] + np.random.uniform(0, 0.5, (particles_count))
            frame[:, 2] = np.random.uniform(0.5, max_values[2], particles_count)
            frame[:, 5] = np.random.uniform(0, 2 * np.pi, particles_count)
            frame[:, 6] = np.arange(0, particles_count)
            frame[:, 7] = 1
            self.times.append(rounded_time - 0.5 * (frames_count))
            pointcloud = utils.DataStructures.array_to_pointcloud2(frame, rospy.Time.from_sec(self.times[-1]))
            self.items.append(pointcloud)
            for i in range(frames_count - 1):
                self.times.append(rounded_time - 0.5 * (frames_count - 1 - i))
                new_frame = np.zeros((particles_count, len(utils.DataStructures.point_cloud_name())))
                ii = tree.query_ball_point(frame[:, :2], max_values[0] * world_map.resolution)
                for j in range(len(ii)):
                    i = ii[j]
                    if len(i) == 0:
                        new_frame[j, 0] = frame[j, 0]
                        new_frame[j, 1] = frame[j, 1]
                        new_frame[j, 2] = frame[j, 2]
                        new_frame[j, 5] = frame[j, 5]
                    else:
                        index = np.random.choice(i, 1)
                        new_frame[j, 0] = positions[0, index] + np.random.uniform(0, 0.5, 1)[0]
                        new_frame[j, 1] = positions[1, index] + np.random.uniform(0, 0.5, 1)[0]
                        new_frame[j, 2] = np.random.uniform(0.5, max_values[2], 1)[0]
                        yaw = utils.Math.calculate_yaw_from_points(0, 0, frame[j, 0] - new_frame[j, 0],
                                                                   frame[j, 1] - new_frame[j, 1])
                        new_frame[j, 5] = yaw
                    new_frame[j, 6] = j
                    new_frame[:, 7] = 1
                frame = new_frame
                pointcloud = utils.DataStructures.array_to_pointcloud2(frame, rospy.Time.from_sec(self.times[-1]))
                self.items.append(pointcloud)
        else:
            for i in range(len(init_frames)):
                self.times.append(rounded_time - 0.5 * (len(init_frames) - i))
                self.items.append(init_frames[i])

    def get_time(self, time):
        self.recalculate_times()
        index = int(((utils.Math.rounding(self.times[-1])) - utils.Math.rounding(time)) / 0.5)
        if len(self.items) > index > -1:
            # print(index, "index for time", time)
            return self.items[len(self.items) - 1 - index]
        return None

    def add(self, notification):
        self.recalculate_times()
        rounded_time = utils.Math.rounding(rospy.Time.now().to_sec())
        if len(self.times) == 0 or self.times[len(self.times) - 1] != rounded_time:
            self.times.append(rounded_time)
            self.items.append(notification)

    def get_time_boundaries(self):
        self.recalculate_times()
        if len(self.items) > 0:
            return utils.Math.rounding(self.times[-1] - len(self.times) * 0.5), \
                   utils.Math.rounding(self.times[-1]), \
                   len(self.times)
        else:
            return -1, -1, 0

    def recalculate_times(self):
        rounded_time = utils.Math.rounding(rospy.Time.now().to_sec())
        if len(self.times) == 3 and self._first_iteration and rounded_time > 1.5:
            self._first_iteration = False
            for i in range(len(self.times)):
                self.times[i] = rounded_time - 0.5 * (len(self.times) - i)
                self.items[i].header.stamp = rospy.Time.from_sec(self.times[i])


class PredictionManagement(object):

    def __init__(self):
        super(PredictionManagement, self).__init__()
        environment_configuration = rospy.get_param("environment_configuration")
        target_configuration = rospy.get_param("target_configuration")
        drone_configuration = rospy.get_param("drone_configuration")
        self._map_frame = environment_configuration["map"]["frame"]
        self._map_file = environment_configuration["map"]["obstacles_file"]
        self._object_size = 1.0
        self.map = utils.Map.get_instance(self._map_file)
        self._predicted_maps_size = target_configuration["localization"]["map_size"]
        self._predicted_maps_center = Point()
        self._predicted_maps_center.x = target_configuration["localization"]["map_center"]["x"]
        self._predicted_maps_center.y = target_configuration["localization"]["map_center"]["y"]
        self._map_resolution = 0.5
        self._prediction_systems = []
        self._predicted_positions = []
        self._state_variables = []
        self._last_prediction_update_time = []
        self._default_system = 0
        self._last_positions_times = [-1, -1]
        self._earliest_positions_times = [float("inf"), float("inf")]
        self._ignore_true_prediction = 0
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self._drone_base_link_frame = drone_configuration["properties"]["base_link"]
        self._predicted_target_frame = target_configuration["localization"]["target_predicted_position"]
        self._players_frames = [self._drone_base_link_frame, self._predicted_target_frame]
        self._drone_type = 0
        self._target_type = 1
        self._target_position_offset = 0
        self._drone_position_offset = 0
        self._target_position_service = None
        self._drone_position_service = None
        self._db = Database(20)

    @abstractmethod
    def compute_recommended_prediction_alg(self):
        raise NotImplementedError()

    @abstractmethod
    def get_position_from_history(self, time, ignore_exceptions_messages=False):
        raise NotImplementedError()

    @abstractmethod
    def prepare_kwargs(self, time, drone_positions, target_positions, world_map):
        raise NotImplementedError()

    @abstractmethod
    def opponent_service_for_drone_positions_necessary(self):
        raise NotImplementedError()

    @abstractmethod
    def opponent_service_for_target_positions_necessary(self):
        raise NotImplementedError()

    @abstractmethod
    def get_main_type(self):
        raise NotImplementedError()

    @abstractmethod
    def check_boundaries_target(self, count_needed):
        raise NotImplementedError("Checking boundaries requested, but not implemented.")

    @abstractmethod
    def check_boundaries_drone(self, count_needed):
        raise NotImplementedError("Checking boundaries requested, but not implemented.")

    @abstractmethod
    def get_topic_bounds(self):
        raise NotImplementedError("This method wasnt implemented.")

    def prepare_map(self, last_time, map_presence):
        world_map = None
        if map_presence:
            world_map = self.map
        return world_map

    def prepare_target_positions(self, last_time, positions_count):
        service = None
        if self.opponent_service_for_target_positions_necessary():
            service = self._target_position_service
        return self.prepare_positions(last_time, positions_count, self._target_position_offset, service)

    def prepare_drone_positions(self, last_time, positions_count):
        service = None
        #print("Drone positions")
        if self.opponent_service_for_target_positions_necessary():
            service = self._drone_position_service
        return self.prepare_positions(last_time, positions_count, self._drone_position_offset, service)

    def prepare_positions(self, last_time, positions_count, offset, service):
        positions = []
        for i in range(positions_count):
            past_time = last_time - (i + offset) * 0.5
            if service is not None:
                header = Header()
                header.stamp = rospy.Time.from_sec(past_time)
                position = service(header, -1, False, None)
                position = position.pointcloud
            else:
                #print("Prepare {}".format(past_time))
                position, _ = self.recursive_get_position_in_time(past_time, preparation=True)
            # print("pos", position)
            if position is None:
                positions = None
                break
            positions.append(position)
        if positions is not None:
            positions.reverse()
        return positions

    def get_latest_player_position_time(self, player_type):
        return self._last_positions_times[player_type]

    def get_earliest_player_position_time(self, player_type):
        return self._earliest_positions_times[player_type]

    def set_earliest_player_position_type(self, player_type, value):
        self._earliest_positions_times[player_type] = value

    def set_latest_player_position_time(self, player_type, value):
        self._last_positions_times[player_type] = value

    def get_player_ff_frame(self, player_type):
        return self._players_frames[player_type]

    def is_prediction_possible(self):
        mechanism_params = self._prediction_systems[self._default_system].get_algorithm_parameters()
        drone_position = Constants.PredictionParametersNames.drone_positions_list
        target_position = Constants.PredictionParametersNames.target_positions_list
        if mechanism_params[drone_position] > 0:
            if self.opponent_service_for_drone_positions_necessary():
                srv = self.get_drone_position_service()
                if not self.check_boundaries_drone(mechanism_params[drone_position]) or srv is None:
                    return False
            else:
                if not self.check_boundaries_drone(mechanism_params[drone_position]):
                    return False
        if mechanism_params[target_position] > 0:
            if self.opponent_service_for_target_positions_necessary():
                srv = self.get_target_position_service()
                if not self.check_boundaries_target(mechanism_params[target_position]) or srv is None:
                    return False
            else:
                if not self.check_boundaries_target(mechanism_params[target_position]):
                    return False
        if mechanism_params[Constants.PredictionParametersNames.map_presence] == Constants.MapPresenceParameter.yes:
            if self.map is None:
                return False
        return True

    def database_boundaries(self, count_needed, player_type):
        earliest_position_time, last_position_time, count = self._db.get_time_boundaries()
        return self.check_boundaries_method(count_needed, player_type, earliest_position_time, last_position_time,
                                            count)

    def check_boundaries_method(self, count_needed, player_type, earliest_position_time, last_position_time, count):
        self.set_earliest_player_position_type(player_type, earliest_position_time)
        self.set_latest_player_position_time(player_type, last_position_time)
        if count < count_needed:
            return False
        return True

    def tf_boundaries(self, count_needed, player_type):
        earliest_position_time, last_position_time, count = self.get_tf_bounds(self.get_player_ff_frame(player_type))
        return self.check_boundaries_method(count_needed, player_type, earliest_position_time, last_position_time,
                                            count)

    def topic_boundaries(self, count_needed, player_type):
        earliest_position_time, last_position_time, count = self.get_topic_bounds()
        return self.check_boundaries_method(count_needed, player_type, earliest_position_time, last_position_time,
                                            count)

    def get_tf_bounds(self, frame):
        earliest_time = 0.1
        latest_time = utils.Math.rounding(rospy.Time.now().to_sec())
        earliest, ret_value_earliest = self.get_time(earliest_time, frame)
        latest, ret_value_latest = self.get_time(0, frame)
        if ret_value_earliest:
            earliest = latest_time
        if not ret_value_latest:
            latest = earliest_time
        #print("Latest time {}".format(latest))
        latest = utils.Math.floor_rounding(latest)
        earliest = utils.Math.floor_rounding(earliest)
        count = (latest_time - earliest) / 0.5
        #print("Rounded latest: {}".format(utils.Math.floor_rounding(latest)))
        return utils.Math.ceiling_rounding(earliest), utils.Math.floor_rounding(latest), count

    def get_time(self, time, frame, silence=True):
        try:
            rospy_time = rospy.Time.from_sec(time)
            t = self._tfBuffer.lookup_transform(self._map_frame, frame, rospy_time)
            return t.header.stamp.to_sec(), True
        except (tf2_ros.ExtrapolationException) as excs:
            if not silence:
                print("Retrieving time from ros_tf database based on: {}".format(excs))
            digits = [float(s) for s in re.findall("\d+\.\d+", str(excs))]
            return digits[1], False
        except Exception as e:
            return -1, True

    def object_prediction_poses_check(self, renew):
        for i in range(len(self._last_prediction_update_time)):
            time_intersection = self.get_latest_player_position_time(self.get_main_type()) - \
                                self._last_prediction_update_time[i]
            if time_intersection / 0.5 > self._ignore_true_prediction:
                # new prediction from history if it works
                # print("Delete")
                self._last_prediction_update_time[i] = utils.Math.floor_rounding(
                    self.get_latest_player_position_time(self.get_main_type()))
                self._predicted_positions[i] = []
                self._state_variables[i] = []
                self._predicted_maps_center = None

    def load_target_position_service(self, service_name):
        self._target_position_service_name = service_name
        srv = self.get_target_position_service()
        if srv is not None:
            return True
        return False

    def load_drone_position_service(self, service_name):
        self._drone_position_service_name = service_name
        srv = self.get_drone_position_service()
        if srv is not None:
            return True
        return False

    def get_target_position_service(self):
        self._target_position_service = self.get_opponent_position_service(self._target_position_service_name)
        return self._target_position_service

    def get_drone_position_service(self):
        self._drone_position_service = self.get_opponent_position_service(self._drone_position_service_name)
        return self._drone_position_service

    def get_opponent_position_service(self, service_name):
        try:
            position_srv = rospy.ServiceProxy(service_name, PositionRequest)
            return position_srv
        except rospy.ServiceException as exc:
            rospy.logwarn("Service unavailable: " + str(exc))
            return None

    def get_prediction_algs_count(self):
        return len(self._prediction_systems)

    def set_default_prediction_alg(self, index):
        if -1 < index < self.get_prediction_algs_count():
            self._default_system = index

    def set_recommended_prediction_alg(self):
        self._default_system = self.compute_recommended_prediction_alg()

    def predict_object_pose(self, time, renew=False, preparation=False):
        rounded_time = utils.Math.rounding(time)
        self.object_prediction_poses_check(renew)
        if self.get_latest_player_position_time(self.get_main_type()) >= rounded_time - 0.5:
            #print("Direct request", rounded_time - 0.5)
            predicted_position, state_variables = self.predict_for_given_time(rounded_time, False, preparation)
            return predicted_position, state_variables
        else:
            object_prediction_poses = self._predicted_positions[self._default_system]
            object_state_variables = self._state_variables[self._default_system]
            latest_prediction_time = len(object_prediction_poses) * 0.5 + self._last_prediction_update_time[
                self._default_system]
            """print("latest prediction time", latest_prediction_time, "len(object_prediction_poses)",
                 len(object_prediction_poses), "self._last_prediction_update_time[self._default_system]",
                 self._last_prediction_update_time[self._default_system], "rounded_time", rounded_time)"""
            # while latest_prediction_time <= rounded_time:
            next_prediction_time = latest_prediction_time + 0.5
            while next_prediction_time <= rounded_time:
                # print("latest prediction time inside while", next_prediction_time)
                if next_prediction_time < rounded_time:
                    preparation = True
                else:
                    preparation = False
                predicted_position, state_variables = self.predict_for_given_time(next_prediction_time, renew, preparation)
                if predicted_position is None:
                    rospy.logwarn("Predicted position is none for time ", next_prediction_time)
                    return None, {}
                object_prediction_poses.append(predicted_position)
                object_state_variables.append(state_variables)
                next_prediction_time += 0.5
            index = int((rounded_time - self._last_prediction_update_time[self._default_system]) / 0.5) - 1
            #print("Index {}".format(index))
            if index < len(object_prediction_poses):
                return object_prediction_poses[index], object_state_variables[index]
            else:
                rospy.logwarn("index >=  len(object_prediction_poses)")
                return None, {}

    def get_opponent_position_in_time(self, time, type, service):
        header = Header()
        header.stamp = rospy.Time.from_sec(time)
        try:
            response = service.call(header, type, -1, False, None)
            if response.pointcloud is not None:
                return response.pointcloud
        except rospy.ServiceException as exc:
            rospy.logwarn("Service did not process request: " + str(exc))
        return None

    def predict_for_given_time(self, time, renew, preparation):
        #print("predict for given time {}".format(time))
        object_prediction_mechanism = self._prediction_systems[self._default_system]
        input_data, kwargs = self.prepare_input_data_for_prediction_algorithm(
            object_prediction_mechanism.get_algorithm_parameters(), time, renew, preparation)
        predicted_position = None
        state_variables = {}
        try:
            predicted_position, state_variables = object_prediction_mechanism.predict_pose(*input_data, **kwargs)
            predicted_position.header.stamp = rospy.Time.from_sec(time)
        except AttributeError as ae:
            rospy.loginfo("Problem with prediction for time " + str(time) + " : " + str(ae))
            rospy.loginfo(traceback.format_exc())
        #print("Predict for given time: state variables {}".format(state_variables))
        return predicted_position, state_variables

    def prepare_input_data_for_prediction_algorithm(self, parameter, time, renew, preparation):
        # input type
        map_presence = parameter[3] == Constants.MapPresenceParameter.yes
        # drone positions
        drone_positions = self.prepare_drone_positions(time, parameter[1])
        # target positions
        target_positions = self.prepare_target_positions(time, parameter[2])
        # map presence
        world_map = self.prepare_map(time, map_presence)
        kwargs = self.prepare_kwargs(time, drone_positions, target_positions, world_map)
        kwargs["final_prediction"] = preparation
        return [drone_positions, target_positions, world_map], kwargs

    def get_position_in_time(self, time, requested_type=Constants.InputOutputParameterType.pointcloud, renew=False,
                             unchecked=False, **kwargs):
        time = utils.Math.rounding(time)
        if not unchecked and not self.is_prediction_possible():
            rospy.logwarn("checking failed")
            return None, {}
        #print("position in time")
        position, state_variables = self.recursive_get_position_in_time(time, renew)
        array = utils.DataStructures.pointcloud2_to_array(position)
        if position is not None:
            array = utils.DataStructures.pointcloud2_to_array(position)
            if requested_type == Constants.InputOutputParameterType.pose:
                pose = utils.DataStructures.array_to_pose(array)
                position = self.create_pose_stamped(pose, time)
            elif requested_type == Constants.InputOutputParameterType.map:
                completed_kwargs = self.complete_kwargs_for_map(**kwargs)
                position = utils.DataStructures.array_to_image(array, completed_kwargs["width"],
                                                               completed_kwargs["height"], completed_kwargs["step"],
                                                               completed_kwargs["center"])
        return position, state_variables

    def create_pose_stamped(self, pose, time):
        if pose is None:
            return None
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.from_sec(time)
        pose_stamped.header.frame_id = self._map_frame
        pose_stamped.pose = pose
        return pose_stamped

    def recursive_get_position_in_time(self, time, renew=False, preparation=False):
        """print("time", time, "earliest", self.get_earliest_player_position_time(self.get_main_type()),
              "last_position_time", self.get_latest_player_position_time(self.get_main_type()))"""
        poses, state_variables = None, {}
        if time > self.get_latest_player_position_time(self.get_main_type()):
            # print("Position prediction.")
            poses, state_variables = self.predict_object_pose(time, renew, preparation)
        else:
            #print("History position.")
            if self.get_earliest_player_position_time(self.get_main_type()) > time:
                #print("earliest position problem")
                return None, {}
            poses = self.get_position_from_history(time)
        return poses, state_variables
        # print("target pose", target_pose)
        # print("target pose is none")

    def complete_kwargs_for_map(self, **kwargs):
        if "width" not in kwargs or kwargs["width"] is None:
            kwargs["width"] = self._predicted_maps_size
        if "height" not in kwargs or kwargs["height"] is None:
            kwargs["height"] = self._predicted_maps_size
        if "step" not in kwargs or kwargs["step"] is None:
            kwargs["step"] = self._map_resolution
        if "center" not in kwargs or kwargs["center"] is None:
            kwargs["center"] = (self._predicted_maps_center.x, self._predicted_maps_center.y)
        return kwargs

    def get_center_of_positions(self, center, positions):
        if center is not None:
            return center
        else:
            last_position = positions[len(positions) - 1]
            p = Point()
            p.x = last_position.pose.position.x
            p.y = last_position.pose.position.y
            return p

    def update_last_position_time(self, time):
        pass

    def reset_all_prediction(self):
        for i in range(len(self._predicted_positions)):
            self.reset_prediction_from_algorithm(i)

    def reset_prediction_from_algorithm(self, index):
        self._predicted_positions[index] = []
        self._state_variables[index] = []

    def prepare_structures(self):
        t = utils.Math.rounding(rospy.Time.now().to_sec())
        for i in range(len(self._prediction_systems)):
            self._predicted_positions.append([])
            self._last_prediction_update_time.append(t)
            self._state_variables.append([])
