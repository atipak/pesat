#!/usr/bin/env python
import rospy
import cv2
import tf as tf_ros
import tf2_ros
import numpy as np
import rospkg
from geometry_msgs.msg import TransformStamped, PoseStamped, Point, PointStamped, Pose
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
from pesat_msgs.msg import ImageTargetInfo, FloatImageStamped, Notification, CameraShot
from pesat_msgs.srv import PositionRequest, PointInTime, CameraRegistration
import helper_pkg.utils as utils
from collections import namedtuple, deque
from helper_pkg.utils import Constants
import helper_pkg.PredictionManagement as pm
from algorithms.target_neural import PredictionNetwork
from algorithms.target_reactive import PredictionNaive
from algorithms.target_particles import PredictionWithParticles
from sensor_msgs.msg import PointCloud2
import time
import traceback

lowerBound0 = np.array([0, 100, 90])
upperBound0 = np.array([10, 255, 255])
lowerBound1 = np.array([160, 100, 90])
upperBound1 = np.array([179, 255, 255])
# np.set_printoptions(threshold=np.nan)

kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))
font = cv2.FONT_HERSHEY_SIMPLEX


class VisionNetwork:
    WIDTH = 856
    HEIGHT = 480
    VALUES = 3
    LABELS = 2

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

    def construct(self):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 3], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.values = tf.placeholder(tf.int64, [None, self.VALUES], name="values")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=self.images, filters=5, kernel_size=[3, 3], padding="same",
                                     activation=None, use_bias=False)
            batchNorm1 = tf.layers.batch_normalization(conv1, training=self.is_training)
            relu1 = tf.nn.relu(batchNorm1)

            # Second branch - class prediction
            # Convolutional Layer #3 and Pooling Layer #1
            conv3 = tf.layers.conv2d(inputs=relu1, filters=10, kernel_size=[3, 3], padding="same",
                                     activation=None, use_bias=False)
            batchNorm3 = tf.layers.batch_normalization(conv3, training=self.is_training)
            relu3 = tf.nn.relu(batchNorm3)
            pool1 = tf.layers.max_pooling2d(inputs=relu3, pool_size=[2, 2], strides=2)

            conv4 = tf.layers.conv2d(inputs=pool1, filters=15, kernel_size=[3, 3], padding="same",
                                     activation=None, use_bias=False)
            batchNorm4 = tf.layers.batch_normalization(conv4, training=self.is_training)
            relu4 = tf.nn.relu(batchNorm4)
            pool2 = tf.layers.max_pooling2d(inputs=relu4, pool_size=[2, 2], strides=2)

            flattened_layer = tf.layers.flatten(pool2, name="flatten")

            # dropout = tf.layers.dropout(hidden_layer, rate=0.5, training=self.is_training)
            values_predictions = tf.layers.dense(flattened_layer, self.VALUES, activation=None,
                                                 name="values_output_layer")
            labels_predictions = tf.layers.dense(flattened_layer, self.LABELS, activation=None,
                                                 name="labels_output_layer")
            self.values_predictions = tf.abs(tf.cast(values_predictions, tf.int64))
            self.labels_predictions = tf.argmax(labels_predictions, axis=1)

            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.values_predictions],
                                {self.images: images, self.is_training: False})

    def restore(self, path):
        self.saver.restore(self.session, path)


# notes
# FOV_Horizontal = 2 * atan(W/2/f) = 2 * atan2(W/2, f)  radians
# FOV_Vertical   = 2 * atan(H/2/f) = 2 * atan2(H/2, f)  radians
# FOV_Diagonal   = 2 * atan2(sqrt(W^2 + H^2)/2, f)


class PredictionLocalization(pm.PredictionManagement):
    def __init__(self):
        super(PredictionLocalization, self).__init__()
        self.Pose = namedtuple("pose", ["x", "y", "yaw"])
        target_configuration = rospy.get_param("target_configuration")
        drone_configuration = rospy.get_param("drone_configuration")
        environment_configuration = rospy.get_param("environment_configuration")
        self._prediction_systems = [PredictionWithParticles()]  # , PredictionNaive(), PredictionNetwork(1)]
        self.prepare_structures()
        self._default_system = 0
        opponent_service_name = drone_configuration["localization"]["position_in_time_service"]
        self.load_drone_position_service(opponent_service_name)
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self._br = tf2_ros.TransformBroadcaster()
        self._target_position_offset = 1
        self._drone_position_offset = 1
        self._observation = None
        self._last_seen = False
        self._seen_brake = False
        self._request_for_update_time = 0
        self._observation_time = 0
        self._count_publisher = rospy.Publisher(target_configuration["localization"]["items_count_topic"],
                                                Float32, queue_size=10)
        self._earliest_time_publisher = rospy.Publisher(target_configuration["localization"]["earliest_time_topic"],
                                                        Float32, queue_size=10)
        self._latest_time_publisher = rospy.Publisher(target_configuration["localization"]["latest_time_topic"],
                                                      Float32, queue_size=10)
        rospy.Subscriber(environment_configuration["watchdog"]["camera_notification_topic"], PointCloud2,
                         self.callback_notify)
        self._db.init_db(self.map, target_position=environment_configuration["init_positions"]["target"])
        self._vision_information = None
        self._samples_length = None

    def callback_notify(self, data):
        self._observation_time = rospy.Time.now().to_sec()
        self._observation = utils.DataStructures.pointcloud2_to_array(data)

    def opponent_service_for_drone_positions_necessary(self):
        return True

    def opponent_service_for_target_positions_necessary(self):
        return False

    def compute_recommended_prediction_alg(self):
        return 0

    def prepare_kwargs(self, time, drone_positions, target_positions, world_map):
        seen = False
        if self._seen_brake:
            self._last_seen = seen
            self._seen_brake = False
            return {"observation": self._observation, "seen": seen}
        if self._vision_information is not None and self._samples_length is not None:
            if self._vision_information.quotient > 0.7 and self._samples_length > 0:
                seen = True
        # reset
        if self._last_seen == seen and self._request_for_update_time > 0:
            self._request_for_update_time = 0
        if self._last_seen != seen and seen and self._request_for_update_time == 0:
            self._request_for_update_time = rospy.Time.now().to_sec()
        if seen and self._request_for_update_time > 0 and np.abs(
                self._request_for_update_time - rospy.Time.now().to_sec()) < 0.2:
            seen = not seen
        self._last_seen = seen
        return {"observation": self._observation, "seen": seen}

    def get_main_type(self):
        return self._target_type

    def check_boundaries_target(self, count_needed):
        return self.database_boundaries(count_needed, self._target_type)

    def check_boundaries_drone(self, count_needed):
        return self.tf_boundaries(count_needed, self._drone_type)

    def get_position_from_history(self, time, ignore_exceptions_messages=False):
        rounded_time = utils.Math.rounding(time)
        positions = self._db.get_time(rounded_time)
        return positions

    def update_structure_from_information_message(self, information, samples_length,seen):
        # print("-------------------------------------------------111--------------------------------------")
        self._vision_information = information
        self._samples_length = samples_length
        if self._last_seen and not seen:
            self._seen_brake = True
        time = rospy.Time.now().to_sec()
        # add predicted value
        if len(self._prediction_systems) > 0 and self._default_system < len(self._prediction_systems):
            current_position, _ = self.get_position_in_time(rospy.Time.now().to_sec())
            if current_position is not None:
                self._db.add(current_position)
                return True
        current_position = self.get_position_from_history(time)
        if current_position is not None:
            self._db.add(current_position)
            return True
        return False

    def pose_to_transform(self, pose, time):
        t = TransformStamped()
        t.header.stamp = rospy.Time.from_sec(time)
        t.header.frame_id = self._map_frame
        t.child_frame_id = self.get_player_ff_frame(self._target_type)
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation.x = pose.orientation.x
        t.transform.rotation.y = pose.orientation.y
        t.transform.rotation.z = pose.orientation.z
        t.transform.rotation.w = pose.orientation.w
        return t

    def add_position_to_history(self, tf_frame):
        # print(tf_frame.header.stamp.to_sec())
        self.update_last_position_time(tf_frame.header.stamp.to_sec())
        t = TransformStamped()
        t.header.stamp = tf_frame.header.stamp
        t.header.frame_id = self._map_frame
        t.child_frame_id = self.get_player_ff_frame(self._target_type)
        t.transform.translation = tf_frame.transform.translation
        t.transform.rotation = tf_frame.transform.rotation
        self._br.sendTransform(t)

    def get_target_speed_in_time(self, time):
        last_time = time - 0.5
        rounded_last_time = utils.Math.rounding(last_time)
        target_last_pose, _ = self.get_position_in_time(rounded_last_time, Constants.InputOutputParameterType.pose,
                                                        False)
        target_current_pose, _ = self.get_position_in_time(time, Constants.InputOutputParameterType.pose, False)
        speed = PointStamped()
        speed.header.stamp = rospy.Time.from_sec(rounded_last_time)
        speed.header.frame_id = self._map_frame
        speed.point.x = target_current_pose.pose.position.x - target_last_pose.pose.position.x
        speed.point.y = target_current_pose.pose.position.y - target_last_pose.pose.position.y
        return speed

    def update_database_topics(self):
        et, lt, count = self._db.get_time_boundaries()
        self._count_publisher.publish(count)
        self._earliest_time_publisher.publish(et)
        self._latest_time_publisher.publish(lt)


class VisionLocalisation(object):

    def __init__(self):
        super(VisionLocalisation, self).__init__()
        rospy.init_node('vision', anonymous=False)
        _camera_configuration = rospy.get_param("drone_configuration")["camera"]
        _drone_configuration = rospy.get_param("drone_configuration")
        vision_configuration = rospy.get_param("target_configuration")
        environment_configuration = rospy.get_param("environment_configuration")
        self.hfov = _camera_configuration["hfov"]
        self._map_file = environment_configuration["map"]["obstacles_file"]
        self._map = utils.Map.get_instance(self._map_file)
        print("Count", self._map.free_target_pixels)
        self.image_height = _camera_configuration["image_height"]
        self.image_width = _camera_configuration["image_width"]
        self.camera_range = _camera_configuration["camera_range"]
        self.focal_length = self.image_width / (2.0 * np.tan(self.hfov / 2.0))
        self.vfov = 2 * np.arctan2(self.image_height / 2, self.focal_length)
        self.dfov = 2 * np.arctan2(np.sqrt(self.image_width ^ 2 + self.image_height ^ 2) / 2, self.focal_length)
        self.vangle_per_pixel = self.vfov / 480
        self.hangle_per_pixel = self.hfov / 856
        self.bridge = CvBridge()
        self.image = None
        self.newest_image = None
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.lock = False
        self.tf_frames_count = 0
        self.information_count = 0
        self.last_position = None
        self._network_on = False
        self.network = None
        self._estimations_distance = 0.5
        self._frequency = 10.0
        self._estimations_count = int(self._frequency * self._estimations_distance)
        self._iteration = 0
        self._estimations = deque(maxlen=3)
        self._distance_limit = 2
        self._weight_start = 1
        self._weight_distance = 0.05
        self._last_estimated_position = None
        self._last_known_information = None
        self._last_known_values = None
        self._last_tf_frame = None
        self._drone_tf_frame = None
        self._radius = 50
        self._min_location_accuracy = 0.2
        self._max_location_accuracy = 3
        self._max_likelihood = 0.98
        self._min_likelihood = 0.7
        self._min_recognition_error = 0.01
        self._max_recognition_error = 0.1
        self._recognition_error_difference = self._max_recognition_error - self._min_recognition_error
        self._step_likelihood = (self._max_likelihood - self._min_likelihood) / self._radius
        self._step_location_accuracy = (self._max_location_accuracy - self._min_location_accuracy) / self._radius
        self._max_fly_speed = _drone_configuration["control"]["video_max_horizontal_speed"]
        self._map_frame = "map"
        self._camera_base_link = "bebop2/camera_base_link"
        self._vision_target_position_frame = vision_configuration["localization"]["target_position"]
        self.pub_target_info = rospy.Publisher(vision_configuration["localization"]["target_information"],
                                               ImageTargetInfo, queue_size=10)
        rospy.Subscriber(_camera_configuration["image_channel"], Image, self.callback_image)
        self._camera_registration_service_name = environment_configuration["watchdog"]["camera_registration_service"]
        self._camera_id = None  # None
        self._pub_shot = rospy.Publisher(environment_configuration["watchdog"]["camera_shot_topic"], CameraShot,
                                         queue_size=10)

    def callback_image(self, data):
        try:
            if not self.lock:
                self._image_header = data.header
                self.newest_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

    def get_circles(self, mask):
        edges = cv2.Canny(mask, 100, 10)
        cv2.imwrite("m.bmp", mask)
        cv2.imwrite("edges.bmp", edges)
        rows = mask.shape[0]
        columns = mask.shape[1]
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=50, param2=15, minRadius=1,
                                   maxRadius=columns)
        return circles

    def get_network_prediction(self):
        # predict
        transposed_image = np.transpose(self.image, (1, 0, 2))
        resized_image = cv2.resize(cv2.cvtColor(transposed_image, cv2.COLOR_RGB2BGR),
                                   (VisionNetwork.WIDTH, VisionNetwork.HEIGHT))
        if self._network_on:
            if self.network is None:
                self.network = VisionNetwork(1)
                self.network.construct()
                rospack = rospkg.RosPack()
                model_path = rospack.get_path('pesat_resources') + "/models/target_localization/ball_recognition"
                self.network.restore(model_path)
            labels, values = self.network.predict([resized_image])
            values = values[0]
            labels = labels[0]
        else:
            values = [0.0, 0.0, 0.0]
            labels = 0
        return values, labels, resized_image

    def compare_algorithms_find_all_choice(self, algorithms):
        similiar_until_now = []
        # trough all algorithms
        for alg_i in range(len(algorithms)):
            for alg_j in range(alg_i, len(algorithms)):
                if len(algorithms[alg_j]) == 0:
                    return None
                for circle_i in range(len(algorithms[alg_i])):
                    if circle_i >= len(algorithms[alg_j]):
                        break
                    for circle_j in range(circle_i, len(algorithms[alg_j])):
                        if abs(algorithms[alg_i][circle_i][2] - algorithms[alg_j][circle_j][2]) < 4 and abs(
                                algorithms[alg_i][circle_i][1] - algorithms[alg_j][circle_j][1]) < 4 and abs(
                            algorithms[alg_i][circle_i][0] - algorithms[alg_j][circle_j][0]) < 4:
                            circle = [(algorithms[alg_i][circle_i][0] + algorithms[alg_j][circle_j][0]) / 2,
                                      (algorithms[alg_i][circle_i][1] + algorithms[alg_j][circle_j][1]) / 2,
                                      (algorithms[alg_i][circle_i][2] + algorithms[alg_j][circle_j][2]) / 2]
                            for circle_j in range(circle_i, len(algorithms[alg_j])):
                                if abs(algorithms[alg_i][circle_i][2] - algorithms[alg_j][circle_j][2]) < 4 and abs(
                                        algorithms[alg_i][circle_i][1] - algorithms[alg_j][circle_j][1]) < 4 and abs(
                                    algorithms[alg_i][circle_i][0] - algorithms[alg_j][circle_j][0]) < 4:
                                    pass

    def check_hough_and_net(self, circles, values, labels):
        if circles is not None and len(circles) > 0 and labels == 0:
            for circle in circles:
                if abs(circle[2] - values[2]) < 4 and abs(circle[1] - values[1]) < 4 and abs(
                        circle[0] - values[0]) < 4:
                    return [(circle[0] + values[0]) / 2, (circle[1] + values[1]) / 2,
                            (circle[2] + values[2]) / 2]
        return None

    def real_object_from_circle(self, circle, maskFinal, image, trans, roll, pitch, yaw):
        # object presented in the image
        center = (int(circle[0]), int(circle[1]))
        radius = int(circle[2])
        # create mask with locality of circle
        circle_mask = np.zeros(self.image.shape[0:2], dtype='uint8')
        cv2.circle(circle_mask, center, radius, 255, -1)
        # how much are red color and circle overlapping
        and_image = cv2.bitwise_and(maskFinal, circle_mask)
        area = int(np.pi * radius * radius)
        area = min(area, cv2.countNonZero(maskFinal))
        nzCount = cv2.countNonZero(and_image)
        quotient = 0
        x = 0
        y = 0
        z = 0
        if area > 0:
            quotient = float(nzCount) / float(area)
        if quotient > 0.8:
            # distance from focal length and average
            distance = self.focal_length / radius
            # image axis
            centerX = image.shape[1] / 2  # cols
            centerY = image.shape[0] / 2  # rows
            # shift of target in image from image axis
            diffX = np.abs(centerX - center[0])
            diffY = np.abs(centerY - center[1])
            # sign for shift of camera
            signX = np.sign(centerX - center[0])
            signY = np.sign(centerY - center[1])
            # shift angle of target in image for camera
            angleX = np.arctan2(diffX, self.focal_length) * signX * -1
            angleY = np.arctan2(diffY, self.focal_length) * signY * -1
            angleX1 = diffX * signX * self.hangle_per_pixel
            angleY1 = diffY * signY * self.vangle_per_pixel
            # direction of camera in drone_position frame
            # direction of camera
            m = tf_ros.transformations.euler_matrix(roll - angleY, pitch, yaw - angleX)
            vector = m.dot([0.0, 0.0, 1.0, 1.0])
            vector_distance = np.sqrt(
                np.power(distance, 2) + np.power(np.tan(angleX) * distance, 2) + np.power(np.tan(angleY) * distance, 2))
            z = trans.transform.translation.z + vector_distance * vector[2]
            x = trans.transform.translation.x + vector_distance * vector[0]
            y = trans.transform.translation.y + vector_distance * vector[1]
        return {"q": quotient, "x": x, "y": y, "z": z}

    def get_final_image_mask(self):
        # convert BGR to HSV
        imgHSV = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        # print(cv2.imshow("image", imgHSV))
        # create the Masks for red color
        mask0 = cv2.inRange(imgHSV, lowerBound0, upperBound0)
        mask1 = cv2.inRange(imgHSV, lowerBound1, upperBound1)
        # morphology
        maskOpen = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
        maskOpen1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernelOpen)
        maskClose1 = cv2.morphologyEx(maskOpen1, cv2.MORPH_CLOSE, kernelClose)

        # join masks
        maskClose[maskClose1 == 255] = 255
        maskFinal = maskClose
        return maskFinal

    def get_correct_circle(self):
        correct_circle = None
        correct_values = None
        if self.image is not None:
            maskFinal = self.get_final_image_mask()
            values, labels, resized_image = self.get_network_prediction()
            circles = self.get_circles(maskFinal)
            if circles is not None:
                circles = circles[0]
            else:
                circles = []
            if len(circles) > 0 or labels == 0:
                try:
                    trans = self.tfBuffer.lookup_transform("map", 'bebop2/camera_base_optical_link',
                                                           self._image_header.stamp, rospy.Duration(0.1))
                    explicit_quat = [trans.transform.rotation.x, trans.transform.rotation.y,
                                     trans.transform.rotation.z, trans.transform.rotation.w]
                    (roll, pitch, yaw) = tf_ros.transformations.euler_from_quaternion(explicit_quat)
                    # network and one circle from hough transform are similar
                    correct_circle = self.check_hough_and_net(circles, values, labels)
                    if correct_circle is not None:
                        correct_values = self.real_object_from_circle(correct_circle, maskFinal, resized_image,
                                                                      trans, roll, pitch, yaw)
                    # choose one circle from circles
                    if correct_circle is None:
                        distance = 0
                        dist = 0
                        # check if predicted circle from net is usable, add it
                        if labels == 0:
                            if len(circles) == 0:
                                circles = [[values[0], values[1], values[2] / 2]]
                            else:
                                circles = np.append(circles, [[values[0], values[1], values[2] / 2]], axis=0)
                        if len(circles) > 0:
                            for circle in circles:
                                circle_values = self.real_object_from_circle(circle, maskFinal, resized_image,
                                                                             trans, roll, pitch, yaw)
                                # circle is at least from 80% determined correct
                                if circle_values["q"] > 0.8:
                                    if self.last_position is not None:
                                        # euclidian distance
                                        dist = np.linalg.norm(self.last_position - np.array(
                                            [circle_values["x"], circle_values["y"], circle_values["z"]]))
                                    if correct_circle is None:
                                        correct_circle = circle
                                        correct_values = circle_values
                                        distance = dist
                                    else:
                                        if self.last_position is not None:
                                            if dist < distance:
                                                correct_circle = circle
                                                correct_values = circle_values
                                                distance = dist
                                        else:
                                            if circle[2] > correct_circle[
                                                2]:  # and circle_values["q"] > correct_values["q"]:
                                                correct_circle = circle
                                                correct_values = circle_values
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException) as e:
                    rospy.loginfo("No map -> camera_optical_link transformation!")
                    rospy.loginfo(e)
        return correct_circle, correct_values

    def set_transient_circle_values(self, correct_circle, correct_values):
        # print(correct_circle)
        # print(correct_values)
        # circle_mask = np.zeros(self.image.shape[0:2], dtype='uint8')
        # cv2.circle(circle_mask, (int(correct_circle[0]), int(correct_circle[1])), int(correct_circle[2]),
        #           255, -1)
        # cv2.imwrite("circle_mask.bmp", circle_mask)
        self._last_known_values = correct_values
        self._last_known_circle = correct_circle
        if correct_values is not None:
            self.last_position = np.array([correct_values["x"], correct_values["y"], correct_values["z"]])

    def publish_information_message(self):
        information = ImageTargetInfo()
        if self._last_known_values is not None:
            information.quotient = self._last_known_values["q"]
            information.centerX = self._last_known_circle[0]
            information.centerY = self._last_known_circle[1]
            information.radius = self._last_known_circle[2]
            self._last_known_information = information
            self._estimations.append(self._last_known_values)
        if self.image is not None:
            self.pub_target_info.publish(information)
        if self._last_known_values is not None:
            return information
        else:
            return None

    def publish_tf_frame(self):
        if len(self._estimations) > 0:
            buckets = []
            for i in range(len(self._estimations)):
                if i == 0:
                    buckets.append([
                        [self._estimations[i]["x"], self._estimations[i]["y"], self._estimations[i]["z"]]])
                else:
                    position = [self._estimations[i]["x"], self._estimations[i]["y"],
                                self._estimations[i]["z"]]
                    for bucket in buckets:
                        if utils.Math.is_near_enough(bucket[len(bucket) - 1], position, self._distance_limit):
                            bucket.append(position)
                        else:
                            buckets.append([[self._estimations[i]["x"], self._estimations[i]["y"],
                                             self._estimations[i]["z"]]])
            maximal_bucket = buckets[0]
            for bucket in buckets:
                if len(maximal_bucket) < len(bucket):
                    maximal_bucket = bucket
            weight_sum = 0
            position_estimation = [0, 0, 0, 0]
            for i in range(len(maximal_bucket) - 1, -1, -1):
                weight = self._weight_start - (len(maximal_bucket) - 1 - i) * self._weight_distance
                position_estimation[0] += weight * maximal_bucket[i][0]
                position_estimation[1] += weight * maximal_bucket[i][1]
                position_estimation[2] += weight * maximal_bucket[i][2]
                weight_sum += weight
            position_estimation = [
                position_estimation[0] / weight_sum,
                position_estimation[1] / weight_sum,
                position_estimation[2] / weight_sum
            ]
            v2 = [1, 0]
            if self._last_estimated_position is not None:
                v1 = [position_estimation[0] - self._last_estimated_position.transform.translation.x,
                      position_estimation[1] - self._last_estimated_position.transform.translation.y]
            else:
                v1 = v2
            v1 = utils.Math.normalize(v1)
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[0], v1[1])
            self._iteration = 0
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = self._vision_target_position_frame
            t.transform.translation.x = position_estimation[0]
            t.transform.translation.y = position_estimation[1]
            t.transform.translation.z = position_estimation[2]
            qt = tf_ros.transformations.quaternion_from_euler(0.0, 0.0, angle)
            t.transform.rotation.x = qt[0]
            t.transform.rotation.y = qt[1]
            t.transform.rotation.z = qt[2]
            t.transform.rotation.w = qt[3]
            self._estimations.clear()
            self._last_estimated_position = t
            self.br.sendTransform(t)
            return t
        return None

    def get_new_data(self):
        self.lock = True
        if self.newest_image is not None:
            self.image = np.copy(self.newest_image)
        self.lock = False
        correct_circle, correct_values = self.get_correct_circle()
        self.set_transient_circle_values(correct_circle, correct_values)
        information = self.publish_information_message()
        tf_frame = self.publish_tf_frame()
        # print("publish", rospy.Time.now().to_sec() - start_time)
        start_time = rospy.Time.now().to_sec()
        samples_lenght, seen = self.send_camera_shot(tf_frame)
        if information is not None:
            self.information_count += 1
        if tf_frame is not None:
            self.tf_frames_count += 1
        # rospy.loginfo("Counts: " +str(self.information_count) + ", " + str(self.tf_frames_count))
        self._iteration += 1
        return information, tf_frame, samples_lenght, seen

    def nearest_possible(self, tf_frame, current_position):
        real_vector = np.array([current_position.transform.translation.x - tf_frame.transform.translation.x,
                                current_position.transform.translation.y - tf_frame.transform.translation.y])
        normalized_real_vector = utils.Math.normalize(real_vector)
        t = real_vector / normalized_real_vector
        shifted_real_vector = np.array([tf_frame.transform.translation.x, tf_frame.transform.translation.y])
        current_t = 0
        while abs(current_t - t) > self._map.box_size:
            point = self._map.get_index_on_map(shifted_real_vector[0], shifted_real_vector[1], 1)
            if self._map.is_free_on_target_map(point[0], point[1]):
                return shifted_real_vector
            shifted_real_vector += 0.5 * normalized_real_vector
            current_t += 0.5
        return np.array([tf_frame.transform.translation.x, tf_frame.transform.translation.y])

    def send_camera_shot(self, tf_frame):
        samples_lenght = 0
        seen = False
        if self._camera_id is not None or (self._camera_id is None and self.try_register_camera()):
            try:
                trans = self.tfBuffer.lookup_transform(self._map_frame, self._camera_base_link, rospy.Time())
                explicit_quat = [trans.transform.rotation.x, trans.transform.rotation.y,
                                 trans.transform.rotation.z, trans.transform.rotation.w]
                (roll, pitch, yaw) = tf_ros.transformations.euler_from_quaternion(explicit_quat)
                position = Pose()
                position.position.x = trans.transform.translation.x
                position.position.y = trans.transform.translation.y
                position.position.z = trans.transform.translation.z
                position.orientation.x = 0
                position.orientation.y = pitch
                position.orientation.z = yaw
                begin = time.time()
                coordinates = self._map.faster_rectangle_ray_tracing_3d(position, self.vfov, self.hfov,
                                                                        self.camera_range)
                # utils.Plotting.plot_map(coordinates, self._map.target_obstacle_map)
                # print("tracing", time.time() - begin)
                shot = CameraShot()
                shot.camera_id = int(self._camera_id)
                if tf_frame is not None:
                    dist = utils.Math.euclidian_distance(utils.Math.Point(position.position.x, position.position.y),
                                                         utils.Math.Point(tf_frame.transform.translation.x,
                                                                          tf_frame.transform.translation.y))
                    print("location: {}, {}".format(tf_frame.transform.translation.x,
                                                    tf_frame.transform.translation.y))
                    map_x, map_y = self._map.get_index_on_map(tf_frame.transform.translation.x,
                                                              tf_frame.transform.translation.y)
                    begin = time.time()
                    samples = utils.CameraCalculation.generate_particles_object_in_fov(
                        [map_x, map_y], self.get_location_accuracy(dist), coordinates, self._map.resolution)
                    # print("samples", time.time() - begin)
                    if self._last_tf_frame is not None:
                        velocity = utils.Math.normalize(
                            [tf_frame.transform.translation.x - self._last_tf_frame.transform.translation.x,
                             tf_frame.transform.translation.y - self._last_tf_frame.transform.translation.y])
                    else:
                        velocity = [0, 0]
                    shot.likelihood = self.current_likelihood(dist)
                    seen = True
                    rospy.loginfo("Object was seen by drone.")
                else:
                    velocity = [0, 0]
                    speed = 0.0
                    if self._drone_tf_frame is not None:
                        diff_time = np.abs(self._drone_tf_frame.header.stamp.to_sec() - trans.header.stamp.to_sec())
                        speed = np.hypot(
                            (self._drone_tf_frame.transform.translation.x - trans.transform.translation.x) / diff_time,
                            (self._drone_tf_frame.transform.translation.y - trans.transform.translation.y) / diff_time)
                    shot.likelihood = self.get_recognition_error(coordinates,
                                                                 self.get_visible_pixel_probability(speed, coordinates))
                    begin = time.time()
                    samples = utils.CameraCalculation.generate_particles_object_out_of_fov(coordinates)
                    # print("samples out", time.time() - begin)
                    seen = False
                    rospy.loginfo("Object wasn't seen by drone.")
                # position_samples = utils.CameraCalculation.generate_position_samples(samples, velocity, np.pi / 4)
                samples_lenght = len(samples)
                pointcloud = utils.DataStructures.array_to_pointcloud2(samples, rospy.Time.now(), self._map_frame)
                shot.pointcloud = pointcloud
                self._drone_tf_frame = trans
                self._pub_shot.publish(shot)
            except Exception as e:
                traceback.print_exc()
                rospy.loginfo("Problem with drone camera shot. " + str(e))
        self._last_tf_frame = tf_frame
        return samples_lenght, seen

    def current_likelihood(self, distance_from_target):
        return self._min_likelihood + distance_from_target * self._step_likelihood

    def current_radius(self):
        return self._radius

    def get_location_accuracy(self, distance_from_target):
        return self._min_location_accuracy + distance_from_target * self._step_location_accuracy

    def get_recognition_error(self, coordinates, visible_pixel_probability):
        return coordinates.shape[0] / visible_pixel_probability

    def get_visible_pixel_probability(self, speed, coordinates):
        pb_v = self._min_recognition_error + self._recognition_error_difference * np.clip(speed, 0, self._max_fly_speed)
        visible_pixel_probability = coordinates.shape[0] + (self._map.free_target_pixels - coordinates.shape[0]) / pb_v
        return visible_pixel_probability

    def try_register_camera(self):
        try:
            registration_camera_srv = rospy.ServiceProxy(self._camera_registration_service_name,
                                                         CameraRegistration)
            camera_id = registration_camera_srv.call(-1, False)
            if camera_id is not None:
                self._camera_id = camera_id.camera_id
            return True
        except rospy.ServiceException as se:
            rospy.loginfo("Problem with registration of drone camera: " + str(se))
            return False


class TargetPositioning(object):
    def __init__(self):
        super(TargetPositioning, self).__init__()
        self._vision_localization = VisionLocalisation()
        self._prediction_localization = PredictionLocalization()
        target_configuration = rospy.get_param("target_configuration")
        position_service = rospy.Service(target_configuration["localization"]["position_in_time_service"],
                                         PositionRequest, self.position_in_time_service)
        speed_service = rospy.Service(target_configuration["localization"]["speed_in_time_service"], PointInTime,
                                      self.speed_in_time_service)

    def position_in_time_service(self, data):
        rounded_time = utils.Math.rounding(data.header.stamp.to_sec())
        self._prediction_localization.set_default_prediction_alg(data.algorithm_index)
        position, _ = self._prediction_localization.get_position_in_time(rounded_time, renew=data.refresh)
        self._prediction_localization.set_recommended_prediction_alg()
        return [position]

    def speed_in_time_service(self, data):
        rounded_time = utils.Math.rounding(data.header.stamp.to_sec())
        speed = self._prediction_localization.get_target_speed_in_time(rounded_time)
        return [speed]

    def launch(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            start_time = rospy.Time.now().to_sec()
            information, _, samples_length, seen = self._vision_localization.get_new_data()
            start_time = rospy.Time.now().to_sec()
            self._prediction_localization.update_structure_from_information_message(information, samples_length, seen)
            self._prediction_localization.update_database_topics()
            rate.sleep()


if __name__ == "__main__":
    tp = TargetPositioning()
    tp.launch()
