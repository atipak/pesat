#!/usr/bin/env python
import rospy
import cv2
import tf as tf_ros
import tf2_ros
import numpy as np
import rospkg
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
from pesat_msgs.msg import ImageTargetInfo
import helper_pkg.utils as utils

lowerBound0 = np.array([0, 100, 90])
upperBound0 = np.array([10, 255, 255])
lowerBound1 = np.array([160, 100, 90])
upperBound1 = np.array([179, 255, 255])
# np.set_printoptions(threshold=np.nan)

kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

font = cv2.FONT_HERSHEY_SIMPLEX


class Network:
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


class TargetLocalisation(object):

    def __init__(self):
        super(TargetLocalisation, self).__init__()
        rospy.init_node('vision', anonymous=False)
        _camera_configuration = rospy.get_param("drone_bebop2")["camera"]
        _vision_configuration = rospy.get_param("target_localization")
        self.hfov = _camera_configuration["hfov"]
        self.image_height = _camera_configuration["image_height"]
        self.image_width = _camera_configuration["image_width"]
        self.focal_length = self.image_width / (2.0 * np.tan(self.hfov / 2.0))
        self.vfov = 2 * np.arctan2(self.image_height / 2, self.focal_length)
        self.dfov = 2 * np.arctan2(np.sqrt(self.image_width ^ 2 + self.image_height ^ 2) / 2, self.focal_length)
        self.vangle_per_pixel = self.vfov / 480
        self.hangle_per_pixel = self.hfov / 856
        self.bridge = CvBridge()
        self.image = None
        self.pub_target_info = rospy.Publisher(_vision_configuration["target_information"], ImageTargetInfo,
                                               queue_size=10)
        rospy.Subscriber(_camera_configuration["image_channel"], Image, self.callback_image)
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.lock = False
        self.last_position = None
        self.network = Network(1)
        self.network.construct()
        rospack = rospkg.RosPack()
        model_path = rospack.get_path('pesat_resources') + "/models/target_localization/ball_recognition"
        self.network.restore(model_path)
        self._point_memory = 5
        self._localization_times = 0
        self._localization_points = None
        self._last_remembered_yaw = 0
        # self.focal_length = rospy.get_param("focal_length")

    def callback_image(self, data):
        try:
            if not self.lock:
                self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

    def rotateX(self, vector, angle):
        return [vector[0],
                vector[1] * np.cos(angle) - vector[2] * np.sin(angle),
                vector[1] * np.sin(angle) + vector[2] * np.cos(angle)]

    def rotateZ(self, vector, angle):
        return [vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
                vector[0] * np.sin(angle) + vector[1] * np.cos(angle),
                vector[2]]

    def rotateY(self, vector, angle):
        return [vector[0] * np.cos(angle) + vector[2] * np.sin(angle),
                vector[1],
                -vector[0] * np.sin(angle) + vector[2] * np.cos(angle)]

    def get_circles(self, mask):
        edges = cv2.Canny(mask, 100, 10)
        cv2.imwrite("m.bmp", mask)
        cv2.imwrite("edges.bmp", edges)
        rows = mask.shape[0]
        columns = mask.shape[1]
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=50, param2=15, minRadius=1,
                                   maxRadius=columns)
        return circles

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

    def launch(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.lock = True
            information = ImageTargetInfo()
            if self.image is not None:
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

                # predict
                transposed_image = np.transpose(self.image, (1, 0, 2))
                resized_image = cv2.resize(cv2.cvtColor(transposed_image, cv2.COLOR_RGB2BGR),
                                           (Network.WIDTH, Network.HEIGHT))
                labels, values = self.network.predict([resized_image])
                values = values[0]
                labels = labels[0]
                circles = self.get_circles(maskFinal)
                if circles is not None:
                    circles = circles[0]
                else:
                    circles = []
                correct_circle = None
                correct_values = None
                if len(circles) > 0 or labels == 0:
                    try:
                        trans = self.tfBuffer.lookup_transform("map", 'bebop2/camera_base_optical_link',
                                                               rospy.Time())
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
                if correct_circle is not None:
                    t = TransformStamped()
                    t.header.stamp = rospy.Time.now()
                    t.header.frame_id = "map"
                    t.child_frame_id = "tarloc/target"
                    t.transform.translation.x = correct_values["x"]
                    t.transform.translation.y = correct_values["y"]
                    t.transform.translation.z = correct_values["z"]
                    if self._localization_points is not None and t.header.stamp.to_sec() - self._localization_times < 1:
                        v2 = [1, 0]
                        v1 = [t.transform.translation.x - self._localization_points[0],
                              t.transform.translation.y - self._localization_points[1]]
                        v1 = utils.Math.normalize(v1)
                        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[0], v1[1])
                    else:
                        angle = self._last_remembered_yaw
                    qt = tf_ros.transformations.quaternion_from_euler(0.0, 0.0, angle)
                    t.transform.rotation.x = qt[0]
                    t.transform.rotation.y = qt[1]
                    t.transform.rotation.z = qt[2]
                    t.transform.rotation.w = qt[3]
                    self._localization_points = [t.transform.translation.x, t.transform.translation.y]
                    self._localization_times = t.header.stamp.to_sec()
                    self.br.sendTransform(t)
                    information.quotient = correct_values["q"]
                    information.centerX = correct_circle[0]
                    information.centerY = correct_circle[1]
                    information.radius = correct_circle[2]
                    # print(correct_circle)
                    # print(correct_values)
                    # circle_mask = np.zeros(self.image.shape[0:2], dtype='uint8')
                    # cv2.circle(circle_mask, (int(correct_circle[0]), int(correct_circle[1])), int(correct_circle[2]),
                    #           255, -1)
                    # cv2.imwrite("circle_mask.bmp", circle_mask)
                    self.last_position = np.array([correct_values["x"], correct_values["y"], correct_values["z"]])
            self.pub_target_info.publish(information)
            self.image = None
            self.lock = False
            rate.sleep()


if __name__ == "__main__":
    tl = TargetLocalisation()
    tl.launch()
