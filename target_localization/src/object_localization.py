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


lowerBound0 = np.array([0, 100, 90])
upperBound0 = np.array([10, 255, 255])
lowerBound1 = np.array([160, 100, 90])
upperBound1 = np.array([179, 255, 255])
#np.set_printoptions(threshold=np.nan)

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


class TargetLocalisation(object):

    def __init__(self):
        super(TargetLocalisation, self).__init__()
        rospy.init_node('vision', anonymous=False)
        self.focal_length = 381
        self.bridge = CvBridge()
        self.image = None
        self.pub_target_info = rospy.Publisher('/target/information', ImageTargetInfo, queue_size=10)
        rospy.Subscriber("/bebop2/camera_base/image_raw", Image, self.callback_image)
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.lock = False
        self.network = Network(1)
        self.network.construct()
        rospack = rospkg.RosPack()
        model_path = rospack.get_path('pesat_resources') + "/models/target_localization/ball_recognition"
        self.network.restore(model_path)
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

    def launch(self):
        rate = rospy.Rate(1)
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
                transposed_image = np.transpose(self.image, (1,0,2))
                print(transposed_image.shape)
                resized_image = cv2.resize(cv2.cvtColor(transposed_image, cv2.COLOR_RGB2BGR), (Network.WIDTH, Network.HEIGHT))
                labels, values = self.network.predict([resized_image])
                values = values[0]
                labels = labels[0]
                print(values, labels)
                if labels == 0:
                    # object presented in teh image
                    center = (values[0], values[1])
                    radius = values[2] / 2
                    # create mask with locality of circle
                    circle_mask = np.zeros(self.image.shape[0:2], dtype='uint8')
                    cv2.circle(circle_mask, center, radius, 255, -1)
                    # how much are red color and circle overlapping
                    and_image = cv2.bitwise_and(maskFinal, circle_mask)
                    print(cv2.countNonZero(maskFinal))
                    print(cv2.countNonZero(circle_mask))
                    cv2.imwrite("mask.bmp", maskFinal)
                    cv2.imwrite("circle.bmp", circle_mask)
                    cv2.imwrite("image.bmp", self.image)
                    area = int(np.pi * radius * radius)
                    nzCount = cv2.countNonZero(and_image)
                    quotient = 0
                    if nzCount > 0:
                        quotient = area / nzCount
                    if quotient == 0: #> 0.1:
                        # distance from focal length and average
                        distance = self.focal_length / (2 * radius)
                        # image axis
                        centerX = resized_image.shape[1] / 2  # cols
                        centerY = resized_image.shape[0] / 2  # rows
                        # shift of target in image from image axis
                        diffX = np.abs(centerX - center[0])
                        diffY = np.abs(centerY - center[1])
                        # sign for shift of camera
                        signX = np.sign(diffX)
                        signY = np.sign(diffY)
                        # shift angle of target in image for camera
                        angleX = np.arctan2(diffX, self.focal_length)
                        angleY = np.arctan2(diffY, self.focal_length)
                        # direction of camera in drone_position frame
                        try:
                            trans = self.tfBuffer.lookup_transform("map", 'bebop2/camera_base_optical_link',
                                                                   rospy.Time())
                            explicit_quat = [trans.transform.rotation.x, trans.transform.rotation.y,
                                             trans.transform.rotation.z, trans.transform.rotation.w]
                            (r, p, y) = tf_ros.transformations.euler_from_quaternion(explicit_quat)
                            vector = [0, 0, 1]
                            # direction of camera
                            vector = self.rotateY(vector, y + signY * angleY)
                            vector = self.rotateX(vector, p + signX * angleX)
                            z = trans.transform.translation.z + distance * vector[2]
                            x = trans.transform.translation.x + distance * vector[0]
                            y = trans.transform.translation.y + distance * vector[1]
                            t = TransformStamped()
                            t.header.stamp = rospy.Time.now()
                            t.header.frame_id = "map"
                            t.child_frame_id = "tarloc/target"
                            t.transform.translation.x = x
                            t.transform.translation.y = y
                            t.transform.translation.z = z
                            t.transform.rotation.w = 1
                            self.br.sendTransform(t)
                            information.quotient = quotient
                            information.centerX = center[0]
                            information.centerY = center[1]
                            information.radius = radius
                        except (
                                tf2_ros.LookupException, tf2_ros.ConnectivityException,
                                tf2_ros.ExtrapolationException):
                            rospy.loginfo("No map -> camera_optical_link transformation!")
            self.pub_target_info.publish(information)
            self.image = None
            self.lock = False
            rate.sleep()


if __name__ == "__main__":
    tl = TargetLocalisation()
    tl.launch()
