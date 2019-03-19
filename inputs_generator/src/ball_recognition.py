#!/usr/bin/env python
import rospy
import pickle
import cv2
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelState
from cv_bridge import CvBridge, CvBridgeError

lowerBound0 = np.array([0, 100, 90])
upperBound0 = np.array([10, 255, 255])
lowerBound1 = np.array([160, 100, 90])
upperBound1 = np.array([179, 255, 255])
np.set_printoptions(threshold=np.nan)

kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

font = cv2.FONT_HERSHEY_SIMPLEX


class TargetLocalisation(object):

    def __init__(self):
        super(TargetLocalisation, self).__init__()
        rospy.init_node('vision', anonymous=False)
        self.focal_length_distance = 962.5
        self.focal_length_horizontal = 371
        self.focal_length_vertical = 385
        self.bridge = CvBridge()
        self.image = None
        self.pub_set_model = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        rospy.Subscriber("/bebop2/camera_base/image_raw", Image, self.callback_image)
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.lock = False
        self.ball_properties = {"size": 1}
        self.last_image = None
        self.height = 1200  # 480
        self.width = 2140  # 856
        self.image_center_x = int(self.width / 2)
        self.image_center_y = int(self.height / 2)
        self.pixels = self.height * self.width
        self.required_width = 1280
        self.required_height = 720
        self.required_center_x = self.required_width / 2
        self.required_center_y = self.required_height / 2
        self.output_width = 856
        self.output_height = 480
        self.x_ratio = self.output_width / float(self.required_width)
        self.y_ratio = self.output_height / float(self.required_height)
        self.ratio = np.average(np.array([self.x_ratio, self.y_ratio]))
        # self.focal_length = rospy.get_param("focal_length")

    def callback_image(self, data):
        try:
            if not self.lock:
                self.image = cv2.cvtColor(self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough"),
                                          cv2.COLOR_RGB2BGR)
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

    def shift_and_rotate_points(self, axis, points, shift, angle):
        changed_points = []
        s = np.sin(angle)
        c = np.cos(angle)
        for p in points:
            norm__p = [p[0] - axis[0], p[1] - axis[1]]
            rotated_p = [norm__p[0] * c - norm__p[1] * s, norm__p[0] * s + norm__p[1] * c]
            shifted_p = [rotated_p[0] + axis[0] + shift[0], rotated_p[1] + axis[1] + shift[1]]
            changed_points.append(shifted_p)
        return changed_points

    def ball_position(self):
        ran = 30
        if np.random.randn() > 0.5:
            x = np.random.rand() * ran
        else:
            x = np.random.rand() * -ran
        if np.random.randn() > 0.5:
            y = np.random.rand() * ran
        else:
            y = np.random.rand() * -ran
        d = {"x": x, "y": y, "z": 1}
        return d

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def save_obj(self, folder, name, obj):
        with open(folder + '/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, folder, name):
        with open(folder + '/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def drone_position(self, ball_position, i):
        ran = 40
        if np.random.randn() > 0.5:
            x = np.random.rand() * ran
        else:
            x = np.random.rand() * -ran
        if np.random.randn() > 0.5:
            y = np.random.rand() * ran
        else:
            y = np.random.rand() * -ran

        x += ball_position["x"]
        y += ball_position["y"]
        z = np.random.rand() * 25
        centered_x = ball_position["x"] - x
        centered_y = ball_position["y"] - y
        centered_z = ball_position["z"] + self.ball_properties["size"] / 2 - z
        sig = np.pi * 1 / 10.0
        if np.random.randn() > 0.5:
            alpha_epsilon = np.random.rand() * sig
        else:
            alpha_epsilon = np.random.rand() * -sig
        if np.random.randn() > 0.5:
            beta_epsilon = np.random.rand() * sig
        else:
            beta_epsilon = np.random.rand() * -sig
        alpha_epsilon = 0
        beta_epsilon = 0
        norm_d = self.normalize([centered_x, centered_y, centered_z])
        ground_dist = np.sqrt(np.square(norm_d[0]) + np.square(norm_d[1]))
        alpha = np.arctan2(norm_d[1], norm_d[0]) + alpha_epsilon
        beta = np.arctan2(norm_d[2], ground_dist) + beta_epsilon
        d = {"x": x, "y": y, "z": z, "roll": 0.0001, "pitch": -beta, "yaw": alpha, "alpha_epsilon": alpha_epsilon,
             "beta_epsilon": beta_epsilon}
        return d

    def random_point_rotation(self, min_limit, max_limit, width, height):
        x = 0
        y = 0
        angle = 0
        while True:
            x = np.random.random_integers(min_limit[0], max_limit[0])
            y = np.random.random_integers(min_limit[1], max_limit[1])
            right_top = (x + width, y)
            right_bottom = (x + width, y + height)
            left_bottom = (x, y + height)
            points = [(x, y), right_top, right_bottom, left_bottom]
            p = np.random.choice([np.pi, -np.pi])
            angle = np.random.rand() * p / 2
            changed_points = self.shift_and_rotate_points((self.width / 2, self.height / 2), points, (0, 0), angle)
            point_ok = True
            for point in changed_points:
                if point[0] < 0 or point[0] > self.width:
                    point_ok = False
                    break
                if point[1] < 0 or point[1] > self.height:
                    point_ok = False
                    break
            if point_ok:
                break
        return x, y, angle

    def launch(self, samples):
        rate = rospy.Rate(3)
        sample = 0
        state = True
        ball_position = None
        drone_position = None
        information = {}
        while not rospy.is_shutdown() and sample < samples:
            self.lock = True
            if state:
                # set drone and ball
                ball_position = self.ball_position()
                ball_state = ModelState()
                ball_state.model_name = "target"
                ball_state.pose.position.x = ball_position["x"]
                ball_state.pose.position.y = ball_position["y"]
                ball_state.pose.position.z = ball_position["z"]
                ball_state.pose.orientation.w = 1
                drone_position = self.drone_position(ball_position, sample)
                drone_state = ModelState()
                drone_state.model_name = "bebop2"
                drone_state.pose.position.x = drone_position["x"]
                drone_state.pose.position.y = drone_position["y"]
                drone_state.pose.position.z = drone_position["z"]
                quat_tf = tf.transformations.quaternion_from_euler(drone_position["roll"], drone_position["pitch"],
                                                                   drone_position["yaw"])
                drone_state.pose.orientation = Quaternion(quat_tf[0], quat_tf[1], quat_tf[2], quat_tf[3])
                self.pub_set_model.publish(drone_state)
                #self.pub_set_model.publish(ball_state)
                state = False
            else:
                if self.image is not None:
                    if self.last_image is None or not np.array_equal(self.last_image, self.image):
                        (x, y, angle) = self.random_point_rotation((0, 0), (860, 480), self.required_width,
                                                                   self.required_height)
                        M = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), angle, 1)
                        im = cv2.warpAffine(self.image, M, (self.width, self.height))
                        im = im[y:y + self.required_height, x:x + self.required_width]
                        im = cv2.resize(im, (self.output_width, self.output_height))
                        # take sample
                        imgHSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
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
                        red_pixels = cv2.countNonZero(maskFinal)
                        if 20 < red_pixels < (self.pixels / 3):
                            # diff_x = np.tan(drone_position["alpha_epsilon"]) * self.focal_length_horizontal
                            # diff_y = np.tan(drone_position["beta_epsilon"]) * self.focal_length_vertical
                            # dist = np.hypot(diff_y, diff_x)
                            # d = np.hypot(self.focal_length_horizontal, dist)
                            distance = np.sqrt(np.square(ball_position["x"] - drone_position["x"]) + np.square(
                                ball_position["y"] - drone_position["y"]) + np.square(
                                ball_position["z"] - drone_position["z"]))
                            size = self.focal_length_distance / distance
                            cen = self.shift_and_rotate_points((self.image_center_x, self.image_center_y),
                                                               [[self.image_center_x, self.image_center_y]],
                                                               (-x, -y), angle)[0]
                            cen = [cen[0] * self.x_ratio, cen[1] * self.y_ratio]
                            size = size * self.ratio
                            if sample >= 5:
                                information[str(sample)] = [int(cen[0]), int(cen[1]), size, im]
                            # mask_name = "inputs/mask_" + str(sample) + ".bmp"
                            # image_name = "inputs/image_" + str(sample) + ".bmp"
                            # cv2.circle(im, (int(cen[0]), int(cen[1])), int(size), (255, 0, 255), 3)
                            # cv2.imwrite(mask_name, maskFinal)
                            # cv2.imwrite(image_name, im)
                            print(sample)
                            sample += 1
                        else:
                            information[str(sample)] = [0, 0, 0, im]
                            print(sample)
                            #image_name = "inputs/image_" + str(sample) + ".bmp"
                            #cv2.imwrite(image_name, im)
                            sample += 1
                        self.last_image = self.image.copy()
                        state = True
            self.lock = False
            rate.sleep()
        file_name = "inputs/info_{}-{}".format(0, sample)
        np.save(file_name, information)


if __name__ == "__main__":
    tl = TargetLocalisation()
    tl.launch(500)
