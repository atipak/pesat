#!/usr/bin/env python
import rospy
from collections import deque, namedtuple
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose, Twist, TwistWithCovariance, TransformStamped
from std_msgs.msg import Bool, Float32
import dronet_perception.msg as dronet_perception
from pesat_msgs.srv import GoalState


class DynamicAvoider(object):

    def __init__(self):
        super(DynamicAvoider, self).__init__()
        rospy.init_node('dynamic_avoidance', anonymous=False)
        self._params = rospy.get_param("dynamic_avoidance")
        _da_configuration = rospy.get_param("dynamic_avoidance")
        _drone_configuration = rospy.get_param("drone_bebop2")["drone"]
        self._br = tf2_ros.TransformBroadcaster()
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        rospy.Subscriber(_da_configuration["switch_topic"], Bool, self.callback_switch)
        rospy.Subscriber(_da_configuration["alert_topic"], Bool, self.callback_alert)
        self._pub_is_avoiding = rospy.Publisher(_da_configuration["is_avoiding_topic"], Bool, queue_size=10)
        self._pub_collision = rospy.Publisher(_da_configuration["collision_topic"], Float32, queue_size=10)
        self._pub_centering = rospy.Publisher(_da_configuration["centering_topic"], Twist, queue_size=10)
        self._pub_recommended_altitude = rospy.Publisher(_da_configuration["recommended_altitude_topic"], Float32,
                                                         queue_size=10)
        self._s = rospy.Service(_da_configuration["next_position_service"], GoalState, self.next_position_callback)
        self._ison = False
        self._alert = False
        self._last_collision_predictions = deque(maxlen=10)
        self._avg_prob = 0
        self._low_bound = self._params["low_bound"]  # 0.7
        self.init_dronet()
        self._last_check_time = 0
        self._check_time = 5.0
        self._saved_horizontal = None
        self._saved_vertical = None
        self._maximum_camera_angle = np.pi / 6
        S = namedtuple("State",
                       ["free_move", "move_up", "unchecked_move", "reset_camera", "camera_down", "camera_back"])
        self._enum = S(0, 1, 2, 3, 4, 5)
        self._state = self._enum.free_move
        self._still_avoiding = False
        self._is_checkpoint_reached = False
        self._current_pose = None
        self._drone_position = None

    def init_dronet(self, da_config):
        self.pub_dronet = rospy.Publisher(self._params['state_change_topic'], Bool, queue_size=10)
        self.dronet_cmd_vel = None
        self.dronet_prediction = None
        rospy.Subscriber(self._params["dronet_cmd_vel_topic"], Twist, self.callback_dronet_vel)
        rospy.Subscriber(self._params["cnn_prediction_topic"], dronet_perception.CNN_out,
                         self.callback_dronet_prediction)
        self.pub_dronet.publish(False)

    # callbacks
    def next_position_callback(self, data):
        self._is_checkpoint_reached = data.last_point_reached
        self._drone_position = data.drone_position
        return [self._current_pose, False]

    def callback_dronet_vel(self, data):
        self.dronet_cmd_vel = data

    def callback_dronet_prediction(self, data):
        self._last_collision_predictions.append(data.collision_prob)
        self.dronet_prediction = data

    def callback_switch(self, data):
        if data:
            # rospy.loginfo_once("Dynamical avoidance was switched on.")
            self.pub_dronet.publish(True)
            self._alert = True
        self._ison = data

    def callback_alert(self, data):
        if data:
            self.pub_dronet.publish(True)
        elif not self._ison:
            self.pub_dronet.publish(False)
        self._alert = data


    def centering(self):
        twist = Twist()
        try:
            # camera centering
            trans = self._tfBuffer.lookup_transform(self._params["camera_base_link"],
                                                    self._params['camera_optical_link'],
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
            trans_last = self._tfBuffer.lookup_transform(self._params["map"], self._params['base_link'], rospy.Time())
            trans = self._tfBuffer.lookup_transform(self._params["map"], self._params['base_link'],
                                                    trans.header.stamp - 0.01)
            vel_x = trans_last.transform.translation.x - trans.transform.translation.x
            vel_y = trans_last.transform.translation.y - trans.transform.translation.y
            angle = np.arctan2(vel_y / vel_x)
            twist.angular.z = angle
        except Exception as e:
            print("Exception:", e)
        return twist

    def next_position(self, stay_on_place=False, camera_reset=False, camera_down=False, camera_back=False):
        next_pose = Pose()
        next_pose.position.x = self._drone_position.transform.translation.x
        next_pose.position.y = self._drone_position.transform.translation.y
        next_pose.position.z = self._drone_position.transform.translation.z
        next_pose.orientation.x = self._drone_position.transform.rotation.x
        next_pose.orientation.y = self._drone_position.transform.rotation.y
        next_pose.orientation.z = self._drone_position.transform.rotation.z
        if stay_on_place:
            return next_pose
        if camera_reset:
            next_pose.orientation.x = 0.0
            next_pose.orientation.y = 0.0
            return next_pose
        if camera_down:
            next_pose.orientation.x = 0.0
            next_pose.orientation.y = -self._maximum_camera_angle
            return next_pose
        if camera_back:
            next_pose.orientation.x = self._saved_horizontal
            next_pose.orientation.y = self._saved_vertical
            return next_pose
        if self.dronet_prediction is not None:
            rospy.loginfo_once("Dynamical avoidance predicts.")
            next_pose.position.z += 0.2
            return next_pose
        else:
            rospy.logwarn_once("No dronet prediction. Dynamical avoidance doesn't work.")
            return next_pose

    def step(self):
        self._avg_prob = np.average(self._last_collision_predictions)
        if self._alert:
            self._pub_centering.publish(self.centering())
            if self._avg_prob > self._low_bound:
                self._pub_collision.publish(self._avg_prob)
        if self._ison:
            try:
                trans = self._tfBuffer.lookup_transform(self._params["map"], self._params['base_link'],
                                                        rospy.Time())
                camera_rotation = self._tfBuffer.lookup_transform(self._params["base_link"],
                                                                  self._params['camera_base_link'], rospy.Time())
                (_, vertical, horizontal) = tf.transformations.euler_from_quaternion(camera_rotation.transform.rotation)
                is_avoiding = self._avg_prob > self._low_bound
                is_avoiding_constraint = False
                # free move ... no obstacle was recognised
                # unchecked move ... the drone is in collision-free altitude, but it has to control if there is still
                # a obstacle under it
                if self._state == self._enum.free_move or self._state == self._enum.unchecked_move:
                    # obstacle in front of camera
                    if is_avoiding:
                        self._state = self._enum.move_up
                    # check if there is time to check end of the obstacle
                    if self._state == self._enum.unchecked_move:
                        # timeout
                        if rospy.Time.now().to_sec() - self._last_check_time > self._check_time:
                            # stop drone
                            self._next_position = self.next_position(stay_on_place=True)
                            # save camera position
                            self._saved_vertical, self._saved_horizontal = vertical, horizontal
                            self._state = self._enum.reset_camera
                            is_avoiding_constraint = True
                    is_avoiding_constraint = is_avoiding
                elif self._state == self._enum.move_up:
                    # until obstacle in front of camera, move drone up
                    if is_avoiding:
                        self._next_position = self.next_position()
                        self._pub_is_avoiding.publish(True)
                    # stop moving up and change state
                    else:
                        self._next_position = self.next_position(stay_on_place=True)
                        self._state = self._enum.unchecked_move
                    is_avoiding_constraint = is_avoiding
                elif self._state == self._enum.reset_camera or self._state == self._enum.camera_down or \
                        self._state == self._enum.camera_back:
                    # reset camera into position 0,0
                    if self._state == self._enum.reset_camera:
                        self._next_position = self.next_position(camera_reset=True)
                    # move camera into position where can see obstacle under drone
                    elif self._state == self._enum.camera_down:
                        self._next_position = self.next_position(camera_down=True)
                    # move camera back
                    else:
                        self._next_position = self.next_position(camera_back=True)
                    # do until the camera is in the right position
                    if not self._is_checkpoint_reached:
                        if is_avoiding:
                            self._still_avoiding = True
                    else:
                        # change states
                        if self._state == self._enum.reset_camera:
                            self._state = self._enum.camera_down
                        elif self._state == self._enum.camera_down:
                            self._state = self._enum.camera_back
                        else:
                            self._last_check_time = rospy.Time.now().to_sec()
                            # behind an obstacle
                            if self._still_avoiding:
                                self._state = self._enum.unchecked_move
                            else:
                                self._state = self._enum.free_move
                            self._still_avoiding = False
                    is_avoiding_constraint = True
                # if there is no obstacle near drone, we don't need any restrictions
                if self._state != self._enum.free_move:
                    self._pub_recommended_altitude.publish(trans.transform.translation.z)
                else:
                    self._pub_recommended_altitude.publish(-1)
                self._pub_is_avoiding.publish(is_avoiding_constraint)

            except Exception as e:
                print("Exception:", e)
                self._pub_is_avoiding.publish(False)


if __name__ == "__main__":
    da = DynamicAvoider()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        da.step()
        rate.sleep()
