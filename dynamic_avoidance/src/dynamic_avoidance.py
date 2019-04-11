#!/usr/bin/env python
import rospy
from collections import deque, namedtuple
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Bool, Float32
import dronet_perception.msg as dronet_perception


class DynamicAvoider(object):

    def __init__(self):
        super(DynamicAvoider, self).__init__()
        rospy.init_node('dynamic_avoidance', anonymous=False)
        self.params = rospy.get_param("dynamic_avoidance")
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        rospy.Subscriber("/dynamic_avoidance/switch", Bool, self.callback_switch)
        rospy.Subscriber("/dynamic_avoidance/alert", Bool, self.callback_alert)
        self._pub_is_avoiding = rospy.Publisher("/dynamic_avoidance/is_avoiding", Bool, queue_size=10)
        self._pub_collision = rospy.Publisher("/dynamic_avoidance/collision", Float32, queue_size=10)
        self._pub_centering = rospy.Publisher("/dynamic_avoidance/centering", Twist, queue_size=10)
        self._pub_cmd_vel = rospy.Publisher(self.params["cmd_vel_topic"], Twist, queue_size=10)
        self._ison = False
        self._alert = False
        self._low_bound = self.params["low_bound"] # 0.7
        self.init_dronet()

    def init_dronet(self):
        self.pub_dronet = rospy.Publisher(self.params['state_change_topic'], Bool, queue_size=10)
        self.dronet_cmd_vel = None
        self.dronet_prediction = None
        rospy.Subscriber(self.params["dronet_cmd_vel_topic"], Twist, self.callback_dronet_vel)
        rospy.Subscriber(self.params["cnn_prediction_topic"], dronet_perception.CNN_out,
                         self.callback_dronet_prediction)
        self.pub_dronet.publish(False)

    # callbacks
    def callback_dronet_vel(self, data):
        self.dronet_cmd_vel = data

    def callback_dronet_prediction(self, data):
        if self.dronet_prediction.collision_prob > self._low_bound:
            self._pub_collision.publish(self.dronet_prediction.collision_prob)
        self.dronet_prediction = data

    def callback_switch(self, data):
        if data:
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
            trans = self.tfBuffer.lookup_transform(self.params["camera_base_link"], self.params['camera_optical_link'],
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
            trans_last = self.tfBuffer.lookup_transform(self.params["map"], self.params['base_link'], rospy.Time())
            trans = self.tfBuffer.lookup_transform(self.params["map"], self.params['base_link'],
                                                   trans.header.stamp - 0.01)
            vel_x = trans_last.transform.translation.x - trans.transform.translation.x
            vel_y = trans_last.transform.translation.y - trans.transform.translation.y
            angle = np.arctan2(vel_y / vel_x)
            twist.angular.z = angle
        except:
            pass
        return Twist

    def velocity_command(self):
        return self.dronet_cmd_vel, self.dronet_prediction.collision_prob > self._low_bound

    def step(self):
        if self._ison:
            cmd_vel, is_avoiding = self.velocity_command()
            if is_avoiding:
                self._pub_is_avoiding.publish(True)
                self._pub_cmd_vel.publish(self.velocity_command())
            else:
                self._pub_is_avoiding.publish(False)
        else:
            if self._alert:
                self._pub_centering.publish(self.centering())


if __name__ == "__main__":
    da = DynamicAvoider()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        da.step()
        rate.sleep()
