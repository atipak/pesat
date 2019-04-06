#!/usr/bin/env python
import rospy
import copy
import json
from collections import deque, namedtuple
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose, Twist, TransformStamped
from std_msgs.msg import Bool
from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
import moveit_commander
from gazebo_msgs.msg import ModelState


class Target():

    def __init__(self):
        rospy.init_node('target', anonymous=False)
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_set_model = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        rospy.Subscriber("/gazebo/get_model_state", ModelState, self.callback_taget_state)
        self.target_state = None

    def callback_target_state(self, data):
        if data.model_name == "target":
            self.target_state = data

    def update_tf_state(self):
        if self.target_state is not None:
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = "/target/base_link"
            t.transform.translation.x = self.target_state.pose.position.x
            t.transform.translation.y = self.target_state.pose.position.y
            t.transform.translation.z = self.target_state.pose.position.z
            t.transform.rotation = self.target_state.orientation
            self.br.sendTransform(t)

    def step(self):
        pass

    def main(self):
        rate = rospy.Rate(10)
        last_time = rospy.Time.now()
        update_rate = 0.5
        while not rospy.is_shutdown():
            if rospy.Time.now() - last_time > update_rate:
                self.update_tf_state()
                last_time = rospy.Time.now()
            self.step()
            rate.sleep()


if __name__ == '__main__':
    target = Target()
    target.main()
