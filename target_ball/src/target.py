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
from gazebo_msgs.msg import ModelState, ModelStates
from std_msgs.msg import Float64


class Target():

    def __init__(self):
        rospy.init_node('target', anonymous=False)
        self._br = tf2_ros.TransformBroadcaster()
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self._pub_set_model = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_target_state)
        rospy.Subscriber("/target_model/move_forward", Float64, self.callback_move_forward)
        rospy.Subscriber("/target_model/move_left", Float64, self.callback_move_left)
        self._target_state = None
        self._speed_left = 0.0
        self._speed_forward = 0.0
        self._index = -1

    def callback_target_state(self, data):
        if self._index == -1:
            for i in range(len(data.name)):
                if data.name[i] == "target":
                    self._index = i
                    break
        self._target_state = data.pose[self._index]

    def callback_move_forward(self, data):
        self._speed_forward = data.data

    def callback_move_left(self, data):
        self._speed_left = data.data

    def update_tf_state(self):
        if self._target_state is not None:
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = "/target/base_link"
            t.transform.translation.x = self._target_state.position.x
            t.transform.translation.y = self._target_state.position.y
            t.transform.translation.z = self._target_state.position.z
            t.transform.rotation = self._target_state.orientation
            self._br.sendTransform(t)

    def step(self, frequence):
        if self._speed_forward != 0 or self._speed_forward != 0:
            model = ModelState()
            model.model_name = "target"
            model.reference_frame = "world"
            model.pose = self._target_state
            model.pose.position.x += self._speed_forward * frequence
            model.pose.position.y += self._speed_left * frequence
            self._pub_set_model.publish(model)


    def main(self):
        r = 10.0
        rate = rospy.Rate(r)
        f = 1.0 / r
        last_time = rospy.Time.now()
        update_rate = rospy.Duration(0.5)
        while not rospy.is_shutdown():
            if rospy.Time.now() - last_time > update_rate:
                self.update_tf_state()
                last_time = rospy.Time.now()
            self.step(f)
            rate.sleep()


if __name__ == '__main__':
    target = Target()
    target.main()
