#!/usr/bin/env python
import rospy
import tf2_ros
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped


class Odom(object):
    def __init__(self):
        rospy.init_node('target_odom', anonymous=False)
        super(Odom, self).__init__()
        target_configuration = rospy.get_param("target_configuration")
        environment_configuration = rospy.get_param("environment_configuration")
        self._br = tf2_ros.TransformBroadcaster()
        self._target_state = None
        self._index = -1
        self._target_base_link_frame = target_configuration["properties"]["base_link"]
        self._map_frame = environment_configuration["map"]["frame"]
        self._seq_number = 0
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_target_state)
        self._pub_joint_states = rospy.Publisher('joint_states', JointState, queue_size=10)

    def callback_target_state(self, data):
        if self._index == -1:
            for i in range(len(data.name)):
                if data.name[i] == "target":
                    self._index = i
                    break
        self._target_state = data.pose[self._index]

    def update_tf_state(self):
        if self._target_state is not None:
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = self._map_frame
            t.child_frame_id = self._target_base_link_frame
            t.transform.translation.x = self._target_state.position.x
            t.transform.translation.y = self._target_state.position.y
            t.transform.translation.z = self._target_state.position.z
            t.transform.rotation = self._target_state.orientation
            self._br.sendTransform(t)


    def pub_empty_joint_states(self):
        js = JointState()
        js.header.frame_id = self._map_frame
        js.header.seq = self._seq_number
        js.header.stamp = rospy.Time.now()
        # js.position = [0.0]
        self._pub_joint_states.publish(js)
        # variables update
        self._seq_number += 1

    def main(self):
        r = 20.0
        rate = rospy.Rate(r)
        while not rospy.is_shutdown():
            self.pub_empty_joint_states()
            self.update_tf_state()
            rate.sleep()

if __name__ == '__main__':
    odom = Odom()
    odom.main()