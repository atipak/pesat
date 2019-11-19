#!/usr/bin/env python
import rospy
import tf as tf_ros
import tf2_ros
import numpy as np


# notes
# FOV_Horizontal = 2 * atan(W/2/f) = 2 * atan2(W/2, f)  radians
# FOV_Vertical   = 2 * atan(H/2/f) = 2 * atan2(H/2, f)  radians
# FOV_Diagonal   = 2 * atan2(sqrt(W^2 + H^2)/2, f)


class Test(object):

    def __init__(self):
        super(Test, self).__init__()
        rospy.init_node('vision', anonymous=False)
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self._iteration = 0
        self._speed_x = []
        self._speed_y = []
        self._position_x = []
        self._position_y = []
        self._position_z = []
        self._angle_yaw = []
        self._last_estimated_time = 0
        self._last_estimated = None
        self._last_real = None

    def print_result(self, count):
        speeds_x = self._speed_x[len(self._speed_x) - count:len(self._speed_x)]
        speeds_y = self._speed_y[len(self._speed_y) - count:len(self._speed_y)]
        positions_x = self._position_x[len(self._position_x) - count:len(self._position_x)]
        positions_y = self._position_y[len(self._position_y) - count:len(self._position_y)]
        positions_z = self._position_z[len(self._position_z) - count:len(self._position_z)]
        angles_yaw = self._angle_yaw[len(self._angle_yaw) - count:len(self._angle_yaw)]
        print("Count:", count)
        print("Speed")
        print("X:: average:", np.average(speeds_x), "var:", np.var(speeds_x))
        print("Y:: average:", np.average(speeds_y), "var:", np.var(speeds_y))
        print("Position")
        print("X:: average:", np.average(positions_x), "var:", np.var(positions_x))
        print("Y:: average:", np.average(positions_y), "var:", np.var(positions_y))
        print("Z:: average:", np.average(positions_z), "var:", np.var(positions_z))
        print("Orientation")
        print("Yaw:: average:", np.average(angles_yaw), "var:", np.var(angles_yaw))

    def launch(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                estimated = self.tfBuffer.lookup_transform("map", 'tarloc/target', rospy.Time())
                if estimated.header.stamp.to_sec() != self._last_estimated_time:
                    self._last_estimated_time = estimated.header.stamp.to_sec()
                    real = self.tfBuffer.lookup_transform("map", 'target/base_link', rospy.Time())
                    if real.header.stamp < estimated.header.stamp:
                        estimated = self.tfBuffer.lookup_transform("map", 'tarloc/target', real.header.stamp)
                    else:
                        real = self.tfBuffer.lookup_transform("map", 'target/base_link', estimated.header.stamp)
                    self._position_x.append(real.transform.translation.x - estimated.transform.translation.x)
                    self._position_y.append(real.transform.translation.y - estimated.transform.translation.y)
                    self._position_z.append(real.transform.translation.z - estimated.transform.translation.z)
                    explicit_quat_real = [real.transform.rotation.x, real.transform.rotation.y,
                                          real.transform.rotation.z, real.transform.rotation.w]
                    explicit_quat_estimated = [estimated.transform.rotation.x, estimated.transform.rotation.y,
                                               estimated.transform.rotation.z, estimated.transform.rotation.w]
                    (_, _, real_yaw) = tf_ros.transformations.euler_from_quaternion(explicit_quat_real)
                    (_, _, estimated_yaw) = tf_ros.transformations.euler_from_quaternion(explicit_quat_estimated)
                    self._angle_yaw.append(real_yaw - estimated_yaw)
                    if self._last_estimated is not None and self._last_real is not None:
                        speed_real_x = real.transform.translation.x - self._last_real.transform.translation.x
                        speed_estimated_x = estimated.transform.translation.x - self._last_estimated.transform.translation.x
                        speed_real_y = real.transform.translation.y - self._last_real.transform.translation.y
                        speed_estimated_y = estimated.transform.translation.y - self._last_estimated.transform.translation.y
                        speed_x = speed_real_x - speed_estimated_x
                        speed_y = speed_real_y - speed_estimated_y
                    else:
                        speed_x = 0
                        speed_y = 0
                    self._speed_x.append(speed_x)
                    self._speed_y.append(speed_y)
                    self._last_estimated = estimated
                    self._last_real = real
                    self._iteration += 1
                    if self._iteration % 10 == 0:
                        self.print_result(10)
                    if self._iteration % 100 == 0:
                        self.print_result(100)
                    if self._iteration % 1000 == 0:
                        self.print_result(1000)
            except Exception as e:
                print(str(e))
            rate.sleep()


if __name__ == "__main__":
    test = Test()
    test.launch()
