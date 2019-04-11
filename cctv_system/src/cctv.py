#!/usr/bin/env python
import rospy
from collections import namedtuple
import threading
import tf2_ros
import numpy as np
from std_msgs.msg import Empty
import helper_pkg.utils as utils

from pesat_msgs.msg import Notification, CameraShot, CameraUpdate
from pesat_msgs.srv import CameraRegistration

cam = namedtuple("Camera", ["center", "end", "start", "radius_squared", "mi", "sigma", "id", "default"])
active_cam = namedtuple("ActiveCamera", ["last_position", "serial_number", "likelihood"])


class Cctv():
    def __init__(self):
        self._cameras = []
        self._switched = True
        self._rate = rospy.Rate(10)
        self._active_cameras = {}
        self._serial_number = 0
        self._pub_notify = rospy.Publisher("/cctv/notifications", Notification, queue_size=10)
        rospy.Subscriber("/cctv/switch", Empty, self.callback_switch)
        rospy.Subscriber("/cctv/shot", CameraShot, self.callback_shot)
        rospy.Subscriber("/cctv/update", CameraUpdate, self.callback_update)
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self._s = rospy.Service('/cctv/registration', CameraRegistration, self.camera_registration)
        self._sn_lock = threading.Lock()
        self._id_lock = threading.Lock()

    def camera_registration(self, req):
        camera = [req.center.x, req.center.y, req.direction, req.camera_range, req.radius, req.mi, req.sigma]
        camera_id = self.add_camera(camera)
        return CameraRegistration(camera_id)

    def callback_shot(self, data):
        camera = self._cameras[data.camera_id]
        sn = self.save_serial_number()
        lik = np.random.normal(camera.mi, camera.sigma)
        notification = Notification()
        notification.serial_number = sn
        notification.likelihood = lik
        notification.x = camera.default.x
        notification.y = camera.default.y
        notification.velocity_x = data.velocity_x
        notification.velocity_y = data.velocity_y
        self._pub_notify.publish(notification)

    def callback_update(self, data):
        default = utils.Math.Point(data.default.x, data.default.y)
        center = utils.Math.Point(data.center.x, data.center.y)
        start = utils.Math.Point(data.start.x, data.start.y)
        end = utils.Math.Point(data.end.x, data.end.y)
        radius_squared = data.radius * data.radius
        self._cameras[data.camera_id] = cam(center, end, start, radius_squared, data.mi,
                                            data.sigma, data.camera_id, default)

    def save_serial_number(self):
        with self._sn_lock:
            sn = self._serial_number
            self._serial_number += 1
        return sn

    def save_camera_id(self):
        with self._id_lock:
            camera_id = len(self._cameras)
        return camera_id

    def add_camera(self, camera):
        x, y, direction, camera_range, radius, mi, sigma = camera[0], camera[1], camera[2], camera[3], camera[4], \
                                                           camera[5], camera[6]
        end_angle = direction + camera_range / 2.0
        start_angle = direction - camera_range / 2.0
        v1 = utils.Math.cartesian_coor(end_angle, radius)
        v2 = utils.Math.cartesian_coor(start_angle, radius)
        if utils.Math.areClockwise(v1, v2):
            end_arm, start_arm = v1, v2
        else:
            end_arm, start_arm = v2, v1
        default = utils.Math.cartesian_coor(direction, radius / 2.0)
        camera_id = self.save_camera_id()
        c = cam(utils.Math.Point(x, y), end_arm, start_arm, radius * radius, mi, sigma, camera_id, default)
        self._cameras.append(c)
        return camera_id

    def read_cameras_from_file(self, file_path):
        with open(file_path, "r") as f:
            for line in f:
                self.add_camera(line.split(","))

    def callback_switch(self, _):
        if self._switched:
            self.rate = rospy.Rate(0.02)
            self._switched = False
        else:
            self.rate = rospy.Rate(10)
            self._switched = True

    def clear_active_cameras(self, current_active):
        for key in self._active_cameras:
            if key not in current_active:
                self._active_cameras[key] = None

    def check_cameras(self):
        if self._switched:
            try:
                trans = self._tfBuffer.lookup_transform("map", 'target/base_link', rospy.Time())
                target = utils.Math.Point(trans.transform.translation.x, trans.transform.translation.y)
                activated = {}
                for camera in self._cameras:
                    if utils.Math.is_inside_sector(target, camera):
                        if camera.id in self._active_cameras:
                            notification = Notification()
                            ac = self._active_cameras[camera.id]
                            notification.serial_number = ac.serial_number
                            notification.likelihood = ac.likelihood
                            notification.x = self._cameras[camera.id].default.x
                            notification.y = self._cameras[camera.id].default.y
                            notification.velocity_x = target.x - ac.last_position.x
                            notification.velocity_y = target.y - ac.last_position.y
                            self._pub_notify.publish(notification)
                            self._active_cameras[camera.id] = active_cam(utils.Math.Point(target.x, target.y),
                                                                         ac.serial_number, ac.likelihood)
                        else:
                            sn = self.save_serial_number()
                            lik = np.random.normal(camera.mi, camera.sigma)
                            self._active_cameras[camera.id] = active_cam(utils.Math.Point(target.x, target.y), sn, lik)
                        activated[camera.id] = True
                self.clear_active_cameras(activated)
            except Exception as e:
                print(e)

    def main(self):
        self.read_cameras_from_file(".")
        while not rospy.is_shutdown():
            self.check_cameras()
            self.rate.sleep()

if __name__ == '__main__':
    cctv = Cctv()
    cctv.main()
