#!/usr/bin/env python
import rospy
import os
from collections import namedtuple
import threading
import tf2_ros
import numpy as np
import helper_pkg.utils as utils
from pesat_msgs.msg import Notification, CameraShot
from pesat_msgs.srv import CameraRegistration
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
import scipy.stats

camera_definition = namedtuple("Camera", ["id", "coordinates", "location_accuracy", "recognition_error"])
active_cam = namedtuple("ActiveCamera", ["last_position", "likelihood"])
camera_shot = namedtuple("CameraShot", ["positions", "likelihood"])


class CameraServer():
    def __init__(self):
        environment_configuration = rospy.get_param("environment_configuration")
        self._serial_numbers = {}
        self._stable_cameras = {}
        self._movable_cameras = {}
        self._last_shots = {}
        self._rate = rospy.Rate(10)
        self._serial_number = 0
        self._camera_id = 0
        self._dead_camera_timeout = 2  # seconds
        self._moveable_camera_update_timeout = 1  # seconds
        self._map_file = environment_configuration["map"]["obstacles_file"]
        self._map = utils.Map.get_instance(self._map_file)
        self._obstacles_map = np.copy(self._map.target_obstacle_map)
        self._free_map = 1 - self._obstacles_map
        self._obstacles_map[self._obstacles_map > 0] = 1
        self._obstacles_pixels = np.sum(self._obstacles_map[self._obstacles_map == 0])
        self._camera_step = 1 / (self._map.resolution + 1)
        self._map_frame = environment_configuration["map"]["frame"]
        self._pub_notify = rospy.Publisher(environment_configuration["watchdog"]["camera_notification_topic"],
                                           PointCloud2, queue_size=10)
        rospy.Subscriber(environment_configuration["watchdog"]["camera_shot_topic"], CameraShot, self.callback_shot)
        self._s = rospy.Service(environment_configuration["watchdog"]["camera_registration_service"],
                                CameraRegistration, self.camera_registration)
        self._sn_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._shot_processing = threading.Lock()

    def camera_registration(self, req):
        if req.camera_id == -1:
            camera_id = self.add_camera(req)
        else:
            camera_id = self.delete_camera(req.camera_id)
        return [camera_id]

    def callback_shot(self, data):
        with self._shot_processing:
            self._last_shots[data.camera_id] = [rospy.Time.now().to_sec(),
                                                camera_shot(data.pointcloud, data.likelihood)]

    def send_sensor_model(self):
        import time as t
        header = Header()
        header.stamp = rospy.Time.now()
        header.seq = self.save_serial_number()
        header.frame_id = self._map_frame
        cameras_shots = np.zeros((self._camera_id, self._map.width, self._map.height))
        tiles = self._map.width * self._map.height
        with self._shot_processing:
            for key in range(self._camera_id):
                if key in self._last_shots:
                    time = self._last_shots[key][0]
                    shot = self._last_shots[key][1]
                    if abs(time - rospy.Time.now().to_sec()) < self._dead_camera_timeout:
                        camera_activated = False
                        if key in self._movable_cameras:
                            camera_activated = self._movable_cameras[key][1]
                        elif key in self._stable_cameras:
                            camera_activated = self._stable_cameras[key]
                        if camera_activated:
                            probs_map = utils.CameraCalculation.particles_to_image(
                                utils.DataStructures.pointcloud2_to_array(shot.positions), self._map.width,
                                self._map.height)
                            predicted_pixels_count = np.count_nonzero(probs_map)
                            if predicted_pixels_count > 0:
                                tiles_outside = tiles - self._obstacles_pixels - predicted_pixels_count
                                rest_probs = 1.0 - shot.likelihood
                                rest_probs /= tiles_outside
                                mask = np.bitwise_and(self._obstacles_map == 1, probs_map == 0)
                                cameras_shots[key, mask] = rest_probs
                                probs_map_mask = probs_map > 0
                                cameras_shots[key, probs_map_mask] = probs_map[probs_map_mask] * shot.likelihood
                                continue
                cameras_shots[key, :] = np.copy(self._obstacles_map)
        if cameras_shots.shape[0] == 0:
            shots_product = np.copy(self._obstacles_map)
        else:
            shots_product = cameras_shots.prod(0)
        shots_product = np.nan_to_num(shots_product)
        shots_sum = np.sum(shots_product)
        nonnegative = np.count_nonzero(shots_product >= 0) == shots_product.size
        if shots_sum == 0 or not nonnegative:
            shots_product = np.copy(self._obstacles_map)
            shots_sum = np.sum(shots_product)
        shots_product /= shots_sum
        #utils.Plotting.plot_probability(shots_product, "tracking_test/probability/shots_{}.png".format(rospy.Time.now().to_sec()))
        begin = t.time()
        pointcloud_msg = utils.DataStructures.array_to_pointcloud2(
            shots_product)  # , tiles, 1.0 / self._map.resolution, 0.5, np.pi))
        pointcloud_msg.header = header
        self._pub_notify.publish(pointcloud_msg)

    def save_serial_number(self):
        with self._sn_lock:
            sn = self._serial_number
            self._serial_number += 1
        return sn

    def add_camera(self, req):
        with self._id_lock:
            camera_id = self._camera_id
            self._camera_id += 1
        if req.stable:
            self._stable_cameras[camera_id] = True
        else:
            self._movable_cameras[camera_id] = [rospy.Time.now().to_sec(), True]
        return camera_id

    def delete_camera(self, camera_id):
        if camera_id in self._stable_cameras:
            self._stable_cameras = False
        elif camera_id in self._movable_cameras:
            self._movable_cameras[camera_id] = [rospy.Time.now().to_sec(), False]
        else:
            return -1
        return camera_id


class Cctv():
    def __init__(self):
        self._cameras = {}
        self._active_cameras = {}
        environment_configuration = rospy.get_param("environment_configuration")
        target_configuration = rospy.get_param("target_configuration")
        self._target_base_link_frame = target_configuration["properties"]["base_link"]
        self._map_frame = environment_configuration["map"]["frame"]
        self._map_file = environment_configuration["map"]["obstacles_file"]
        self._map = utils.Map.get_instance(self._map_file)
        self._file_path = environment_configuration["watchdog"]["cameras_file"]
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self.read_cameras_from_file(self._file_path)
        self._cameras_in_coordinates = {}
        self.registration_camera_srv = rospy.ServiceProxy(
            environment_configuration["watchdog"]["camera_registration_service"],
            CameraRegistration)
        self._pub_shot = rospy.Publisher(environment_configuration["watchdog"]["camera_shot_topic"], CameraShot,
                                         queue_size=10)

    def read_cameras_from_file(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    self.add_camera(line.split(","))

    def add_camera(self, camera):
        position = Pose()
        position.position.x, position.position.y, position.orientation.z = camera[0], camera[1], camera[2]
        position.orientation.y, camera_range, vfov, hfov = camera[3], camera[4], camera[5], camera[6]
        location_accuracy, recognition_error = camera[7], camera[8]
        camera_id = self.registration_camera_srv(-1, True)
        coordinates = self._map.faster_rectangle_ray_tracing_3d(position, vfov, hfov, camera_range)
        for item in coordinates:
            if item[0] not in self._cameras_in_coordinates:
                self._cameras_in_coordinates[item[0]] = {}
            if item[1] not in self._cameras_in_coordinates[item[0]]:
                self._cameras_in_coordinates[item[0]][item[1]] = {}
            self._cameras_in_coordinates[item[0]][item[1]][camera_id] = True
        c = camera_definition(camera_id, coordinates, location_accuracy, recognition_error)
        self._cameras[camera_id] = c

    def clear_active_cameras(self, current_active):
        for key in self._active_cameras:
            if key not in current_active:
                self._active_cameras[key] = None

    def check_cameras(self):
        try:
            trans = self._tfBuffer.lookup_transform(self._map_frame, self._target_base_link_frame, rospy.Time())
            p = self._map.map_point_from_real_coordinates(trans.transform.translation.x, trans.transform.translation.y,
                                                          trans.transform.translation.z)
            in_cameras = {}
            if p.x in self._cameras_in_coordinates:
                if p.y in self._cameras_in_coordinates[p.x]:
                    in_cameras = self._cameras_in_coordinates[p.x][p.y]
            activated = {}
            for camera_id in self._cameras:
                camera = self._cameras[camera_id]
                shot = CameraShot()
                shot.camera_id = camera.id
                if camera_id in in_cameras:
                    if camera.id in self._active_cameras:
                        ac = self._active_cameras[camera.id]
                        velocity = utils.Math.normalize(
                            [p.real_x - ac.last_position.real_x, p.real_y - ac.last_position.real_y])
                        shot.likelihood = ac.likelihood
                        self._active_cameras[camera.id] = active_cam(p, ac.likelihood)
                    else:
                        mean = 1 - camera.recognition_error
                        var = mean - camera.recognition_error
                        current_uncertainity = np.random.normal(mean, var)
                        lik = abs(mean - current_uncertainity)
                        velocity = [0, 0]
                        self._active_cameras[camera.id] = active_cam(p, lik)
                    samples = utils.CameraCalculation.generate_particles_object_in_fov([p.real_x, p.real_y],
                                                                                       camera.location_accuracy,
                                                                                       camera.coordinates,
                                                                                       self._map.resolution)
                    activated[camera.id] = True
                else:
                    velocity = [0, 0]
                    shot.likelihood = camera.recognition_error
                    samples = utils.CameraCalculation.generate_particles_object_out_of_fov(camera.coordinates)
                position_samples = utils.CameraCalculation.generate_position_samples(samples, velocity, np.pi / 4)
                pointcloud = utils.DataStructures.array_to_pointcloud2(position_samples, rospy.Time.now(),
                                                                       self._map_frame)
                shot.pointcloud = pointcloud
                self._pub_shot.publish(shot)
            self.clear_active_cameras(activated)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    rospy.init_node('camera_server', anonymous=False)
    rate = rospy.Rate(10)
    camera_server = CameraServer()
    cctv = Cctv()
    while not rospy.is_shutdown():
        cctv.check_cameras()
        camera_server.send_sensor_model()
        rate.sleep()
