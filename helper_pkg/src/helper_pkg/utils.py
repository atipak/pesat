import numpy as np
import rospy
from collections import namedtuple, deque
import itertools
import os
import json
from multiprocessing import Lock
from helper_pkg.building_blocks import Building
from pesat_msgs.srv import PointFloat, PosePlan
from geometry_msgs.msg import Point, Pose, PoseStamped
from sensor_msgs.msg import PointField, PointCloud2
from scipy.spatial.transform import Rotation as R
from pyoctree import pyoctree as ot
import time
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
import cv2
from scipy.stats import multivariate_normal


class Constants():
    class InputOutputParameterType:
        pose = 0
        map = 1
        pointcloud = 2

    class MapPresenceParameter:
        yes = 0
        no = 1

    class PredictionParametersNames:
        input_type = 0
        drone_positions_list = 1
        target_positions_list = 2
        map_presence = 3
        output_type = 4

    class PointCloudNames:
        X = 0
        Y = 1
        Z = 2
        ROLL = 3
        PITCH = 4
        YAW = 5
        PREDECESSOR = 6


class DataStructures():
    class SliceableDeque(deque):
        def __getitem__(self, index):
            if isinstance(index, slice):
                return type(self)(itertools.islice(self, index.start,
                                                   index.stop, index.step))
            return deque.__getitem__(self, index)

    @staticmethod
    def point_cloud_name(size=4):
        names = ["x", "y", "z", "roll", "pitch", "yaw", "predecessor"]
        fields = [PointField(names[i], i * size, PointField.FLOAT32, 1) for i in range(len(names))]
        return fields

    @staticmethod
    def array_to_pointcloud2(data, stamp=None, frame_id=None):
        size = 4
        msg = PointCloud2()
        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        msg.fields = DataStructures.point_cloud_name()
        msg.height = data.shape[0]
        msg.width = data.shape[1]
        msg.is_bigendian = False
        msg.point_step = data.shape[1]
        msg.row_step = size * data.shape[0]
        msg.is_dense = int(np.isfinite(data).all())
        msg.data = np.asarray(data, np.float32).tostring()
        return msg

    @staticmethod
    def pointcloud2_to_array(cloud_msg):
        cloud_arr = np.fromstring(cloud_msg.data, np.float32)
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

    @staticmethod
    def pointcloud2_to_pose_stamped(cloud_msg):
        array = DataStructures.pointcloud2_to_array(cloud_msg)
        pose = DataStructures.array_to_pose(array)
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header = cloud_msg.header
        return pose_stamped

    @staticmethod
    def array_to_image(data, width, height, step, center=(0, 0)):
        buckets = {}
        rounded_center = (Math.rounding(center[0]) / step, Math.rounding(center[1]) / step)
        for d in data:
            box_index_x = np.round(d[0] / step)
            box_index_y = np.round(d[1] / step)
            map_x = np.clip(box_index_x + rounded_center[0] - width / 2, 0, width - 1)
            map_y = np.clip(box_index_y + rounded_center[1] - height / 2, 0, height - 1)
            if map_x not in buckets:
                buckets[map_x] = {}
            if map_y not in buckets[map_x]:
                buckets[map_x][map_y] = 0
            buckets[map_x][map_y] += 1
        image = np.zeros((width, height))
        for x_coor in buckets:
            for y_coor in buckets[x_coor]:
                image[x_coor, y_coor] = buckets[x_coor][y_coor] / len(data)
        return image

    @staticmethod
    def array_to_pose(data):
        if len(data) == 0:
            return None
        buckets = {}
        max_coor = (Math.rounding(data[0][0]), Math.rounding(data[0][1]), Math.rounding(data[0][2]))
        max_value = 1
        for d in data:
            rounded_x = Math.rounding(d[0])
            rounded_y = Math.rounding(d[1])
            rounded_z = Math.rounding(d[2])
            if rounded_x not in buckets:
                buckets[rounded_x] = {}
            if rounded_y not in buckets[rounded_x]:
                buckets[rounded_x][rounded_y] = {}
            if rounded_z not in buckets[rounded_x][rounded_y]:
                buckets[rounded_x][rounded_y][rounded_z] = []
            buckets[rounded_x][rounded_y][rounded_z].append(d)
            if len(buckets[rounded_x][rounded_y][rounded_z]) > max_value:
                max_coor = (rounded_x, rounded_y, rounded_z)
                max_value = len(buckets[rounded_x][rounded_y][rounded_z])
        avg_result = np.average(buckets[max_coor[0]][max_coor[1]][max_coor[2]], axis=0)
        pose = Pose()
        pose.position.x = avg_result[0]
        pose.position.y = avg_result[1]
        pose.position.z = avg_result[2]
        pose.orientation.x = avg_result[3]
        pose.orientation.y = avg_result[4]
        pose.orientation.z = avg_result[5]
        return pose

    @staticmethod
    def image_to_array(image, samples_count, step, noise_position_std, noise_orientation_std, center=(0, 0)):
        new_samples = np.zeros((samples_count, 7))
        coordinates = np.reshape(np.array(np.meshgrid(np.arange(0, image.shape[0]), np.arange(0, image.shape[1]))).T,
                                 (image.shape[0] * image.shape[1], 2)) * step + np.array(center) - (
                              np.array(image.shape) / 2) * step
        samples = np.reshape(image * samples_count, (image.shape[0] * image.shape[1])).astype(np.int32)
        temp_count = np.sum(samples)
        new_samples[:temp_count, :2] = np.repeat(coordinates, samples, axis=0)
        new_samples[temp_count:, :2] = coordinates[np.random.choice(len(coordinates), samples_count - temp_count)]
        new_samples[:, :3] += np.random.normal(0, noise_position_std, (samples_count, 3))
        new_samples[:, 3:6] += np.random.normal(0, noise_orientation_std, (samples_count, 3))
        return np.array(new_samples)

    @staticmethod
    def pose_to_array(pose, samples_count, noise_position_std, noise_orientation_std):
        position_noise = np.random.normal(0, noise_position_std, (samples_count, 3))
        orientation_noise = np.random.normal(0, noise_orientation_std, (samples_count, 3))
        new_samples = []
        for i in range(samples_count):
            new_samples.append([pose.position.x + position_noise[i][0],
                                pose.position.y + position_noise[i][1],
                                pose.position.z + position_noise[i][2],
                                pose.orientation.x + orientation_noise[i][0],
                                pose.orientation.y + orientation_noise[i][1],
                                pose.orientation.z + orientation_noise[i][2],
                                0])

        return np.array(new_samples)

    @staticmethod
    def image_to_particles(image):
        particles = []
        # [left up corner x, left up corner y, 0, width, height, weight, 0]
        _, _, index = DataStructures.recursive_decomposition(image, 0, 0, image.shape[0] - 1, image.shape[1] - 1,
                                                             particles, 0)
        return np.array(particles)

    @staticmethod
    def recursive_decomposition(image, start_x, start_y, end_x, end_y, particles, index):
        if start_x > end_x or start_y > end_y:
            return 0, False, 0
        if (start_x, start_y) == (end_x, end_y):
            if image[start_x, start_y] > 0.00000001:
                return image[start_x, start_y], True, 0
            else:
                return 0, True, 0
        else:
            x_diff = end_x - start_x
            y_diff = end_y - start_y
            width_half = int(x_diff / 2)
            height_half = int(y_diff / 2)
            i = 1
            if height_half > 0:
                i += 1
            if width_half > 0:
                i += 1
            if width_half > 0 and height_half > 0:
                i += 1
            val = np.zeros(i)
            i = 0
            # left up
            val[i], same1, index1 = DataStructures.recursive_decomposition(image, start_x, start_y,
                                                                           start_x + width_half,
                                                                           start_y + height_half, particles, index)
            # left bottom
            index += index1
            if height_half > 0:
                # print("original", start_x, start_y, end_x, end_y)
                # print("added", width_half, height_half)
                # print("new", start_x, start_x + width_half, start_y + height_half + 1, end_y)
                i += 1
                val[i], same2, index1 = DataStructures.recursive_decomposition(image, start_x,
                                                                               start_y + height_half + 1,
                                                                               start_x + width_half, end_y, particles,
                                                                               index)
            else:
                rec2, same2, index1 = 0, False, 0
            # right up
            index += index1
            if width_half > 0:
                i += 1
                val[i], same3, index1 = DataStructures.recursive_decomposition(image, start_x + width_half + 1, start_y,
                                                                               end_x, start_y + height_half, particles,
                                                                               index)
            else:
                rec3, same3, index1 = 0, False, 0
            # right bottom
            index += index1
            if width_half > 0 and height_half > 0:
                i += 1
                val[i], same4, index1 = DataStructures.recursive_decomposition(image, start_x + width_half + 1,
                                                                               start_y + height_half + 1, end_x,
                                                                               end_y, particles, index)
            else:
                rec4, same4, index1 = 0, False, 0
            index += index1
            var = np.var(val)
            if not same1 or not same2 or not same3 or not same4 or var > 0.00005:
                i = 0
                index1 = 0
                if same1:
                    if val[i] != 0:
                        index1 += 1
                        particles.append([start_x, start_y, 0, width_half, height_half, val[i], 0])
                if same2:
                    i += 1
                    if val[i] != 0:
                        index1 += 1
                        particles.append(
                            [start_x, start_y + height_half + 1, 0, width_half, y_diff - height_half, val[i], 0])
                if same3:
                    i += 1
                    if val[i] != 0:
                        index1 += 1
                        particles.append(
                            [start_x + width_half + 1, start_y, 0, x_diff - width_half, height_half, val[i], 0])
                if same4:
                    i += 1
                    if val[i] != 0:
                        index1 += 1
                        particles.append(
                            [start_x + width_half + 1, start_y + height_half + 1, 0, x_diff - width_half,
                             y_diff - height_half, val[i], 0])
                return val[0], False, index + index1
            return val[0], True, index


class Math():
    Point = namedtuple("Point", ["x", "y"])
    Sector = namedtuple("Sector", ["center", "end", "start", "radius_squared"])
    Annulus = namedtuple("Annulus", ["center", "end", "start", "nearer_radius_squared", "farther_radius_squared"])

    @staticmethod
    def create_sector(position, angle, direction, radius):
        center = Math.Point(position[0], position[1])
        dir = Math.rotate_2d_vector(angle, [direction[0], direction[1]])
        start = Math.Point(dir[0], dir[1])
        dir = Math.rotate_2d_vector(-angle, [direction[0], direction[1]])
        end = Math.Point(dir[0], dir[1])
        radius_squared = radius * radius
        sector = Math.Sector(center, start, end, radius_squared)
        return sector

    @staticmethod
    def create_annulus(position, angle, direction, interval):
        center = Math.Point(position[0], position[1])
        dir = Math.rotate_2d_vector(angle, [direction[0], direction[1]])
        start = Math.Point(dir[0], dir[1])
        dir = Math.rotate_2d_vector(-angle, [direction[0], direction[1]])
        end = Math.Point(dir[0], dir[1])
        nearer_radius_squared = interval[0] * interval[0]
        farther_radius_squared = interval[1] * interval[1]
        annulus = Math.Annulus(center, start, end, nearer_radius_squared, farther_radius_squared)
        return annulus

    @staticmethod
    def rounding(value):
        value_sign = np.sign(value)
        abs_value = np.abs(value)
        diff = abs_value - int(abs_value)
        if diff < 0.25:
            r = 0.0
        elif diff < 0.75:
            r = 0.5
        else:
            r = 1
        return (int(abs_value) + r) * value_sign

    @staticmethod
    def vectorized_rounding(value):
        value_sign = np.sign(value)
        abs_value = np.abs(value)
        diff = abs_value - abs_value.astype(np.int32)
        r = np.array(len(diff))
        r[0.75 > r > 0.25] = 0.5
        r[1 > r > 0.75] = 1.0
        return abs_value.astype(np.int32) + r * value_sign

    @staticmethod
    def points_add(p1, p2):
        point = Point()
        point.x = p1.x + p2.x
        point.y = p1.y + p2.y
        point.z = p1.z + p2.z
        return point

    @staticmethod
    def points_mul(p1, p2):
        point = Point()
        point.x = p1.x * p2.x
        point.y = p1.y * p2.y
        point.z = p1.z * p2.z
        return point

    @staticmethod
    def point_constant_mul(p1, const):
        point = Point()
        point.x = p1.x * const
        point.y = p1.y * const
        point.z = p1.z * const
        return point

    @staticmethod
    def floor_rounding(value):
        rounded_value = Math.rounding(value)
        if rounded_value > value:
            return rounded_value - 0.5
        return rounded_value

    @staticmethod
    def ceiling_rounding(value):
        rounded_value = Math.rounding(value)
        if rounded_value < value:
            return rounded_value + 0.5
        return rounded_value

    @staticmethod
    def is_within_radius(p, radius_squared):
        return p.x * p.x + p.y * p.y <= radius_squared

    @staticmethod
    def is_within_interval(p, start, end):
        return start <= p.x * p.x + p.y * p.y <= end

    @staticmethod
    def is_inside_sector(p, sector):
        rel_p = Math.Point(p.x - sector.center.x, p.y - sector.center.y)
        return not Math.are_clockwise(sector.start, rel_p) and Math.are_clockwise(sector.end, rel_p) and \
               Math.is_within_radius(rel_p, sector.radius_squared)

    @staticmethod
    def is_inside_annulus(p, annulus):
        rel_p = Math.Point(p.x - annulus.center.x, p.y - annulus.center.y)
        return not Math.are_clockwise(annulus.start, rel_p) and Math.are_clockwise(annulus.end, rel_p) and \
               Math.is_within_interval(rel_p, annulus.nearer_radius_squared, annulus.farther_radius_squared)

    @staticmethod
    def cartesian_coor(angle, radius):
        return Math.Point(np.cos(angle) * radius, np.sin(angle) * radius)

    @staticmethod
    def correct_halfspace(point, rotated, axis, center, halfspace):
        centered_point = Math.Point(point.x - center.x, point.y - center.y)
        coordinates = Math.cartesian_coor(rotated, axis.x)
        if Math.are_clockwise(centered_point, coordinates) and halfspace:
            return True
        elif not Math.are_clockwise(centered_point, coordinates) and not halfspace:
            return True
        else:
            return False

    @staticmethod
    def are_clockwise(p1, p2):
        return -p1.x * p2.y + p1.y * p2.x > 0

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm < 0.0001:
            return v
        return v / norm

    @staticmethod
    def inside_ellipse(point, center, axis, rotation):
        cosa = np.cos(rotation)
        sina = np.sin(rotation)
        a_top = np.square(cosa * (point.x - center.x) + sina * (point.y - center.y))
        b_top = np.square(sina * (point.x - center.x) - cosa * (point.y - center.y))
        a_bottom = np.square(axis.x)
        b_bottom = np.square(axis.y)
        ellipse = (a_top / a_bottom) + (b_top / b_bottom)
        if ellipse <= 1:
            return True
        else:
            return False

    @staticmethod
    def two_lowest_indices(array):
        biggest_value = float("inf")
        second_biggest_value = float("inf")
        biggest_index = -1
        second_biggest_index = -1
        for i in range(len(array)):
            a = array[i]
            if a < biggest_index:
                second_biggest_value = biggest_value
                second_biggest_index = biggest_index
                biggest_value = a
                biggest_index = i
            if a < second_biggest_value:
                second_biggest_value = a
                second_biggest_index = i
        return np.array([biggest_index, second_biggest_index])

    @staticmethod
    def euclidian_distance(math_point_start, math_point_end):
        return np.sqrt(
            np.square(math_point_end.x - math_point_start.x) + np.square(math_point_start.y - math_point_end.y))

    @staticmethod
    def calculate_yaw_from_points(x_map, y_map, x_map_next, y_map_next):
        direction_x = x_map_next - x_map
        direction_y = y_map_next - y_map
        # direction_length = np.sqrt(np.power(direction_x, 2) + np.power(direction_y, 2))
        # print("direction length", direction_length)
        # if abs(direction_length) > 1:
        #    return np.arccos(1 / direction_length)
        # else:
        #    return 0
        return np.arctan2(direction_y, direction_x)

    @staticmethod
    def rotate_2d_vector(theta, vector):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        vec = np.array(vector)
        vec = np.matmul(R, vec)
        return vec

    @staticmethod
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    @staticmethod
    def rotateX(vector, angle):
        return [vector[0],
                vector[1] * np.cos(angle) - vector[2] * np.sin(angle),
                vector[1] * np.sin(angle) + vector[2] * np.cos(angle)]

    @staticmethod
    def rotateZ(vector, angle):
        return [vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
                vector[0] * np.sin(angle) + vector[1] * np.cos(angle),
                vector[2]]

    @staticmethod
    def rotateY(vector, angle):
        return [vector[0] * np.cos(angle) + vector[2] * np.sin(angle),
                vector[1],
                -vector[0] * np.sin(angle) + vector[2] * np.cos(angle)]

    @staticmethod
    def is_near_enough(position_one, position_two, distance_limit):
        point_one = Math.Point(position_one[0], position_one[1])
        point_two = Math.Point(position_two[0], position_two[1])
        dist = Math.euclidian_distance(point_one, point_two)
        return dist <= distance_limit


class Map():
    MapInstance = None
    MapLock = Lock()

    class MapPoint:
        def __init__(self):
            self._real_x = 0
            self._real_y = 0
            self._real_z = 0
            self._x, self._y, self._z = 0, 0, 0

        def set_map_coordinates(self, map_x, map_y):
            self._x, self._y = int(map_x), int(map_y)
            return self

        def set_real_coordinates(self, x, y, z):
            self._real_x = x
            self._real_y = y
            self._real_z = z
            return self

        def set_z(self, real_z):
            self._real_z = real_z

        @property
        def x(self):
            return self._x

        @property
        def y(self):
            return self._y

        @property
        def real_x(self):
            return self._real_x

        @property
        def real_y(self):
            return self._real_y

        @property
        def real_z(self):
            return self._real_z

    @classmethod
    def map_from_file(cls, file_name, resolution=2):
        if os.path.isfile(file_name):
            with open(file_name, "r") as file:
                information = json.load(file)
            try:
                m = Building.create_map_from_objects(information["objects"], information["world"]["width"],
                                                     information["world"]["height"],
                                                     information["world"]["maximal_height"], resolution)
                map = Map(m, m, max_altitude=information["world"]["maximal_height"])
                s = np.zeros(m[0].shape)
                s.fill(255)
                s[m[0] > 0] = 0
                cv2.imwrite("map_created.png", s)
                map._file_path = file_name
                return map
            except Exception as e:
                print("Exception occurred: ", e)
        else:
            print("File doesn't exist")
        ar = np.zeros((60, 60), dtype=np.uint8)
        obstacles = [ar, ar]
        map = Map(obstacles, obstacles, max_altitude=20)
        return map

    @staticmethod
    def get_map_height_width_from_objects(objects):
        width = 0
        height = 0
        try:
            for object in objects:
                object = Building.from_json_file_object_to_object(object)
                corners = Building.split_to_points(object)[0]
                for corner in corners:
                    width = np.max([width, abs(corner[0])])
                    height = np.max([height, abs(corner[1])])
        except Exception as e:
            print(e)
        return int(height), int(width)

    @staticmethod
    def load_corners_from_obstacle_file(file_path):
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                information = json.load(file)
            try:
                all_corners = []
                centers_of_rectangles = []
                mapping_corners_to_rectangles = {}
                i = -1
                for object in information["objects"]:
                    i += 1
                    object = Building.from_json_file_object_to_object(object)
                    corners = Building.split_to_points(object)[0]
                    center = Building.calculate_center_of_object(object)
                    centers_of_rectangles.append(center)
                    for corner in corners:
                        if corner[0] in mapping_corners_to_rectangles:
                            mapping_corners_to_rectangles[corner[0]][corner[1]] = i
                        else:
                            mapping_corners_to_rectangles[corner[0]] = {corner[1]: i}
                        all_corners.append(corner)
                return all_corners, centers_of_rectangles, mapping_corners_to_rectangles
            except Exception as e:
                print(e)
                return [], [], {}
        else:
            return [], [], {}

    @staticmethod
    def check_bounds(values, limits):
        all_right = True
        lower_limit = values[0]
        lower_bound = limits[0]
        if lower_limit < lower_bound:
            lower_limit = lower_bound
            all_right = False
        upper_limit = values[1]
        upper_bound = limits[1]
        if upper_limit > upper_bound:
            upper_limit = upper_bound
            all_right = False
        return lower_limit, upper_limit, all_right

    @classmethod
    def get_instance(cls, file_name=None, resolution=2):
        with cls.MapLock:
            if file_name is not None and cls.MapInstance is None:
                cls.MapInstance = cls.map_from_file(file_name, resolution)
            return cls.MapInstance

    def depth_array(self, obstacles_arrays):
        if obstacles_arrays[1] is None or obstacles_arrays[1].shape != obstacles_arrays[0].shape:
            return np.zeros(obstacles_arrays[0].shape)
        else:
            return obstacles_arrays[1]

    def __init__(self, static_obstacles_arrays, dynamic_obstacles_arrays, resolution=2, center=(0, 0),
                 max_altitude=20.0):
        self._static_obstacles_height = static_obstacles_arrays[0]
        self._static_obstacles_depth = self.depth_array(static_obstacles_arrays)
        if dynamic_obstacles_arrays is not None:
            self._dynamic_obstacles_height = dynamic_obstacles_arrays[0]
            self._dynamic_obstacles_depth = self.depth_array(dynamic_obstacles_arrays)
        else:
            self._dynamic_obstacles_height = self._static_obstacles_height
            self._dynamic_obstacles_depth = self._static_obstacles_depth
        self._width = self._static_obstacles_height.shape[0]
        self._height = self._static_obstacles_height.shape[1]
        self._resolution = resolution
        self._box_size = 1.0 / resolution
        self._map_center = center
        self._center_shift_x = self._width / 2 + self._map_center[0]
        self._center_shift_y = self._height / 2 + self._map_center[1]
        self._corners = None
        self._rectangles_centers = None
        self._corners_rectangles = None
        self._file_path = None
        self._object_type = None
        self._drone_admissibility_service = None
        self._target_admissibility_service = None
        self._drone_plan_service = None
        self._target_plan_service = None
        self._point_height_service = None
        self._tree = None
        self._map_details = None
        self._drone_min_altitude, self._drone_max_altitude = self.get_altitudes("drone")
        self._target_min_altitude, self._target_max_altitude = self.get_altitudes("target")
        self._drone_size, self._target_size = 0.5, 1
        self._maximum_random_iteration = 100
        self._pixel_per_meter = 256 / max_altitude
        self._drone_pixel_distance = self._drone_max_altitude / 256
        self._target_pixel_distance = self._target_max_altitude / 256
        self._pixels_with_obstacles = np.nonzero(
            np.bitwise_or(self._static_obstacles_height, self._dynamic_obstacles_height))
        self._target_obstacle_map = self.get_target_obstacle_map()
        self._free_target_pixels = np.count_nonzero(self._target_obstacle_map)

    def __setstate__(self, state):
        [self._static_obstacles_height, self._static_obstacles_depth] = state["static_obstacles_arrays"]
        [self._dynamic_obstacles_height, self._dynamic_obstacles_depth] = state["dynamic_obstacles_arrays"]
        resolution = state["resolution"]
        self._map_center = state["center"]
        self._pixel_per_meter = state["pixel_per_meter"]
        self._file_path = state["file_path"]
        self._width = self._static_obstacles_height.shape[0]
        self._height = self._static_obstacles_height.shape[1]
        self._resolution = resolution
        self._box_size = 1.0 / resolution
        self._center_shift_x = self._width / 2 + self._map_center[0]
        self._center_shift_y = self._height / 2 + self._map_center[1]
        self._corners = None
        self._rectangles_centers = None
        self._corners_rectangles = None
        self._file_path = None
        self._object_type = None
        self._drone_admissibility_service = None
        self._target_admissibility_service = None
        self._drone_plan_service = None
        self._target_plan_service = None
        self._point_height_service = None
        self._tree = None
        self._drone_min_altitude, self._drone_max_altitude = self.get_altitudes("drone")
        self._target_min_altitude, self._target_max_altitude = self.get_altitudes("target")
        self._drone_size, self._target_size = 0.5, 1
        self._maximum_random_iteration = 100
        self._drone_pixel_distance = self._drone_max_altitude / 256
        self._target_pixel_distance = self._target_max_altitude / 256
        self._pixels_with_obstacles = np.nonzero(
            np.bitwise_or(self._static_obstacles_height, self._dynamic_obstacles_height))
        self._target_obstacle_map = self.get_target_obstacle_map()
        self._free_target_pixels = np.count_nonzero(self._target_obstacle_map)

    def __getstate__(self):
        return {"static_obstacles_arrays": [self._static_obstacles_height, self._static_obstacles_depth],
                "dynamic_obstacles_arrays": [self._dynamic_obstacles_height, self._dynamic_obstacles_depth],
                "resolution": self.resolution, "center": self._map_center, "pixel_per_meter": self._pixel_per_meter,
                "file_path": self._file_path}

    @property
    def tree(self):
        if self._tree is None:
            point_coords = np.array([])
            connectivity = np.array([])
            with open(self._file_path) as file:
                information = json.load(file)
                point_coords = np.array(information["mesh"]["triangles"])
                connectivity = np.array(information["mesh"]["connectivity"], dtype=np.int32)
                rospy.loginfo("PyOctree was loaded from file.")
            self._tree = ot.PyOctree(point_coords, connectivity)
        return self._tree

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def real_width(self):
        return self._width * self._box_size

    @property
    def real_height(self):
        return self._height * self._box_size

    @property
    def obstacles_pixels(self):
        return self._pixels_with_obstacles

    @property
    def corners(self):
        if self._corners is None and self._file_path is not None:
            self._corners, self._rectangles_centers, self._corners_rectangles = Map.load_corners_from_obstacle_file(
                self._file_path)
        return self._corners

    @property
    def rectangles_centers(self):
        if self._rectangles_centers is None and self._file_path is not None:
            self._corners, self._rectangles_centers, self._corners_rectangles = Map.load_corners_from_obstacle_file(
                self._file_path)
        return self._rectangles_centers

    @property
    def corners_rectangles(self):
        if self._corners_rectangles is None and self._file_path is not None:
            self._corners, self._rectangles_centers, self._corners_rectangles = Map.load_corners_from_obstacle_file(
                self._file_path)
        return self._corners_rectangles

    @property
    def map(self):
        return self._static_obstacles_height

    @property
    def resolution(self):
        return self._resolution

    @property
    def target_obstacle_map(self):
        return self._target_obstacle_map

    @property
    def free_target_pixels(self):
        return self._free_target_pixels

    @property
    def box_size(self):
        return self._box_size

    @property
    def map_details(self):
        if self._map_details is None:
            with open(self._file_path, "r") as file:
                information = json.load(file)
            try:
                with open(self._file_path, "r") as file:
                    information = json.load(file)
                self._map_details = information["world"]
            except:
                print("Invalid map file path")
        return self._map_details

    def map_point_from_real_coordinates(self, real_x, real_y, real_z):
        map_point = self.MapPoint()
        map_x, map_y = self.get_index_on_map(real_x, real_y)
        map_point.set_real_coordinates(real_x, real_y, real_z)
        map_point.set_map_coordinates(map_x, map_y)
        return map_point

    def map_point_from_map_coordinates(self, map_x, map_y, real_z=0.0):
        map_point = self.MapPoint()
        map_point.set_map_coordinates(map_x, map_y)
        real_x, real_y = self.get_coordination_from_map_indices(map_point)
        map_point.set_real_coordinates(real_x, real_y, real_z)
        return map_point

    def get_target_obstacle_map(self):
        min_pixel = int(self._pixel_per_meter * self._target_min_altitude)
        max_pixel = int(self._pixel_per_meter * self._target_max_altitude + 1)
        target_obstacles_map = np.zeros((self.width, self.height))
        object_size = self.get_right_object_size("target")
        boxes_count = np.ceil((object_size / 2.0 + self._box_size / 2.0) / self._box_size)
        for x in range(self.width):
            for y in range(self.height):
                lower_limit_x, upper_limit_x, all_right_x = Map.check_bounds(
                    [int(x - boxes_count), int(x + boxes_count)], [0, self.width])
                lower_limit_y, upper_limit_y, all_right_y = Map.check_bounds(
                    [int(y - boxes_count), int(y + boxes_count)], [0, self.height])
                if not all_right_x or not all_right_y:
                    continue
                for k in range(min_pixel, max_pixel):
                    height = (1.0 / self._pixel_per_meter) * k
                    if self.is_fast_free_for_object_map(lower_limit_x, upper_limit_x, lower_limit_y, upper_limit_y,
                                                        "target", height, object_size):
                        target_obstacles_map[x, y] += 1
        target_obstacles_map /= (max_pixel - min_pixel)
        return target_obstacles_map

    def get_altitudes(self, value):
        config_string = "_configuration"
        max_altitude = 0
        min_altitude = 0
        try:
            param_string = value + config_string
            min_altitude = rospy.get_param(param_string + "/map_server/min_altitude")
            max_altitude = rospy.get_param(param_string + "/map_server/max_altitude")
        except Exception as e:
            rospy.loginfo("Object type " + value + "wasn't found. " + str(e))
        return min_altitude, max_altitude

    def load_all_services(self):
        self.load_point_height_service()
        self.drone_load_services()
        self.target_load_services()

    def load_point_height_service(self):
        config_string = "drone_configuration"
        if self._point_height_service is None:
            point_height_service = config_string + "/map_server/point_height_service"
            try:
                service_name = rospy.get_param(point_height_service)
                self._point_height_service = rospy.ServiceProxy(service_name, PointFloat)
            except rospy.ServiceException as e:
                self._point_height_service = None
                rospy.loginfo_once("Point height service loading failed: " + str(e))

    def load_admissibility_service(self, service, config_string, value):
        if service is None:
            admissibility_service = config_string + "/map_server/admissibility_service"
            try:
                service_name = rospy.get_param(admissibility_service)
                service = rospy.ServiceProxy(service_name, PointFloat)
            except rospy.ServiceException as e:
                service = None
                rospy.loginfo_once(value + " admissibility service loading failed: " + str(e))
        return service

    def load_plan_service(self, service, config_string, value):
        if service is None:
            pose_planning_service = config_string + "/map_server/pose_planning_service"
            try:
                service_name = rospy.get_param(pose_planning_service)
                service = rospy.ServiceProxy(service_name, PosePlan)
            except rospy.ServiceException as e:
                service = None
                rospy.loginfo_once(value + " pose planning service loading failed: " + str(e))
        return service

    def drone_load_services(self):
        config_string = "drone_configuration"
        self._drone_admissibility_service = self.load_admissibility_service(self._drone_admissibility_service,
                                                                            config_string, "Drone")
        self._drone_plan_service = self.load_plan_service(self._drone_plan_service, config_string, "Drone")

    def target_load_services(self):
        config_string = "target_configuration"
        self._target_admissibility_service = self.load_admissibility_service(self._target_admissibility_service,
                                                                             config_string, "Target")
        self._target_plan_service = self.load_plan_service(self._target_plan_service, config_string, "Target")

    def get_index_on_map(self, x, y):
        box_index_x = np.round(x / self._box_size)
        box_index_y = np.round(y / self._box_size)
        shifted_x = np.clip(box_index_x + self._center_shift_x, 0, self.width - 1)
        shifted_y = np.clip(box_index_y + self._center_shift_y, 0, self.height - 1)
        if isinstance(x, np.ndarray):
            return shifted_x.astype(np.int32), shifted_y.astype(np.int32)
        else:
            return int(shifted_x), int(shifted_y)

    def get_coordination_from_map_indices(self, map_point):
        shifted_x = map_point.x - self._center_shift_x
        shifted_y = map_point.y - self._center_shift_y
        real_x = shifted_x * self._box_size
        real_y = shifted_y * self._box_size * -1
        return [real_x, real_y]

    def is_fast_free_for_object_map(self, lower_limit_x, upper_limit_x, lower_limit_y, upper_limit_y, object_type,
                                    real_z, object_size):
        map_z = real_z * self._pixel_per_meter
        object_pixel_size = (object_size / 2.0) * self._pixel_per_meter
        pixels_height_bottom = int(map_z - object_pixel_size)
        pixels_height_upper = int(map_z + object_pixel_size)
        return self.fast_free(lower_limit_x, upper_limit_x, lower_limit_y, upper_limit_y, pixels_height_bottom,
                              pixels_height_upper, object_type)

    def fast_free(self, lower_limit_x, upper_limit_x, lower_limit_y, upper_limit_y, pixels_height_bottom,
                  pixels_height_upper, object_type):
        s_height = self._static_obstacles_height[lower_limit_x:upper_limit_x, lower_limit_y: upper_limit_y]
        s_depth = self._static_obstacles_depth[lower_limit_x:upper_limit_x, lower_limit_y: upper_limit_y]
        static_o = np.bitwise_or(s_height <= pixels_height_bottom, pixels_height_upper <= s_depth)
        if object_type == "target":
            d_height = self._dynamic_obstacles_height[lower_limit_x:upper_limit_x, lower_limit_y: upper_limit_y]
            d_depth = self._dynamic_obstacles_depth[lower_limit_x:upper_limit_x, lower_limit_y: upper_limit_y]
            dynamic_o = np.bitwise_or(d_height <= pixels_height_bottom, pixels_height_upper <= d_depth)
        else:
            dynamic_o = True
        return (np.bitwise_and(static_o, dynamic_o)).all()

    def is_free(self, map_point, object_type):
        object_size = self.get_right_object_size(object_type)
        pixel_distance = self.get_right_pixel_distance(object_type)
        if not 0 <= map_point.x <= self.width or not 0 <= map_point.y <= self.height:
            return False
        map_z = map_point.real_z * self._pixel_per_meter
        object_pixel_size = (object_size / 2.0) * self._pixel_per_meter
        pixels_height_bottom = int(map_z - object_pixel_size)
        pixels_height_upper = int(map_z + object_pixel_size)
        if object_type == "drone":
            return self.free_drone(map_point, pixels_height_bottom, pixels_height_upper)
        else:
            return self.free_target(map_point, pixels_height_bottom, pixels_height_upper)

    def free_target(self, map_point, pixels_height_bottom, pixels_height_upper):
        s_height = self._static_obstacles_height[map_point.x, map_point.y]
        s_depth = self._static_obstacles_depth[map_point.x, map_point.y]
        d_height = self._dynamic_obstacles_height[map_point.x, map_point.y]
        d_depth = self._dynamic_obstacles_depth[map_point.x, map_point.y]
        return (s_height <= pixels_height_bottom or pixels_height_upper <= s_depth) and (
                d_height <= pixels_height_bottom or pixels_height_upper <= d_depth)

    def free_drone(self, map_point, pixels_height_bottom, pixels_height_upper):
        s_height = self._static_obstacles_height[map_point.x, map_point.y]
        s_depth = self._static_obstacles_depth[map_point.x, map_point.y]
        return s_height <= pixels_height_bottom or pixels_height_upper <= s_depth

    def get_right_admissibility_service(self, object_type):
        if object_type == "drone":
            return self._drone_admissibility_service
        else:
            return self._target_admissibility_service

    def get_right_plan_service(self, object_type):
        if object_type == "drone":
            return self._drone_plan_service
        else:
            return self._target_plan_service

    def is_free_for_drone(self, map_point):
        return self.is_free_for_object(map_point, "drone")

    def is_free_for_target(self, map_point):
        return self.is_free_for_object(map_point, "target")

    def get_right_object_size(self, object_type):
        if object_type == "drone":
            return self._drone_size
        else:
            return self._target_size

    def get_right_pixel_distance(self, object_type):
        if object_type == "drone":
            return self._drone_pixel_distance
        else:
            return self._target_pixel_distance

    def get_right_max_altitude(self, object_type):
        if object_type == "drone":
            return self._drone_max_altitude
        else:
            return self._target_max_altitude

    def get_right_min_altitude(self, object_type):
        if object_type == "drone":
            return self._drone_min_altitude
        else:
            return self._target_min_altitude

    def is_free_for_object(self, map_point, object_type):
        self.load_all_services()
        if self.get_right_admissibility_service(object_type) is not None:
            return self.is_free_for_object_service(map_point, object_type)
        else:
            return self.is_free_for_object_map(map_point, object_type)

    def is_free_for_object_map(self, map_point, object_type):
        object_size = self.get_right_object_size(object_type)
        boxes_count = np.ceil((object_size / 2.0 + self._box_size / 2.0) / self._box_size)
        lower_limit_x, upper_limit_x, all_right_x = Map.check_bounds(
            [int(map_point.x - boxes_count), int(map_point.x + boxes_count)], [0, self.width])
        lower_limit_y, upper_limit_y, all_right_y = Map.check_bounds(
            [int(map_point.y - boxes_count), int(map_point.y + boxes_count)], [0, self.height])
        if not all_right_x or not all_right_y:
            return False
        for x_coor in range(lower_limit_x, upper_limit_x):
            for y_coor in range(lower_limit_y, upper_limit_y):
                map_point_r = self.map_point_from_map_coordinates(x_coor, y_coor, map_point.real_z)
                if not self.is_free(map_point_r, object_type):
                    return False
        return True

    def is_free_for_object_service(self, map_point, object_type):
        try:
            point = Point()
            point.x = map_point.real_x
            point.y = map_point.real_y
            point.z = map_point.real_z
            response = self.get_right_admissibility_service(object_type).call(point)
            return response > 0
        except rospy.ServiceException as exc:
            return self.is_free_for_object_map(map_point, object_type)

    def random_free_place_for_drone_in_neighbour(self, map_point, distance, object_type):
        return self.random_free_place_for_object_in_neighbour(map_point, distance, "drone")

    def random_free_place_for_target_in_neighbour(self, map_point, distance, object_type):
        return self.random_free_place_for_object_in_neighbour(map_point, distance, "target")

    def random_free_place_for_object_in_neighbour(self, map_point, distance, object_type):
        self.load_all_services()
        if self.get_right_admissibility_service(object_type) is not None:
            return self.random_free_place_for_object_in_neighbour_service(map_point, distance, object_type)
        else:
            return self.random_free_place_for_object_in_neighbour_map(map_point, distance, object_type)

    def random_free_place_for_object_in_neighbour_map(self, map_point, distance, object_type):
        boxes_count = np.floor(distance / self._box_size)
        lower_limit_x, upper_limit_x, all_right_x = Map.check_bounds(
            [int(map_point.x - boxes_count), int(map_point.x + boxes_count)], [0, self.width])
        lower_limit_y, upper_limit_y, all_right_y = Map.check_bounds(
            [int(map_point.y - boxes_count), int(map_point.y + boxes_count)], [0, self.height])
        for x_coor in range(lower_limit_x, upper_limit_x):
            for y_coor in range(lower_limit_y, upper_limit_y):
                probe_map_point = self.map_point_from_map_coordinates(x_coor, y_coor, map_point.real_z)
                if self.is_free_for_object(probe_map_point, object_type):
                    return probe_map_point
        return None

    def random_free_place_for_object_in_neighbour_service(self, map_point, distance, object_type):
        lower_x, higher_x, _ = Map.check_bounds([int(map_point.real_x - distance), int(map_point.real_x + distance)],
                                                [0, self.real_width])
        lower_y, higher_y, _ = Map.check_bounds([int(map_point.real_y - distance), int(map_point.real_y + distance)],
                                                [0, self.real_height])
        i = 0
        while i < self._maximum_random_iteration:
            x = np.random.uniform(lower_x, higher_x)
            y = np.random.uniform(lower_y, higher_y)
            if self.get_right_min_altitude(object_type) == self.get_right_max_altitude(object_type):
                z = self.get_right_max_altitude(object_type)
            else:
                z = np.random.uniform(self.get_right_min_altitude(object_type),
                                      self.get_right_max_altitude(object_type))
            map_point = self.map_point_from_real_coordinates(x, y, z)
            if self.is_free_for_object(map_point, object_type):
                return map_point
            i += 1
        return None

    def crop_map(self, center, width, height):
        center = [center[0], center[1]]
        if width > self.width:
            width = self.width
        if height > self.height:
            height = self.height
        half_width = width / 2
        half_height = height / 2
        if center[0] - half_width < 0:
            center[0] = half_width
        if center[0] + half_width > self.width:
            center[0] = self.width - half_width
        if center[1] - half_height < 0:
            center[1] = half_height
        if center[1] + half_height > self.height:
            center[1] = self.height - half_height
        return self._static_obstacles_height[center[0] - half_width:center[0] + half_width,
               center[1] - half_height:center[1] + half_height]

    def rectangle_ray_tracing_3d(self, position, vfov, hfov, max_range):
        start_time = time.time()
        height = 0.2
        h_angles = [position.orientation.z - hfov / 2.0, position.orientation.z + hfov / 2.0]
        v_angles = np.clip([position.orientation.y - vfov / 2.0, position.orientation.y + vfov / 2.0],
                           (0, 0), (np.pi / 2, np.pi / 2))
        angles = [[v_angles[0], h_angles[0]],  # bottom left
                  [v_angles[1], h_angles[0]],  # upper left
                  [v_angles[1], h_angles[1]],  # upper right
                  [v_angles[0], h_angles[1]]]  # bottom right
        r = R.from_euler('yz', angles)
        start_point = np.array([position.position.x, position.position.y, position.position.z])
        vectors = r.apply(np.array([1, 0, 0]))
        t = np.abs(height + (-position.position.z / vectors[:, 2]))
        t = np.nan_to_num(t)
        t = np.clip(t, None, max_range)
        xyz = np.empty((4, 2))
        xyz[:, 0] = t * vectors[:, 0] + position.position.x
        xyz[:, 1] = t * vectors[:, 1] + position.position.y
        rectangle = np.zeros((self.width, self.height))
        index = 0
        for point in xyz:
            map_point = self.map_point_from_real_coordinates(point[0], point[1], height)
            xyz[index] = np.array([map_point.x, map_point.y])
            index += 1
        cv2.fillConvexPoly(rectangle, xyz.astype(np.int32), 1)
        obstacles = np.logical_and(self._static_obstacles_height > 0, self._static_obstacles_depth < height)
        rectangle = np.logical_and(np.logical_not(obstacles), rectangle)
        indices = np.argwhere(rectangle > 0)
        right_indices = np.zeros(len(indices), dtype=np.bool)
        ray = np.empty((2, 3), dtype=np.float32)
        ray[0, :] = start_point
        i = -1
        for index in indices:
            i += 1
            map_point = self.map_point_from_map_coordinates(index[1], index[0], height)
            ray[1, :2] = [map_point.real_x + 0.25, map_point.real_y + 0.25]
            ray[1, 2] = height
            intersections = self.tree.rayIntersection(ray)
            if len(intersections) >= 1 and (
                    (np.abs(intersections[0].p[0] - ray[0, 0]) + np.abs(intersections[0].p[1]) - ray[0, 1]) < (np.abs(
                ray[1, 0] - ray[0, 0]) + np.abs(ray[1, 1] - ray[0, 1]))):
                continue
            right_indices[i] = 1
        return indices[right_indices, :]

    def append_to_collection(self, x, y, collection):
        if x not in collection:
            collection[x] = {}
        if y not in collection[x]:
            collection[x][y] = True

    def ray_tracing_3d(self, position, vfov, hfov, max_range, degree_resolution=0.5):
        start_time = rospy.Time.now().to_sec()
        step = np.deg2rad(degree_resolution)
        lh = int(hfov / step)
        lv = int(vfov / step)
        h_angles = np.empty((lh + 1))
        v_angles = np.empty((lv + 1))
        h_angles.fill(step)
        v_angles.fill(step)
        h_angles = h_angles * np.arange(0, lh + 1) + position.orientation.z
        v_angles = v_angles * np.arange(0, lv + 1) + position.orientation.y
        cp = Math.cartesian_product(h_angles, v_angles)
        r = R.from_euler('zy', cp)
        v = np.array([1, 0, 0])
        start_point = np.array([position.position.x, position.position.y, position.position.z])
        end_points = r.apply(v) * max_range + start_point
        ray_point_list = np.zeros((len(end_points), 2, 3))
        ray_point_list[:, 1] = end_points
        ray_point_list[:, 0] = start_point
        coordinates = {}
        # print("begin 3d", rospy.Time.now().to_sec() - start_time)
        a = []
        b = []
        c = []
        for i in range(len(end_points)):
            ray = np.array(ray_point_list[i], dtype=np.float32)
            start_time = time.time()
            intersections = self.tree.rayIntersection(ray)
            a.append(time.time() - start_time)
            if len(intersections) == 1:
                r = intersections[0]
                if np.round(r.p[2]) == 0.0:
                    start_time = time.time()
                    map_point = self.map_point_from_real_coordinates(r.p[0], r.p[1], 0)
                    b.append(time.time() - start_time)
                    start_time = time.time()
                    stepped_v = [map_point.x, map_point.y]
                    if stepped_v[0] not in coordinates:
                        coordinates[stepped_v[0]] = {}
                    if stepped_v[1] not in coordinates[stepped_v[0]]:
                        coordinates[stepped_v[0]][stepped_v[1]] = True
                    c.append(time.time() - start_time)

        # print(np.var(a, axis=0))
        # print("mean_a", np.mean(a, axis=0))
        # print("mean_b", np.mean(b, axis=0))
        # print("mean_c", np.mean(c, axis=0))
        # print("len", len(end_points))
        # print(np.min(a, axis=0))
        # print(np.max(a, axis=0))
        # print(np.median(a, axis=0))
        return coordinates

    def through_obstacles(self, start, end):
        # print("=============")
        ray = np.array([start, end]).astype(np.float32)
        if ray[1, 0] == -50 and ray[1, 1] == -30:
            # print("fiii")
            pass
        vector = end - start
        sum_vector = np.sum(np.abs(vector))
        vector_signs = np.sign(vector)
        intersections = self.tree.rayIntersection(ray)
        for intersection in intersections:
            intersection_vector_to = intersection.p - ray[0]
            intersection_vector_from = intersection.p - ray[1]
            sum_intersection_vector_to = np.sum(np.abs(intersection_vector_to))
            sum_intersection_vector_from = np.sum(np.abs(intersection_vector_from))
            # print(sum_intersection_vector_to, sum_intersection_vector_from, sum_vector)
            if (sum_intersection_vector_to > 0.2) and (sum_intersection_vector_from > 0.2) and (
                    sum_intersection_vector_to < sum_vector):
                intersection_vector_signs = np.sign(intersection_vector_to)
                # print(intersection_vector_signs[:2], vector_signs[:2], intersection_vector_signs[:2] + vector_signs[:2],
                #      np.sum(intersection_vector_signs[:2] + vector_signs[:2]))
                if ((intersection_vector_signs[:2] + vector_signs[:2]) != 0).all():
                    # print(np.sum(np.abs(intersection_vector_to)))
                    # print(np.sum(np.abs(intersection_vector_from)))
                    # print("distance",
                    #      (np.abs(intersection.p[0] - ray[0, 0]) + np.abs(intersection.p[1] - ray[0, 1])))
                    # print("vector")
                    # print(vector, intersection_vector_to)
                    # print("ray")
                    # for r in ray:
                    #    print(r)
                    # print("intersection", intersection.p)
                    return True
        # print("ray", ray)
        # for intersection in intersections:
        #    print(intersection.p)
        return False

    def ray(self, start_position, vector, stop_if_bump):
        norm_v = Math.normalize(vector)
        step_count = int(np.hypot(vector[0], vector[1])) + 1
        coordinates = {}
        bumped = 0
        for i in range(step_count):
            stepped_v = (norm_v * i)
            map_point = self.map_point_from_real_coordinates(start_position[0] + stepped_v[0],
                                                             start_position[1] + stepped_v[1], 0)
            if self.is_free(map_point, "target"):
                stepped_v = [map_point.x, map_point.y]
                if stepped_v[0] not in coordinates:
                    coordinates[stepped_v[0]] = {}
                if stepped_v[1] not in coordinates[stepped_v[0]]:
                    coordinates[stepped_v[0]][stepped_v[1]] = True
            else:
                bumped += 1
                if stop_if_bump:
                    break
        return coordinates, bumped

    def target_plan(self, start_position, end_position):
        return self.plan(start_position, end_position, "target")

    def drone_plan(self, start_position, end_position):
        return self.plan(start_position, end_position, "drone")

    def plan(self, start_position, end_position, object_type):
        self.load_all_services()
        try:
            response = self.get_right_plan_service(object_type).call(start_position, end_position)
            if response is not None:
                return response.plan
        except rospy.ServiceException as exc:
            rospy.logwarn("Service did not process request: " + str(exc))
        return []

    def height_point(self, map_point):
        return 0


class CameraCalculation():

    @staticmethod
    def generate_samples_object_in_fov(mean, location_accuracy, coordinates, resolution):
        conv = [[location_accuracy * resolution, 0],
                [0, location_accuracy * resolution]]
        samples = np.random.multivariate_normal(mean, conv, 100)
        samples_in_fov = []
        for sample in samples:
            if sample[0] in coordinates and sample[1] in coordinates[sample[0]]:
                samples_in_fov.append(sample)
        return samples_in_fov

    @staticmethod
    def generate_particles_object_in_fov(mean, location_accuracy, coordinates, resolution):
        conv = [[location_accuracy * resolution, 0],
                [0, location_accuracy * resolution]]
        mvn = multivariate_normal(mean, conv)
        weights = mvn.pdf(coordinates)
        mask = weights > 0.000001
        right_coordinates = coordinates[mask, :]
        right_weights = weights[mask]
        particles = np.zeros((np.count_nonzero(mask), 3))
        particles[:, :2] = right_coordinates
        particles[:, 2] = right_weights
        return particles

    @staticmethod
    def generate_particles_object_out_of_fov(coordinates):
        weight = CameraCalculation.coordinates_length(coordinates)
        particles = np.zeros((len(coordinates), 3))
        particles[:, :2] = coordinates
        particles[:, 2] = weight
        return particles

    @staticmethod
    def particles_to_image(array, width, height):
        image = np.zeros((width, height))
        image[array[:, 0].astype(np.int32), array[:, 1].astype(np.int32)] = array[:, 2]
        image /= np.sum(array[:, 2])
        return image

    @staticmethod
    def generate_samples_object_out_of_fov(coordinates, samples_count_per_pixel=3):
        samples = []
        for x_coor in coordinates:
            for y_coor in coordinates[x_coor]:
                for _ in range(samples_count_per_pixel):
                    samples.append([x_coor, y_coor])
        return np.array(samples)

    @staticmethod
    def generate_position_samples(samples, velocity, var_velocity):
        position_samples = np.zeros((len(samples), 7))
        for i in range(len(samples)):
            position_samples[i, 0] = samples[i, 0]
            position_samples[i, 1] = samples[i, 1]
            yaw = Math.calculate_yaw_from_points(0, 0, velocity[0], velocity[1])
            position_samples[i, 5] = yaw + np.random.uniform(-var_velocity, var_velocity, 1)[0]
        return position_samples

    @staticmethod
    def get_visible_coordinates(world_map, position, interval, hfov, camera_step):
        end_angle = position.orientation.z + hfov / 2.0
        start_angle = position.orientation.z - hfov / 2.0
        step_count = (interval[1] - interval[0]) / camera_step + 1
        coordinates = CameraCalculation.get_coordinates_of_fov(world_map, position, end_angle, start_angle, step_count,
                                                               interval, np.deg2rad(0.5))
        return coordinates, CameraCalculation.coordinates_length(coordinates)

    @staticmethod
    def get_coordinates_of_fov(world_map, start_position, start_angle, end_angle, step_count, interval, step):
        coordinates = {}
        pixel_size = 1 / world_map.resolution
        while start_angle > end_angle:
            v = Math.cartesian_coor(start_angle, interval[1])
            norm_v = Math.normalize(np.array([v.x, v.y]))
            for i in range(step_count):
                stepped_v = (norm_v * i + interval[0])
                map_point = world_map.map_point_from_real_coordinates(start_position.position.x + stepped_v[0],
                                                                      start_position.position.y + stepped_v[1], 0)
                if world_map.is_free(map_point, "target"):
                    # stepped_v = (stepped_v / pixel_size).astype(int)
                    stepped_v = [map_point.x, map_point.y]
                    if stepped_v[0] not in coordinates:
                        coordinates[stepped_v[0]] = {}
                    if stepped_v[1] not in coordinates[stepped_v[0]]:
                        coordinates[stepped_v[0]][stepped_v[1]] = True
                else:
                    break
            start_angle -= step
        return coordinates

    @staticmethod
    def coordinates_length(coordinates):
        l = 0
        for key in coordinates:
            l += len(coordinates[key])
        return l

    @staticmethod
    def write_coordinates(coordinates, target_array):
        for x_coor in coordinates:
            for y_coor in coordinates[x_coor]:
                # map_point = world_map.map_point_from_real_coordinates(x_coor, y_coor, 0)
                target_array[x_coor, y_coor] = 1
        return target_array

    @staticmethod
    def combine_coordinates(coordinates_collection):
        coordinates = coordinates_collection[0]
        for i in range(1, len(coordinates_collection)):
            for coor_x in coordinates_collection[i]:
                for coor_y in coordinates_collection[i][coor_x]:
                    if coor_x not in coordinates:
                        coordinates[coor_x] = {}
                    if coor_y not in coordinates[coor_x]:
                        coordinates[coor_x][coor_y] = True
        return coordinates


class Graph:

    @staticmethod
    def get_path(Pr, i, j):
        path = [j]
        k = j
        while Pr[i, k] != -9999:
            path.append(Pr[i, k])
            k = Pr[i, k]
        return path
