import numpy as np
import os
import json
from multiprocessing import Lock
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree, ConvexHull
from pyoctree import pyoctree as ot
import time
import cv2
import traceback
from pesat_utils.pesat_math import Math
from building_blocks import Building


class BaseMap():
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
                m = Building.create_map_from_objects(information["objects"]["statical"], information["world"]["width"],
                                                     information["world"]["height"],
                                                     information["world"]["maximal_height"], resolution)
                map = BaseMap(m, m, max_altitude=information["world"]["maximal_height"], resolution=resolution)
                s = np.zeros(m[0].shape)
                s.fill(255)
                s[m[0] > 0] = 0
                cv2.imwrite("map_created.png", s)
                map._file_path = file_name
                return map
            except Exception as e:
                traceback.print_exc()
                print("Exception occurred: ", e)
        else:
            print("File doesn't exist")
        ar = np.zeros((60, 60), dtype=np.uint8)
        obstacles = [ar, ar]
        map = BaseMap(obstacles, obstacles, max_altitude=20, resolution=resolution)
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
                rectangles_height = []
                array_corners_to_rectangles = []
                i = -1
                for object in information["objects"]["statical"]:
                    i += 1
                    object = Building.from_json_file_object_to_object(object)
                    corners = Building.split_to_points(object)[0]
                    center = Building.calculate_center_of_object(object)
                    centers_of_rectangles.append(center)
                    rectangles_height.append(object[0].height)
                    for corner in corners:
                        if corner[0] in mapping_corners_to_rectangles:
                            mapping_corners_to_rectangles[corner[0]][corner[1]] = i
                        else:
                            mapping_corners_to_rectangles[corner[0]] = {corner[1]: i}
                        array_corners_to_rectangles.append(i)
                        all_corners.append(corner)
                return all_corners, centers_of_rectangles, mapping_corners_to_rectangles, rectangles_height, array_corners_to_rectangles
            except Exception as e:
                print(e)
                return [], [], {}, [], []
        else:
            return [], [], {}, [], []

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
        self._rectangles_height = None
        self._array_corners_to_rectangles = None
        self._file_path = None
        self._object_type = None
        self._drone_admissibility_service = None
        self._target_admissibility_service = None
        self._drone_plan_service = None
        self._target_plan_service = None
        self._point_height_service = None
        self._tree = None
        self._map_details = None
        self._drone_min_altitude, self._drone_max_altitude = 3, 15
        self._target_min_altitude, self._target_max_altitude = 0.5, 0.5
        self._drone_size, self._target_size = 0.5, 1.0
        self._drone_width, self._target_width = 0.5, 1.5
        self._maximum_random_iteration = 100
        self._pixel_per_meter = 256 / max_altitude
        self._pixels_with_obstacles = np.nonzero(
            np.bitwise_or(self._static_obstacles_height, self._dynamic_obstacles_height))
        self._target_obstacle_map = self.get_target_obstacle_map()
        self._free_target_pixels = np.count_nonzero(self._target_obstacle_map)
        self._corners_tree = None

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
        self._rectangles_height = None
        self._array_corners_to_rectangles = None
        self._object_type = None
        self._drone_admissibility_service = None
        self._target_admissibility_service = None
        self._drone_plan_service = None
        self._target_plan_service = None
        self._point_height_service = None
        self._tree = None
        self._drone_min_altitude, self._drone_max_altitude = 3, 15
        self._target_min_altitude, self._target_max_altitude = 0.5, 0.5
        self._drone_size, self._target_size = 0.5, 1.0
        self._drone_width, self._target_width = 0.5, 1.5
        self._drone_pixel_distance = self._drone_max_altitude / 256
        self._target_pixel_distance = self._target_max_altitude / 256
        self._target_obstacle_map = self.get_target_obstacle_map()
        self._free_target_pixels = np.count_nonzero(self._target_obstacle_map)
        self._corners_tree = None

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
            self._corners, self._rectangles_centers, self._corners_rectangles, self._rectangles_height, self._array_corners_to_rectangles = BaseMap.load_corners_from_obstacle_file(
                self._file_path)
        return self._corners

    @property
    def rectangles_height(self):
        if self._rectangles_height is None and self._file_path is not None:
            self._corners, self._rectangles_centers, self._corners_rectangles, self._rectangles_height, self._array_corners_to_rectangles = BaseMap.load_corners_from_obstacle_file(
                self._file_path)
        return self._rectangles_height

    @property
    def array_corners_to_rectangles(self):
        if self._array_corners_to_rectangles is None and self._file_path is not None:
            self._corners, self._rectangles_centers, self._corners_rectangles, self._rectangles_height, self._array_corners_to_rectangles = BaseMap.load_corners_from_obstacle_file(
                self._file_path)
        return self._array_corners_to_rectangles

    @property
    def corners_tree(self):
        if self._corners_tree is None:
            if len(self.corners) > 0:
                self._corners_tree = cKDTree(self.corners)
        return self._corners_tree

    @property
    def rectangles_centers(self):
        if self._rectangles_centers is None and self._file_path is not None:
            self._corners, self._rectangles_centers, self._corners_rectangles, self._rectangles_height, self._array_corners_to_rectangles = BaseMap.load_corners_from_obstacle_file(
                self._file_path)
        return self._rectangles_centers

    @property
    def corners_rectangles(self):
        if self._corners_rectangles is None and self._file_path is not None:
            self._corners, self._rectangles_centers, self._corners_rectangles, self._rectangles_height, self._array_corners_to_rectangles = BaseMap.load_corners_from_obstacle_file(
                self._file_path)
        return self._corners_rectangles

    @property
    def map(self):
        return self._static_obstacles_height

    @property
    def resolution(self):
        return self._resolution

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

    @property
    def target_obstacle_map(self):
        return self._target_obstacle_map

    @property
    def free_target_pixels(self):
        return self._free_target_pixels

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

    def get_index_on_map(self, x, y, safety=True):
        box_index_x = np.round(x / self._box_size)
        box_index_y = -np.round(y / self._box_size)
        shifted_x = box_index_x + self._center_shift_x
        shifted_y = box_index_y + self._center_shift_y
        if safety:
            shifted_x = np.clip(shifted_x, 0, self.width - 1)
            shifted_y = np.clip(shifted_y, 0, self.height - 1)
        if isinstance(x, np.ndarray):
            return shifted_x.astype(np.int32), shifted_y.astype(np.int32)
        else:
            return int(shifted_x), int(shifted_y)

    def get_coordination_from_map_indices_vectors(self, map_x, map_y):
        shifted_x = map_x - self._center_shift_x
        shifted_y = map_y - self._center_shift_y
        real_x = shifted_x * self._box_size + np.random.uniform(0, self._box_size, len(map_x))
        real_y = -1 * shifted_y * self._box_size + np.random.uniform(0, self._box_size, len(map_y))
        return real_x, real_y

    def get_coordination_from_map_indices(self, map_point):
        shifted_x = map_point.x - self._center_shift_x
        shifted_y = map_point.y - self._center_shift_y
        real_x = shifted_x * self._box_size
        real_y = shifted_y * self._box_size * -1
        return [real_x, real_y]

    def is_free_on_target_map(self, x, y):
        return self.target_obstacle_map[y, x] > 0

    def find_nearest_free_position(self, x, y):
        const = 10
        cut_map = self.target_obstacle_map[y - const: y + const, x - const: x + const]
        indices = np.argwhere(cut_map > 0)
        nearest_in_cut_map = indices[
            np.argmin(np.sum(np.abs(indices - np.array([cut_map.shape[0], cut_map.shape[1]])), 1))]
        return max(0, x - const) + nearest_in_cut_map[1], max(0, y - const) + nearest_in_cut_map[0]

    def get_target_obstacle_map(self):
        min_pixel = int(self._pixel_per_meter * self._target_min_altitude)
        max_pixel = int(self._pixel_per_meter * self._target_max_altitude + 1)
        target_obstacles_map = np.zeros((self.width, self.height))
        object_size = self.get_right_object_size("target")
        object_width = self.get_right_object_width("target")
        boxes_count = np.ceil((object_width / 2.0 + self._box_size / 2.0) / self._box_size)
        for x in range(self.width):
            for y in range(self.height):
                lower_limit_x, upper_limit_x, all_right_x = BaseMap.check_bounds(
                    [int(x - boxes_count), int(x + boxes_count)], [0, self.width])
                lower_limit_y, upper_limit_y, all_right_y = BaseMap.check_bounds(
                    [int(y - boxes_count), int(y + boxes_count)], [0, self.height])
                if not all_right_x or not all_right_y:
                    continue
                for k in range(min_pixel, max_pixel):
                    height = (1.0 / self._pixel_per_meter) * k
                    if self.is_fast_free_for_object_map(lower_limit_x, upper_limit_x + 1, lower_limit_y,
                                                        upper_limit_y + 1, "target", height, object_size):
                        target_obstacles_map[x, y] += 1
        target_obstacles_map /= (max_pixel - min_pixel)
        return target_obstacles_map

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

    def get_right_object_size(self, object_type):
        if object_type == "drone":
            return self._drone_size
        else:
            return self._target_size

    def get_right_object_width(self, object_type):
        if object_type == "drone":
            return self._drone_width
        else:
            return self._target_width

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

    def faster_rectangle_ray_tracing_3d(self, position, vfov, hfov, max_range):
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
        t = height + (-position.position.z / vectors[:, 2])
        t = np.nan_to_num(t)
        t[t < 0] = max_range
        t = np.clip(t, None, max_range)
        xyz = np.empty((4, 2))
        xyz[:, 0], xyz[:, 1] = self.get_index_on_map(t * vectors[:, 0] + position.position.x,
                                                     t * vectors[:, 1] + position.position.y, False)
        rectangle = np.zeros((self.width, self.height), dtype=np.uint8)
        visible_points = np.zeros((self.width, self.height), dtype=np.uint8)
        cv2.fillConvexPoly(rectangle, xyz.astype(np.int32), 1)
        cv2.fillConvexPoly(visible_points, xyz.astype(np.int32), 1)
        obstacles = np.logical_and(self._static_obstacles_height > 0, self._static_obstacles_depth < height)
        rectangle = np.logical_and(np.logical_not(obstacles), rectangle).astype(np.uint8)
        # ones mean free
        # cv2.imwrite("rectangle_map.png", rectangle*255)
        # cv2.imwrite("veisible_points_map.png", visible_points * 255)
        # find corners
        ii = np.array(self.corners_tree.query_ball_point([start_point[:2]], max_range)[0]).astype(np.int32)
        if len(ii) > 0:
            corners = np.array(self.corners)
            cr = np.array(self.array_corners_to_rectangles)
            rectangles = np.zeros(len(self.rectangles_height))
            inside_area = corners[ii]
            map_corners_points_x, map_corners_points_y = self.get_index_on_map(inside_area[:, 0], inside_area[:, 1])
            mask = visible_points[map_corners_points_y, map_corners_points_x] == 1
            iii = ii[mask]
            rectangles[cr[iii]] = 1
            indices = np.argwhere(rectangles > 0)
            # c_points = np.zeros((self.width, self.height), dtype=np.uint8)
            for i in indices:
                i = i[0]
                cs = np.zeros((4, 3))
                cs[:, :2] = corners[i * 4:(i + 1) * 4]
                cs[:, 2] = self.rectangles_height[i]
                vectors = cs - start_point
                t = height + (-position.position.z / vectors[:, 2])
                t = np.nan_to_num(t)
                t[t < 0] = max_range
                t = np.clip(t, None, max_range)
                xyz = np.empty((8, 2))
                xyz[:4, 0], xyz[:4, 1] = self.get_index_on_map(cs[:, 0], cs[:, 1])
                xyz[4:, 0], xyz[4:, 1] = self.get_index_on_map(t * vectors[:, 0] + position.position.x,
                                                               t * vectors[:, 1] + position.position.y, False)
                hull = ConvexHull(xyz)
                # cv2.fillConvexPoly(c_points, xyz[:4, :2].astype(np.int32), 1)
                # cv2.imwrite("maps/c_points_{}_map.png".format(i), c_points * 255)
                cv2.fillConvexPoly(rectangle, xyz[hull.vertices].astype(np.int32), 0)
        cv2.imwrite("rectangle_map.png".format(), rectangle * 255)
        indices = np.argwhere(rectangle > 0)
        coordinates = np.empty(indices.shape, np.int32)
        coordinates[:, [0, 1]] = indices[:, [1, 0]]
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

    def height_point(self, map_point):
        return 0


class OldClass(object):

    def append_to_collection(self, x, y, collection):
        if x not in collection:
            collection[x] = {}
        if y not in collection[x]:
            collection[x][y] = True

    def point_outside_condition(self, point):
        return point[0] > self.width or point[1] > self.height or point[0] < 0 or point[1] < 0

    def target_point(self, point):
        target_value = np.array(point)
        if point[0] > self.width:
            target_value[0] = self.width
        if point[1] > self.height:
            target_value[1] = self.height
        if point[0] < 0:
            target_value[0] = 0
        if point[1] < 0:
            target_value[1] = 0
        return target_value

    def intersection_point(self, target_value, point, rolled_point):
        vector = point - rolled_point
        t = (target_value - rolled_point) / vector
        argmin_t = np.argmin(np.abs(t))
        opposite_index = 1 - argmin_t
        opposite_value = vector[opposite_index] * t[argmin_t] + rolled_point[opposite_index]
        if opposite_index == 0 and opposite_value > self.width:
            opposite_value = self.width
        if opposite_index == 1 and opposite_value > self.height:
            opposite_value = self.height
        if opposite_value < 0:
            opposite_value = 0
        p = np.zeros(2)
        p[argmin_t] = target_value[argmin_t]
        p[opposite_index] = opposite_value
        return p

    def port_inside_map(self, points):
        rolled_points = np.roll(points, 1, 0)
        corrected_points = []
        for point_index in xrange(len(points)):
            point = np.array(points[point_index])
            rolled_point = np.array(rolled_points[point_index])
            if self.point_outside_condition(point) or self.point_outside_condition(rolled_point):
                target_value = self.target_point(point)
                intersection = self.intersection_point(target_value, point, rolled_point)
                corrected_points.append(intersection.astype(np.int32))
                if self.point_outside_condition(point):
                    continue
            corrected_points.append(point)
        return np.array(corrected_points)

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

    def fast_rectangle_ray_tracing_3d(self, position, vfov, hfov, max_range):
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
        xyz[:, 0], xyz[:, 1] = self.get_index_on_map(t * vectors[:, 0] + position.position.x,
                                                     t * vectors[:, 1] + position.position.y, False)
        rectangle = np.zeros((self.width, self.height), dtype=np.uint8)
        cv2.fillConvexPoly(rectangle, xyz.astype(np.int32), 1)
        obstacles = np.logical_and(self._static_obstacles_height > 0, self._static_obstacles_depth < height)
        rectangle = np.logical_and(np.logical_not(obstacles), rectangle).astype(np.uint8)
        # find corners
        ii = np.array(self.corners_tree.query_ball_point([start_point[:2]], max_range)[0])
        corners = np.array(self.corners)
        cr = np.array(self.array_corners_to_rectangles)
        rectangles = np.zeros(len(self.rectangles_height))
        inside_area = corners[ii]
        map_corners_points_x, map_corners_points_y = self.get_index_on_map(inside_area[:, 0], inside_area[:, 1])
        mask = rectangle[map_corners_points_x, map_corners_points_y] == 1
        iii = ii[mask]
        rectangles[cr[iii]] = 1
        indices = np.argwhere(rectangles > 0)
        for i in indices:
            i = i[0]
            cs = np.zeros((4, 3))
            cs[:, :2] = corners[i * 4:(i + 1) * 4]
            cs[:, 2] = self.rectangles_height[i]
            vectors = cs - start_point
            t = np.abs(height + (-position.position.z / vectors[:, 2]))
            t = np.nan_to_num(t)
            t = np.clip(t, None, max_range)
            xyz = np.empty((8, 2))
            xyz[:4, 0], xyz[:4, 1] = self.get_index_on_map(cs[:, 0], cs[:, 1])
            xyz[4:, 0], xyz[4:, 1] = self.get_index_on_map(t * vectors[:, 0] + position.position.x,
                                                           t * vectors[:, 1] + position.position.y, False)
            hull = ConvexHull(xyz)
            cv2.fillConvexPoly(rectangle, xyz[hull.vertices].astype(np.int32), 0)
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

    def not_so_faster_rectangle_ray_tracing_3d(self, position, vfov, hfov, max_range):
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
        xyz[:, 0], xyz[:, 1] = self.get_index_on_map(t * vectors[:, 0] + position.position.x,
                                                     t * vectors[:, 1] + position.position.y, False)
        rectangle = np.zeros((self.width, self.height), dtype=np.uint8)
        cv2.fillConvexPoly(rectangle, xyz.astype(np.int32), 1)
        obstacles = np.logical_and(self._static_obstacles_height > 0, self._static_obstacles_depth < height)
        rectangle = np.logical_and(np.logical_not(obstacles), rectangle).astype(np.uint8)
        # find corners
        ii = np.array(self.corners_tree.query_ball_point([start_point[:2]], max_range)[0])
        corners = np.array(self.corners)
        cr = np.array(self.array_corners_to_rectangles)
        rectangles = np.zeros(len(self.rectangles_height))
        rectangles[cr[ii]] = 1
        indices = np.argwhere(rectangles > 0)
        for i in indices:
            i = i[0]
            cs = np.zeros((4, 3))
            cs[:, :2] = corners[i * 4:(i + 1) * 4]
            cs[:, 2] = self.rectangles_height[i]
            vectors = cs - start_point
            t = np.abs(height + (-position.position.z / vectors[:, 2]))
            t = np.nan_to_num(t)
            t = np.clip(t, None, max_range)
            xyz = np.empty((8, 2))
            xyz[:4, 0], xyz[:4, 1] = self.get_index_on_map(cs[:, 0], cs[:, 1])
            xyz[4:, 0], xyz[4:, 1] = self.get_index_on_map(t * vectors[:, 0] + position.position.x,
                                                           t * vectors[:, 1] + position.position.y, False)
            hull = ConvexHull(xyz)
            cv2.fillConvexPoly(rectangle, xyz[hull.vertices].astype(np.int32), 0)
        indices = np.argwhere(rectangle > 0)
        return indices

    def ray_tracing_3d(self, position, vfov, hfov, max_range, degree_resolution=0.5):
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