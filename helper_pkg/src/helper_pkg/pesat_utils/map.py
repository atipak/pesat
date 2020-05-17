import numpy as np
import rospy
from pesat_msgs.srv import PointFloat, PosePlan
from geometry_msgs.msg import Point
from helper_pkg.pesat_utils.pesat_math import Math
from helper_pkg.pesat_utils.base_map import BaseMap


class Map(BaseMap):
    def __init__(self, static_obstacles_arrays, dynamic_obstacles_arrays, resolution=2, center=(0, 0),
                 max_altitude=20.0):
        super(Map, self).__init__(static_obstacles_arrays, dynamic_obstacles_arrays, resolution, center, max_altitude)
        self._drone_min_altitude, self._drone_max_altitude = self.get_altitudes("drone")
        self._target_min_altitude, self._target_max_altitude = self.get_altitudes("target")
        self._drone_pixel_distance = self._drone_max_altitude / 256
        self._target_pixel_distance = self._target_max_altitude / 256
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
        self._drone_min_altitude, self._drone_max_altitude = self.get_altitudes("drone")
        self._target_min_altitude, self._target_max_altitude = self.get_altitudes("target")
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
            return response.value > 0
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
