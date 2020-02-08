import rospy
import os
import copy
from geometry_msgs.msg import Pose
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.cluster.hierarchy import fclusterdata
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Voronoi
import cv2
from shapely.geometry import Polygon, LineString, MultiPolygon, Point, MultiPoint
import numpy as np
import helper_pkg.utils as utils
import time
import triangle
import matplotlib.pyplot as plt
from shapely.ops import triangulate, nearest_points
from shapely.strtree import STRtree
from shapely.validation import explain_validity
from scipy.optimize import differential_evolution
import random
from deap import base
from deap import creator
from deap import tools
import traceback

import multiprocessing
import pickle as pck
from copy_reg import pickle
from types import MethodType


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


class SectionMap(object):
    def __init__(self):
        super(SectionMap, self).__init__()
        pickle(MethodType, _pickle_method, _unpickle_method)
        _camera_configuration = rospy.get_param("drone_configuration")["camera"]
        logic_configuration = rospy.get_param("logic_configuration")
        self.hfov = _camera_configuration["hfov"]
        self.image_height = _camera_configuration["image_height"]
        self.image_width = _camera_configuration["image_width"]
        self.focal_length = self.image_width / (2.0 * np.tan(self.hfov / 2.0))
        self.vfov = 2 * np.arctan2(self.image_height / 2, self.focal_length)
        self.camera_range = _camera_configuration["camera_range"]
        self.camera_max_angle = logic_configuration["tracking"]["camera_max_angle"]
        self._max_section_area = 40000
        self.vertices_tree = None
        self.centers_tree = None
        self.vertices = np.array([])
        self.delimiters = np.array([])
        self.free_pixels, self.world_map, self.correct_triangles = 0, None, []
        self.rtc = None
        self.section_map_image = None
        self.iterations = 0

    def create_sections_regions_and_points(self, map_file):
        environment_configuration = rospy.get_param("environment_configuration")
        max_target_velocity = rospy.get_param("target_configuration")["strategies"]["max_velocity"]
        begin = time.time()
        world_map = utils.Map.map_from_file(map_file, 2)
        print("World map was loaded", "during time", time.time() - begin)
        begin = time.time()
        sections, obstacles = self.create_sections(world_map, environment_configuration)
        print("Section were created:", len(sections), "during time", time.time() - begin)
        regions = self.create_regions(world_map, sections, obstacles)
        # regions = [section1: [cluster1: [point1: [x, y, z, vertical orientation of camera, yaw],
        # point2, ...], cluster2, ...], section2, ...]
        sections_objects = {}
        section_index = 0
        region_index = 0
        point_index = 0
        point_coordinates = np.zeros((world_map.width, world_map.height))
        built_up_coefficient = world_map.map_details["built_up"]
        half_target_velocity = max_target_velocity / 2.0
        regions_id = []
        sections_id = []
        points_id = []
        points_into_voronoi = []
        for section in sections:
            section_object = SectionMapObject()
            section_object.type = SectionMapObjectTypes.Section
            section_object.object_id = section_index
            polygon = Polygon(section)
            if polygon.contains(polygon.centroid):
                section_object.centroid = np.array(polygon.centroid)
            else:
                [np_polygon, _] = nearest_points(polygon, polygon.centroid)
                section_object.centroid = np.array(np_polygon)
            sections_objects[section_object.object_id] = section_object
            a = np.array(zip(*polygon.exterior.xy))
            d = np.max(distance.cdist(a, a))
            print(d, built_up_coefficient, half_target_velocity)
            section_object.maximal_time = d / ((1 - built_up_coefficient) * half_target_velocity)
            region_visibility = 0
            sections_regions = regions[section_index]
            section_map, section_map_without_obstacles = self.section_map(section, world_map)
            for region in sections_regions:
                region_object = SectionMapObject()
                region_object.type = SectionMapObjectTypes.Region
                region_object.object_id = region_index
                points = []
                for r in region:
                    points.append(r[:2])
                if len(points) == 1:
                    polygon = Point(points[0])
                    d = 1
                elif len(points) == 2:
                    polygon = MultiPoint(points).convex_hull
                    d = Point(points[0]).distance(Point(points[1]))
                else:
                    polygon = MultiPoint(points).convex_hull
                    if type(polygon) == LineString:
                        coords = polygon.coords
                        d = Point(coords[0]).distance(Point(coords[1]))
                    elif type(polygon) == Point:
                        d = 1
                    else:
                        a = np.array(zip(*polygon.exterior.xy))
                        d = np.max(distance.cdist(a, a))
                region_object.maximal_time = d / ((1 - built_up_coefficient) * half_target_velocity)
                region_object.centroid = polygon.centroid
                section_object.objects[region_object.object_id] = region_object
                points_visibility = 0
                for point in region:
                    position = Pose()
                    position.position.x = point[0]
                    position.position.y = point[1]
                    position.position.z = point[2]
                    position.orientation.y = point[3]
                    position.orientation.z = point[4]
                    coordinates = world_map.rectangle_ray_tracing_3d(position, self.vfov, self.hfov, self.camera_range)
                    point_object = SectionMapObject()
                    point_object.type = SectionMapObjectTypes.Point
                    point_object.object_id = point_index
                    data_id = 0
                    for coor in coordinates:
                        if section_map[coor[1], coor[0]] == 1 and point_coordinates[coor[0], coor[1]] != 1:
                            point_object.objects[data_id] = [coor[1], coor[0]]
                            point_coordinates[coor[0], coor[1]] = 1
                            data_id += 1
                    point_object.centroid = point[:2]
                    point_object.data = point
                    point_object.maximal_time = 1
                    points_visibility += len(coordinates)
                    point_object.visibility = len(coordinates) * np.square(world_map.box_size)
                    point_object.maximal_entropy = np.clip(np.log2(len(coordinates)), 0, None)
                    region_object.objects[point_object.object_id] = point_object
                    regions_id.append(region_index)
                    sections_id.append(section_index)
                    points_id.append(point_index)
                    points_into_voronoi.append([point[0], point[1]])
                    point_index += 1
                region_index += 1
                region_visibility += points_visibility
                region_object.visibility = points_visibility * np.square(world_map.box_size)
                region_object.maximal_entropy = np.clip(np.log2(points_visibility), 0, None)
            section_index += 1
            section_object.visibility = region_visibility * np.square(world_map.box_size)
            section_object.maximal_entropy = np.clip(np.log2(region_visibility), 0, None)
        vor = Voronoi(np.array(points_into_voronoi))
        for ridge_point in vor.ridge_points:
            s = sections_id[ridge_point[0]]
            s1 = sections_id[ridge_point[1]]
            r = regions_id[ridge_point[0]]
            r1 = regions_id[ridge_point[1]]
            p = points_id[ridge_point[0]]
            p1 = points_id[ridge_point[1]]
            if s1 != s:
                if s1 not in sections_objects[s].neighbors:
                    sections_objects[s].neighbors[s1] = []
                if [r, r1, p, p1] not in sections_objects[s].neighbors[s1]:
                    sections_objects[s].neighbors[s1].append([r, r1, p, p1])
                if s not in sections_objects[s1].neighbors:
                    sections_objects[s1].neighbors[s] = []
                if [r, r1, p, p1] not in sections_objects[s1].neighbors[s]:
                    sections_objects[s1].neighbors[s].append([r, r1, p, p1])
            if r1 != r:
                if r1 not in sections_objects[s].objects[r].neighbors:
                    sections_objects[s].objects[r].neighbors[r1] = []
                if [p, p1] not in sections_objects[s].objects[r].neighbors[r1]:
                    sections_objects[s].objects[r].neighbors[r1].append([p, p1])
                if r not in sections_objects[s1].objects[r1].neighbors:
                    sections_objects[s1].objects[r1].neighbors[r] = []
                if [p, p1] not in sections_objects[s1].objects[r1].neighbors[r]:
                    sections_objects[s1].objects[r1].neighbors[r].append([p, p1])
            if p != p1:
                if p1 not in sections_objects[s].objects[r].objects[p].neighbors:
                    sections_objects[s].objects[r].objects[p].neighbors[p1] = []
                if [p, p1] not in sections_objects[s].objects[r].objects[p].neighbors[p1]:
                    sections_objects[s].objects[r].objects[p].neighbors[p1].append([p, p1])
                if p not in sections_objects[s1].objects[r1].objects[p1].neighbors:
                    sections_objects[s1].objects[r1].objects[p1].neighbors[p] = []
                if [p, p1] not in sections_objects[s1].objects[r1].objects[p1].neighbors[p]:
                    sections_objects[s1].objects[r1].objects[p1].neighbors[p].append([p, p1])

        return sections_objects

    def create_sections(self, world_map, environment_configuration):
        begin = time.time()
        corners = self.add_map_corners_to_corners(world_map)
        print("Map corners joined with obstacles ones", "during time", time.time() - begin)
        begin = time.time()
        rtc = self.map_rectangles_to_corners(corners)
        self.rtc = rtc
        print("Mapping rectangles to corners created", "during time", time.time() - begin)
        begin = time.time()
        verticies, delimiters = self.create_verticies(rtc)
        obstacles = self.get_shapely_obstacles(verticies, delimiters)
        print("Vertices for graph created:", len(verticies), "during time", time.time() - begin)
        # SectionMap.print_vertices(verticies, obstacles)
        begin = time.time()
        edges = self.create_edges(verticies, delimiters, world_map, environment_configuration)
        print("Edges for graph created:", np.count_nonzero(edges), "during time", time.time() - begin)
        # SectionMap.print_edges(verticies, edges, obstacles)
        # first section is map
        verticies_indices = range(len(verticies))
        sections_for_reduction = [verticies_indices[delimiters[len(delimiters) - 2] + 1:]]
        # print(sections_for_reduction)
        sections = []
        maximum = np.max(edges)
        self._max_section_area = np.clip(int(world_map.real_width * world_map.real_height / 50), 4000, 20000)
        print("Launching algorithm for making sections")
        while len(sections_for_reduction) > 0:
            section = np.array(sections_for_reduction.pop())
            rolled_section = np.roll(section, 1)
            # saved_index = 0
            # l_index = 0
            # for l_index in range(len(section)):
            #    v = verticies[section[l_index]]
            #    if v[0] == -10 and v[1] == -4:
            #        print(v)
            #        saved_index = l_index
            #        break
            # print(section)
            # print(rolled_section)
            # indeces = zip(rolled_section, section)
            mask = edges[rolled_section, section] == 1.0
            mask_reversed = edges[section, rolled_section] == 1.0
            edges[rolled_section[mask], section[mask]] += maximum
            edges[section[mask_reversed], rolled_section[mask_reversed]] += maximum
            # print(edges)
            graph = csr_matrix(edges)
            _, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True)
            [vertex1_index] = np.random.choice(len(section), 1, False)
            # vertex1_index = saved_index
            vertex2_index = (vertex1_index + len(section) / 2) % len(section)
            lower_index, upper_index = (vertex1_index, vertex2_index) if vertex1_index < vertex2_index else (
                vertex2_index, vertex1_index)
            correct_path = False
            path_upper_to_lower = []
            for i in range(int(len(section) / 2)):
                v1_index, v2_index = (vertex1_index + i) % len(section), (vertex2_index + i) % len(section)
                lower_index, upper_index = (v1_index, v2_index) if v1_index < v2_index else (
                    v2_index, v1_index)
                path_upper_to_lower = utils.Graph.get_path(predecessors, section[lower_index], section[upper_index])
                if len(np.intersect1d(path_upper_to_lower, section)) != len(path_upper_to_lower):
                    correct_path_upper_to_lower = [path_upper_to_lower[0]]
                    for pi in range(1, len(path_upper_to_lower)):
                        p = path_upper_to_lower[pi]
                        if p not in section:
                            correct_path_upper_to_lower.append(p)
                        else:
                            correct_path_upper_to_lower.append(p)
                            break
                    path_upper_to_lower = correct_path_upper_to_lower
                    correct_first = np.in1d(section, correct_path_upper_to_lower[0]).nonzero()[0][0]
                    correct_last = np.in1d(section, correct_path_upper_to_lower[-1]).nonzero()[0][0]
                    lower_index, upper_index = (correct_first, correct_last) if correct_first < correct_last else (
                        correct_last, correct_first)
                    path_lower_to_upper = path_upper_to_lower[::-1]
                    # print("vertices", verticies[section[lower_index]], verticies[section[upper_index]])
                    # print("paths", path_upper_to_lower, path_lower_to_upper)
                    section_one = [section[i] for i in range(lower_index + 1, upper_index)]
                    section_two = [section[i] for i in range(upper_index + 1, len(section))]
                    section_two_part_one = [section[i] for i in range(0, lower_index)]
                    section_two.extend(section_two_part_one)
                    section_one.extend(path_upper_to_lower)
                    section_two.extend(path_lower_to_upper)
                    if len(section_one) > 2 and len(section_two) > 2:
                        section_one_polygon = Polygon(self.section_to_verticies(section_one, verticies))
                        section_two_polygon = Polygon(self.section_to_verticies(section_two, verticies))
                        if section_one_polygon.is_valid and section_two_polygon.is_valid and section_one_polygon.area > 25 and section_two_polygon.area > 25:
                            if self.get_section_area(section_one, verticies) > self._max_section_area:
                                sections_for_reduction.append(section_one)
                            else:
                                sections.append(section_one)
                            if self.get_section_area(section_two, verticies) > self._max_section_area:
                                sections_for_reduction.append(section_two)
                            else:
                                sections.append(section_two)
                            correct_path = True
                            break
            if not correct_path:
                sections.append(section)
        vertex_sections = []
        for s in sections:
            vertex_sections.append(np.array(self.section_to_verticies(s, verticies)))
        vertex_sections = np.array(vertex_sections)
        SectionMap.print_section(vertex_sections, obstacles)
        return vertex_sections, obstacles

    def section_to_verticies(self, section, verticies):
        vertex_section = []
        for v in section:
            vertex_section.append(verticies[v])
        return vertex_section

    def get_section_area(self, section, verticies):
        area = 0
        for i in range(len(section)):
            index = section[i]
            if i + 1 < len(section):
                next_index = section[i + 1]
            else:
                next_index = section[0]
            red = verticies[index][0] * verticies[next_index][1]
            blue = verticies[index][1] * verticies[next_index][0]
            area += red - blue
        area = np.abs(area) / 2
        return area

    def add_map_corners_to_corners(self, world_map):
        corners = copy.copy(world_map.corners)
        corners.append([-world_map.real_width / 2, -world_map.real_height / 2])
        corners.append([-world_map.real_width / 2, world_map.real_height / 2])
        corners.append([world_map.real_width / 2, world_map.real_height / 2])
        corners.append([world_map.real_width / 2, -world_map.real_height / 2])
        return corners

    def create_edges(self, verticies, delimiters, world_map, environment_configuration):
        under_vision_map = self.locations_under_vision(world_map, environment_configuration)
        print("Sector map created.")
        centers = world_map.rectangles_centers
        self.centers_tree = cKDTree(centers)
        print("KD tree of centers created.")
        self.vertices_tree = cKDTree(verticies)
        print("KD tree of vertices created.")
        edges = np.zeros((len(verticies), len(verticies)))
        rectangles_index = 0
        for index in range(len(verticies)):
            dd, ii = self.centers_tree.query(verticies[index], 5)
            max_distance = np.max(dd)
            vertices_indices = np.array(self.vertices_tree.query_ball_point(verticies[index], max_distance))
            vertices_indices = vertices_indices[vertices_indices > index]
            upper_delimiter_index = np.searchsorted(delimiters, index)
            if upper_delimiter_index == len(delimiters):
                upper_delimiter = delimiters[upper_delimiter_index - 1]
            else:
                upper_delimiter = delimiters[upper_delimiter_index]
            if upper_delimiter_index == 0:
                bottom_delimiter = 0
            else:
                bottom_delimiter = delimiters[upper_delimiter_index - 1]
            for v in vertices_indices:
                if bottom_delimiter <= v <= upper_delimiter:
                    continue
                edges[index, v] = self.calculate_distance(verticies[index], verticies[v], under_vision_map, world_map)
            if index == delimiters[rectangles_index]:
                if rectangles_index - 1 < 0:
                    edges[index, 0] = 1.0
                else:
                    edges[index, delimiters[rectangles_index - 1] + 1] = 1.0
                rectangles_index += 1
            else:
                edges[index, index + 1] = 1.0
        for v1 in range(len(edges)):
            for v2 in range(len(edges[v1])):
                if edges[v1, v2] > 0 and edges[v2, v1] == 0:
                    edges[v2, v1] = edges[v1, v2]
        self.recalculate_edges(edges)
        return edges

    def recalculate_edges(self, edges):
        maximum = np.max(edges)
        minimum = np.min(edges)
        edges[np.logical_and(edges > 0, edges != 1.0)] = maximum * edges[
            np.logical_and(edges > 0, edges != 1.0)] + maximum

    def calculate_distance(self, vertex, vertex2, sector_map, world_map):
        hit = world_map.through_obstacles(np.array([vertex[0], vertex[1], 0.5]),
                                          np.array([vertex2[0], vertex2[1], 0.5]))
        if hit:
            distance = 0
        else:
            line_map = np.zeros((world_map.width, world_map.height))
            v = world_map.get_index_on_map(*vertex)
            v2 = world_map.get_index_on_map(*vertex2)
            line_map = cv2.line(line_map, v, v2, -1)
            cell_size = 1.0 / world_map.resolution
            distance = cell_size * (np.count_nonzero(line_map) - np.count_nonzero(np.logical_and(line_map, sector_map)))
        return distance

    def locations_under_vision(self, world_map, environment_configuration):
        file_path = environment_configuration["watchdog"]["cameras_file"]
        sectors_map = np.zeros((world_map.width, world_map.height))
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    # x, y, z, pitch, orientation, hfov, vfov, camera_range, location_accuracy, recognition_error
                    camera = line.split(",")
                    position = Pose()
                    position.position.x = camera[0]
                    position.position.y = camera[1]
                    position.position.z = camera[2]
                    position.orientation.y = camera[3]
                    position.orientation.z = camera[4]
                    hfov = camera[5]
                    vfov = camera[6]
                    camera_range = camera[7]
                    coordinates, _ = world_map.ray_tracing_3d(position, vfov, hfov, camera_range)
                    sectors_map = utils.CameraCalculation.write_coordinates(coordinates, sectors_map)
        return sectors_map

    def create_verticies(self, rtc):
        delimiters = []
        index = 0
        verticies = []
        for rectangle in rtc:
            corners = rtc[rectangle]
            for i in range(4):
                current = i % 4
                next = (i + 1) % 4
                vector = [corners[next][0] - corners[current][0], corners[next][1] - corners[current][1]]
                distance = utils.Math.euclidian_distance(utils.Math.Point(corners[current][0], corners[current][1]),
                                                         utils.Math.Point(corners[next][0], corners[next][1]))
                if distance > 1:
                    vector = utils.Math.normalize(vector)
                    range_end = int(distance)
                    if distance % 1 > 0:
                        range_end += 1
                    for j in range(range_end):
                        point = vector * j + corners[current]
                        verticies.append(point)
                        index += 1
            delimiters.append(index - 1)
        self.vertices = np.array(verticies)
        self.delimiters = np.array(delimiters)
        return self.vertices, self.delimiters

    def map_rectangles_to_corners(self, corners):
        iterations = int(len(corners) / 4)
        rtc = {}
        index = 0
        for i in range(iterations):
            rtc[index] = []
            for j in range(4):
                if i * 4 + j >= len(corners):
                    break
                corner = corners[i * 4 + j]
                rtc[index].append(corner)
            index += 1
        return rtc

    def get_shapely_obstacles(self, vertices, delimiters):
        obstacles = []
        points = []
        rectangles_index = 0
        for index in range(len(vertices)):
            if rectangles_index == len(delimiters) - 1:
                break
            if index == self.delimiters[rectangles_index]:
                if rectangles_index - 1 < 0:
                    points.append(vertices[0])
                else:
                    points.append(vertices[delimiters[rectangles_index - 1] + 1])
                obstacles.append(Polygon(points))
                points = []
                rectangles_index += 1
            else:
                points.append(vertices[index])
        return obstacles

    def get_shapely_obstacles_from_corners(self):
        obstacles = []
        for rectangle in self.rtc:
            corners = self.rtc[rectangle]
            corners.append(corners[0])
            print(corners)
            obstacles.append(Polygon(corners))
        return obstacles

    def simplify_polygon(self, polygon):
        coordinates = np.array(polygon.exterior.coords)
        if len(coordinates) < 3:
            return polygon
        if (coordinates[0] == coordinates[-1]).all():
            coordinates = coordinates[:-1]
        ds = distance.cdist(coordinates, coordinates)
        start_point_index = 0
        end_point_index = 2
        e = 1e-8
        new_coors = []
        first_round = True
        while end_point_index != 2 or first_round:
            first_round = False
            if ds[start_point_index, end_point_index - 1] + ds[end_point_index - 1, end_point_index] >= ds[
                start_point_index, end_point_index] + e:
                start_point_index = end_point_index - 1
                new_coors.append(start_point_index)
            end_point_index = (end_point_index + 1) % len(coordinates)
        new_coors = coordinates[new_coors]
        new_polygon = Polygon(np.array(new_coors))
        return new_polygon

    def triangulate_section(self, section, obstacles):
        section_polygon = Polygon(section)
        if not section_polygon.is_valid:
            print(section)
            print(explain_validity(section_polygon))
        inner_obstacles = []
        for obstacle_index in range(len(obstacles)):
            obstacle = obstacles[obstacle_index]

            if abs(section_polygon.intersection(obstacle).area - obstacle.area) < 0.001:
                # obstacle = self.simplify_polygon(obstacle)
                # obstacle = obstacle.simplify(0.0, True)
                section_polygon = section_polygon.difference(obstacle)
                inner_obstacles.append(obstacle)
        if False:
            if type(section_polygon) == MultiPolygon:
                new_section_polygon = []
                for poly in section_polygon:
                    new_section_polygon.append(poly.simplify(0.0, True))
                section_polygon = MultiPolygon(new_section_polygon)
            else:
                section_polygon = section_polygon.simplify(0.0, True)
        t = triangulate(section_polygon)
        correct_triangles = []
        for x in t:
            if section_polygon.contains(x):
                correct_triangles.append(x)
        return correct_triangles, inner_obstacles

    def join_triangles_to_triangles(self, maximal_iterations, correct_triangles):
        extensible_polygons = True
        iterations = 0
        while extensible_polygons and iterations < maximal_iterations:
            extensible_polygons = False
            iterations += 1
            ds = np.array([np.array(tri.centroid) for tri in correct_triangles])
            used = np.zeros(len(ds))
            centers_tree = cKDTree(ds)
            union = []
            for d_index in range(len(ds)):
                d = ds[d_index]
                if used[d_index] == 1:
                    extensible_polygons = True
                    continue
                dd, ii = centers_tree.query(d, 5)
                found = False
                if len(ii) > 1:
                    for i in ii[1:]:
                        if i >= len(used):
                            break
                        if used[i] == 0 and type(
                                correct_triangles[i].intersection(correct_triangles[d_index])) is LineString:
                            polygon = correct_triangles[i].union(correct_triangles[d_index])
                            simplified_polygon = self.simplify_polygon(polygon)
                            if len(np.array(simplified_polygon.exterior.coords)) == 4:
                                union.append(simplified_polygon)
                                used[i] = 1
                                used[d_index] = 1
                                extensible_polygons = True
                                found = True
                                break
                if found:
                    continue
                union.append(correct_triangles[d_index])
                used[d_index] = 1
            correct_triangles = union
        return correct_triangles

    def join_triangles_to_rectangles(self, correct_triangles):
        ds = np.array([np.array(tri.centroid) for tri in correct_triangles])
        used = np.zeros(len(ds))
        centers_tree = cKDTree(ds)
        union = []
        for d_index in range(len(ds)):
            d = ds[d_index]
            if used[d_index] == 1:
                extensible_polygons = True
                continue
            dd, ii = centers_tree.query(d, 5)
            found = False
            for i in ii[1:]:
                if used[i] == 0 and type(
                        correct_triangles[i].intersection(correct_triangles[d_index])) is LineString:
                    coordinates_one = np.array(correct_triangles[i].exterior.coords)
                    coordinates_two = np.array(correct_triangles[d_index].exterior.coords)
                    dist_one = np.max(distance.cdist(coordinates_one, coordinates_one))
                    dist_two = np.max(distance.cdist(coordinates_two, coordinates_two))
                    if dist_one == dist_two:
                        polygon = correct_triangles[i].union(correct_triangles[d_index])
                        simplified_polygon = self.simplify_polygon(polygon)
                        union.append(simplified_polygon)
                        used[i] = 1
                        used[d_index] = 1
                        found = True
                        break
            if found:
                continue
            union.append(correct_triangles[d_index])
            used[d_index] = 1
        correct_triangles = union
        return correct_triangles

    def remove_bad_triangles(self, correct_triangles, inner_obstacles, min_distance=0.5):
        new_triangles = []
        for tri in correct_triangles:
            c = tri.centroid
            for o in inner_obstacles:
                if o.contains(c):
                    continue
            dd, ii = self.vertices_tree.query(np.array(c), 2)
            if dd[1] >= min_distance:
                new_triangles.append(tri)
        return new_triangles

    def reduce_triangles(self, maximal_iterations, min_distance, correct_triangles, inner_obstacles):
        correct_triangles = self.join_triangles_to_triangles(maximal_iterations, correct_triangles)
        correct_triangles = self.join_triangles_to_rectangles(correct_triangles)
        correct_triangles = self.remove_bad_triangles(correct_triangles, inner_obstacles, min_distance)
        return correct_triangles

    def create_bounds(self, bounds, count):
        all_bounds = []
        for c in range(count):
            for b in bounds:
                all_bounds.append(b)
        return all_bounds

    def create_regions(self, world_map, sections, obstacles):
        section_points = []
        HEIGHTS_MIN, HEIGHTS_MAX = 3, 16
        CAMERA_MIN, CAMERA_MAX = 0, np.deg2rad(self.camera_max_angle)
        YAW_MIN, YAW_MAX = 0, 2 * np.pi
        CXPB = 0.02
        MUTPB = 0.8
        NGEN = 100
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        inp = [(section, obstacles, world_map, HEIGHTS_MIN, CAMERA_MIN, YAW_MIN, HEIGHTS_MAX, CAMERA_MAX, YAW_MAX,
                NGEN, MUTPB, CXPB) for section in sections]
        section_points = pool.map(self.multiproc, inp)
        return section_points

    def multiproc(self, args):
        try:
            section, obstacles, world_map = args[0], args[1], args[2]
            HEIGHTS_MIN, CAMERA_MIN, YAW_MIN = args[3], args[4], args[5]
            HEIGHTS_MAX, CAMERA_MAX, YAW_MAX = args[6], args[7], args[8]
            NGEN, MUTPB, CXPB = args[9], args[10], args[11]
            correct_triangles, inner_obstacles = self.triangulate_section(section, obstacles)
            min_area = 0.5
            maximal_iterations = 50
            correct_triangles = self.reduce_triangles(maximal_iterations, min_area, correct_triangles, inner_obstacles)
            correct_triangles = np.array(correct_triangles)
            section_map, section_map_without_obstacles = self.section_map(section, world_map)
            free_pixels = np.count_nonzero(section_map)
            # bounds = [(0, len(correct_triangles) - 1), (3, 15), (0, np.pi / 3), (0, 2 * np.pi)]
            # points_count = (len(inner_obstacles) + 1) * 2
            # all_bounds = self.create_bounds(bounds, points_count)

            N_CYCLES = (len(inner_obstacles) + 1) * 2
            PLACES_MIN, PLACES_MAX = 0, len(correct_triangles) - 1

            MINS = [PLACES_MIN, HEIGHTS_MIN, CAMERA_MIN, YAW_MIN]
            MAXS = [PLACES_MAX - 1, HEIGHTS_MAX - 1, CAMERA_MAX, YAW_MAX]

            toolbox = base.Toolbox()
            # toolbox.register("map", pool.map)
            toolbox.register("attr_height", random.randint, HEIGHTS_MIN, HEIGHTS_MAX)
            toolbox.register("attr_camera", random.uniform, CAMERA_MIN, CAMERA_MAX)
            toolbox.register("attr_yaw", random.uniform, YAW_MIN, YAW_MAX)
            toolbox.register("mate", self.crossover, N_CYCLES=N_CYCLES)
            toolbox.register("mutate", self.mutation, N_CYCLES=N_CYCLES)
            toolbox.register("select", tools.selTournament, tournsize=3)
            toolbox.register("evaluate", self.fitness_function, free_pixels=free_pixels, world_map=world_map,
                             correct_triangles=correct_triangles, section_map=section_map, print_info=False)
            toolbox.decorate("mutate", self.checkBounds(MINS, MAXS))
            toolbox.register("attr_place", random.randint, PLACES_MIN, PLACES_MAX)
            toolbox.register("individual", tools.initCycle, creator.Individual,
                             (toolbox.attr_place, toolbox.attr_height, toolbox.attr_camera, toolbox.attr_yaw),
                             n=N_CYCLES)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            pop = toolbox.population(n=100)
            current_best = None
            current_best_value = float("inf")
            print("# individuals: " + str(N_CYCLES))
            print("# triangles: " + str(len(correct_triangles)))

            iteration = 0
            for g in range(NGEN):
                begin = time.time()
                iteration += 1
                var = 1.0 - float(iteration) / NGEN
                position_var = int(PLACES_MAX * var)
                height_var = int(12 * var)
                camera_var = var * CAMERA_MAX
                yaw_var = var * YAW_MAX

                # Select the next generation individuals
                offspring = toolbox.select(pop, len(pop))
                # Clone the selected individuals
                offspring = map(toolbox.clone, offspring)

                # Apply crossover on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < CXPB:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # print("Time crossover: " + str(time.time() - begin))
                # begin = time.time()

                # Apply mutation on the offspring
                for mutant in offspring:
                    if random.random() < MUTPB:
                        toolbox.mutate(mutant, position_var=position_var, height_var=height_var, camera_var=camera_var,
                                       yaw_var=yaw_var)
                        del mutant.fitness.values

                # print("Time mutation: " + str(time.time() - begin))
                # begin = time.time()

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # print("Time evaluation: " + str(time.time() - begin))
                # begin = time.time()

                # The population is entirely replaced by the offspring
                pop[:] = offspring
                fitnesses = [i.fitness.values for i in pop]
                minimal_value = np.min(fitnesses)
                if minimal_value < current_best_value:
                    current_best = pop[np.argmin(fitnesses)]
                    current_best_value = minimal_value
                print("iteration:" + str(iteration) + ", minimal value: " + str(minimal_value))
                print("Time: " + str(time.time() - begin))

            print("individual:", current_best)
            print("minimal value", current_best_value)
            points = np.reshape(np.array(current_best), (-1, 4))
            without_duplicates = []
            for p1 in points:
                found = False
                for p2 in without_duplicates:
                    if p1[0] == p2[0] and abs(p1[1] - p2[1]) <= 1 and abs(p1[2] - p2[2]) <= np.deg2rad(0.5) and abs(
                            p1[3] - p2[3]) <= np.deg2rad(15):
                        found = True
                        break
                if not found:
                    without_duplicates.append(p1)
            points = np.array(without_duplicates)
            triangles = correct_triangles[np.round(points[:, 0]).astype(np.int32)]
            xy = np.array([np.array(tri.centroid) for tri in triangles])
            max_distance = self.get_max_distance(xy)
            cluster = fclusterdata(xy, t=max_distance, criterion="distance")
            # print(cluster, len(cluster))
            clusters = [[] for _ in range(len(np.unique(cluster)))]
            # x, y, z, vertical orientation of camera, yaw
            for index in range(len(cluster)):
                clusters[cluster[index] - 1].append(
                    [xy[index, 0], xy[index, 1], points[index, 1], points[index, 2], points[index, 3]])
            # self.plot_regions()
            print(clusters)
            return clusters
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            raise e

    def plot_regions(self, section, correct_triangles, obstacles, xy, points):
        plt.figure()
        plt.fill(section[:, 0], section[:, 1], color=(0.9, 0.0, 0.0, 0.1))
        # Or if you want different settings for the grids:
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(-5, 5.5, 0.5)
        minor_ticks = np.arange(-5, 5.5, 0.5)

        plt.xticks(major_ticks)
        plt.yticks(minor_ticks)
        plt.grid(True)
        for tri in correct_triangles:
            x, y = tri.exterior.xy
            c = np.array(tri.centroid)
            plt.scatter(c[0], c[1])
            plt.plot(x, y)
        for o in obstacles:
            x, y = o.exterior.xy
            plt.fill(x, y, color=(0.1, 0.2, 0.5, 0.3))
        plt.scatter(xy[:, 0], xy[:, 1])
        for p_index in range(len(points)):
            xy_point = xy[p_index]
            end_vector = np.array(utils.Math.rotate_2d_vector(points[p_index, 3], [1, 0]) + xy_point)
            start_vector = xy_point
            # if tri.exterior.distance(tri.centroid) > median:
            plt.plot([start_vector[0], end_vector[0]], [start_vector[1], end_vector[1]])
        plt.show()
        plt.close()

    def section_map(self, section, world_map):
        map_points = []
        for point in section:
            map_x, map_y = world_map.get_index_on_map(point[0], point[1])
            map_points.append([map_x, map_y])
        section_map = np.zeros((world_map.width, world_map.height), np.uint8)
        obstacles_map = np.zeros((world_map.width, world_map.height), np.uint8)
        map_points = np.array(map_points).astype(np.int32)
        section_map = cv2.fillConvexPoly(section_map, map_points, 1)
        obstacles_map[world_map.target_obstacle_map > 0] = 1
        section_map_without_obstacles = np.bitwise_and(obstacles_map, section_map)
        return section_map, section_map_without_obstacles

    def distance_of_points_in_clusters(self, clusters):
        clusters_points_distance = 0
        for c in clusters:
            clusters_points_distance += np.sum(distance.pdist(c))
        return clusters_points_distance

    def distance_between_clusters(self, clusters):
        clusters_distance = 0
        for i in range(len(clusters)):
            for j in range(i, len(clusters)):
                if i != j:
                    clusters_distance += np.min(distance.cdist(clusters[i], clusters[j], 'euclidean'))
        return clusters_distance

    def create_clusters(self, xy):
        max_distance = self.get_max_distance(xy)
        cluster = fclusterdata(xy, t=max_distance, criterion="distance")
        # print(cluster, len(cluster))
        clusters = [[] for _ in range(len(np.unique(cluster)))]
        for index in range(len(cluster)):
            clusters[cluster[index] - 1].append(xy[index])
        clusters = np.array(clusters)
        return clusters

    def get_unseen_pixels(self, world_map, xy, points, free_pixels, section_map):
        seen_pixels = np.zeros((world_map.width, world_map.height))
        for triangle_index in range(len(xy)):
            position = Pose()
            position.position.x = xy[triangle_index, 0]
            position.position.y = xy[triangle_index, 1]
            position.position.z = np.round(points[triangle_index, 1])
            position.orientation.y = points[triangle_index, 2]
            position.orientation.z = points[triangle_index, 3]
            coordinates = world_map.faster_rectangle_ray_tracing_3d(position, self.vfov, self.hfov, self.camera_range)
            # print(coordinates)
            seen_pixels[coordinates[:, 0], coordinates[:, 1]] = 1
        seen_pixels_from_section = np.logical_and(seen_pixels, section_map)
        if False:
            print("xy", xy)
            print("points", points)
            print("fdf")
            seen_pixels_for_saving = np.zeros((world_map.width, world_map.height))
            seen_pixels_for_saving[seen_pixels > 0] = 255
            seen_pixels_for_saving[seen_pixels == 0] = 120
            section_map_for_saving = np.zeros((world_map.width, world_map.height))
            section_map_for_saving[section_map > 0] = 255
            cv2.imwrite("seen_pixels.png", seen_pixels_for_saving)
            cv2.imwrite("section_map.png", section_map_for_saving)
            # cv2.imwrite("seen_pixels_from_section.png", seen_pixels_from_section)
            plt.figure(figsize=(1, 1))
            major_ticks = np.arange(-5, 5.5, 0.5)
            minor_ticks = np.arange(-5, 5.5, 0.5)
            plt.xticks(major_ticks)
            plt.yticks(minor_ticks)
            plt.grid(True)
            plt.scatter(xy[:, 0], xy[:, 1])
            for p_index in range(len(points)):
                xy_point = xy[p_index]
                end_vector = np.array(utils.Math.rotate_2d_vector(points[p_index, 3], [1, 0]) + xy_point)
                start_vector = xy_point
                # if tri.exterior.distance(tri.centroid) > median:
                plt.plot([start_vector[0], end_vector[0]], [start_vector[1], end_vector[1]])
            plt.show()
            plt.close()
        unseen_pixels = free_pixels - np.count_nonzero(seen_pixels_from_section)
        return unseen_pixels

    def checkBounds(self, min, max):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    child_copy = np.reshape(child, (-1, 4))
                    child_copy = np.clip(child_copy, min, max)
                    child[:] = np.reshape(child_copy, (-1))[:]
                return offspring

            return wrapper

        return decorator

    def crossover(self, child1, child2, N_CYCLES):
        spot = random.randint(1, N_CYCLES)
        child1_copied = np.copy(child1)
        child1[:spot * 4 + 1] = child2[:spot * 4 + 1]
        child2[spot * 4:] = child1_copied[spot * 4:]
        return (child1, child2)

    def mutation(self, mutant, N_CYCLES, position_var, height_var, camera_var, yaw_var):
        mutant1 = np.reshape(mutant, (-1, 4))
        if position_var > 0:
            positions_change = np.random.randint(-position_var, position_var, N_CYCLES)
            mutant1[:, 0] += positions_change
        if height_var > 0:
            height_change = np.random.randint(-height_var, height_var, N_CYCLES)
            mutant1[:, 1] += height_change
        if camera_var > 0:
            camera_change = np.random.normal(0.0, camera_var, N_CYCLES)
            mutant1[:, 2] += camera_change
        if yaw_var > 0:
            yaw_change = np.random.normal(0.0, yaw_var, N_CYCLES)
            mutant1[:, 3] += yaw_change
        mutant[:] = np.reshape(mutant1, (-1))[:]
        return (mutant,)

    def fitness_function(self, original_points, free_pixels, world_map, correct_triangles, section_map, print_info):
        # free_pixels, world_map, correct_triangles, section_map, print_info = args
        points = np.reshape(np.array(original_points), (-1, 4))
        triangles = correct_triangles[np.round(points[:, 0]).astype(np.int32)]
        xy = np.array([np.array(tri.centroid) for tri in triangles])
        fitness = 0
        clusters = self.create_clusters(xy)
        # points distance inside cluster
        clusters_points_distance = self.distance_of_points_in_clusters(clusters)
        fitness += clusters_points_distance
        # minimal distance between clusters
        clusters_distance = self.distance_between_clusters(clusters)
        fitness += clusters_distance
        # not seen areas
        unseen_pixels = self.get_unseen_pixels(world_map, xy, points, free_pixels, section_map)
        if print_info:
            print("Sum of distance of points inside clusters: " + str(clusters_points_distance))
            print("Sum of distance clusters: " + str(clusters_distance))
            print("Unseen pixels in section: " + str(unseen_pixels))
            print("Free pixels: " + str(free_pixels))
        fitness += unseen_pixels * free_pixels
        return (fitness,)

    def get_max_distance(self, _):
        return 5

    def coloring(self, boundaries_sections):
        colors = {}
        areas = {}
        fresh_color = 0
        boundaries_areas = {}
        for s in boundaries_sections:
            for s1 in boundaries_sections[s]:
                for point in boundaries_sections[s][s1]:
                    found = False
                    for i1 in range(-1, 1):
                        for j1 in range(-1, 1):
                            if i1 + point[0] in colors and j1 + point[1] in colors[i1 + point[0]]:
                                found = True
                                self.add_new_point(point[0], point[1], colors[i1 + point[0]][j1 + point[1]], colors)
                                break
                        if found:
                            break
                    if not found:
                        self.add_new_point(point[0], point[1], fresh_color, colors)
                        fresh_color += 1
                for i in colors:
                    for j in colors[i]:
                        if colors[i][j] not in areas:
                            areas[colors[i][j]] = []
                        areas[colors[i][j]].append([i, j])
                for area in areas:
                    self.add_new_point(s, s1, area, boundaries_areas)

    def add_new_point(self, i, j, item, collection):
        if i not in collection:
            collection[i] = {}
        if j not in collection[i]:
            collection[i][j] = []
        if item not in collection[i][j]:
            collection[i][j].append(item)

    @staticmethod
    def pickle_sections(section_objects, section_file):
        with open(section_file, "wb") as fp:  # Pickling
            pck.dump(section_objects, fp)

    @staticmethod
    def unpickle_sections(section_file):
        with open(section_file, "rb") as fp:  # Unpickling
            return pck.load(fp)

    @staticmethod
    def show_sections(sections):
        print("SECTIONS")
        for (_, section) in sections.items():
            print("-section " + str(section.object_id))
            print("    NEIGHBORS")
            for neighbor in section.neighbors:
                print("    -neighbor " + str(neighbor) + " : " + str(section.neighbors[neighbor]))
            print("    REGIONS")
            for region_index in section.objects:
                region = section.objects[region_index]
                print("    -region " + str(region.object_id))
                print("       NEIGHBORS")
                for neighbor in region.neighbors:
                    print("       -neighbor " + str(neighbor) + " : " + str(region.neighbors[neighbor]))
                print("       POINTS")
                for point_index in region.objects:
                    point = region.objects[point_index]
                    print("       -point " + str(point.object_id))
                    print("          NEIGHBORS")
                    for neighbor in point.neighbors:
                        print("          -neighbor " + str(neighbor) + " : " + str(point.neighbors[neighbor]))

    @staticmethod
    def section_score(sections):
        print("SECTIONS")
        for (_, section) in sections.items():
            print("-section " + str(section.object_id) + "-- score: " + str(section.score) + ", entropy " + str(
                section.entropy) + ", visibility: " + str(section.visibility))
            print("    REGIONS")
            for region_index in section.objects:
                region = section.objects[region_index]
                print("    -region " + str(region.object_id) + "-- score: " + str(region.score) + ", entropy " + str(
                    region.entropy) + ", visibility: " + str(region.visibility))
                print("       POINTS")
                for point_index in region.objects:
                    point = region.objects[point_index]
                    print(
                        "       -point " + str(point.object_id) + "-- score: " + str(point.score) + ", entropy: " + str(
                            point.entropy) + ", visibility: " + str(point.visibility))

    @staticmethod
    def print_vertices(vertices, obstacles):
        plt.figure()
        SectionMap.draw_obstacles(obstacles)
        SectionMap.draw_verticies(vertices)
        plt.savefig("vertices_map.png")
        plt.close()
        return None

    @staticmethod
    def draw_verticies(vertices):
        plt.scatter(vertices[:, 0], vertices[:, 1], 3, "k")

    @staticmethod
    def draw_edges(vertices, edges):
        for v1 in range(len(edges)):
            for v2 in range(len(edges[v1])):
                if edges[v1, v2] > 0:
                    plt.plot([vertices[v1, 0], vertices[v2, 0]], [vertices[v1, 1], vertices[v2, 1]], "k")

    @staticmethod
    def print_edges(vertices, edges, obstacles):
        plt.figure()
        SectionMap.draw_obstacles(obstacles)
        SectionMap.draw_edges(vertices, edges)
        plt.savefig("edges_map.png")
        plt.close()
        return None

    @staticmethod
    def draw_section(sections):
        patterns = ['-', '+', 'x', '\\', '*', 'o', 'O', '.']
        for section in sections:
            pattern = np.random.choice(patterns, 1)[0]
            plt.fill(section[:, 0], section[:, 1], fill=True, hatch=pattern)

    @staticmethod
    def draw_obstacles(obstacles):
        for o in obstacles:
            x, y = o.exterior.xy
            plt.fill(x, y, color=(0.2, 0.2, 0.2, 0.3))

    @staticmethod
    def draw_list_obstacles(obstacles):
        for o in obstacles:
            plt.fill(o[:, 0], o[:, 1], color=(0.2, 0.2, 0.2, 0.3))

    @staticmethod
    def print_section(sections, obstacles):
        plt.figure()
        SectionMap.draw_obstacles(obstacles)
        SectionMap.draw_section(sections)
        plt.savefig("sections_map.png")
        plt.close()

    @staticmethod
    def print_orientation_points(sections, world_map):
        plt.figure()
        obstacles = SectionMap.obstacles_corners(world_map)
        SectionMap.draw_list_obstacles(obstacles)
        SectionMap.draw_orientation_points(sections)
        plt.savefig("orientation_point_map.png")
        plt.close()
        SectionMap.draw_visible_pixels(sections, world_map)

    @staticmethod
    def draw_orientation_points(sections):
        logic_configuration = rospy.get_param("logic_configuration")
        angle = np.deg2rad(logic_configuration["tracking"]["camera_max_angle"])
        for (_, section) in sections.items():
            for region_index in section.objects:
                region = section.objects[region_index]
                for point_index in region.objects:
                    point = region.objects[point_index]
                    x = point.data[0]
                    y = point.data[1]
                    z = point.data[2]
                    vc = point.data[3]
                    yaw = point.data[4]
                    size = np.interp(z, [3, 15], [5, 20]).astype(np.int32)
                    a = np.interp(vc, [0, angle], [0.1, 1])
                    plt.scatter(x, y, size, "k")
                    start_vector = np.array([x, y])
                    end_vector = np.array(utils.Math.rotate_2d_vector(yaw, [1, 0]) * a + start_vector)
                    plt.plot([start_vector[0], end_vector[0]], [start_vector[1], end_vector[1]])

    @staticmethod
    def draw_visible_pixels(sections, world_map):
        m = np.logical_not(np.copy(world_map.target_obstacle_map))
        for (_, section) in sections.items():
            for region_index in section.objects:
                region = section.objects[region_index]
                for point_index in region.objects:
                    point = region.objects[point_index]
                    for (di, d) in point.objects.items():
                        m[d[0], d[1]] = 1
        m = m.astype(np.uint8)
        m[m > 0] = 255
        cv2.imwrite("visibility_map.png", m)

    @staticmethod
    def obstacles_corners(world_map):
        corners = copy.copy(world_map.corners)
        iterations = int(len(corners) / 4)
        obstacles = []
        for i in range(iterations):
            obstacle = []
            for j in range(4):
                if i * 4 + j >= len(corners):
                    break
                corner = corners[i * 4 + j]
                obstacle.append(corner)
            obstacles.append(obstacle)
        return np.array(obstacles)


class SectionMapObjectTypes:
    Section = 0
    Region = 1
    Point = 2

    NAMES = ["Section", "Region", "Point"]

    @staticmethod
    def names(object_type):
        return SectionMapObjectTypes.NAMES[object_type]


class SectionMapObject(object):
    def __init__(self):
        super(SectionMapObject, self).__init__()
        self.object_id = 0
        self.score = 0
        self.entropy = 0
        self.neighbors = {}
        self.centroid = [0, 0]
        self.objects = {}
        self.type = SectionMapObjectTypes.Section
        self.data = None
        self.visibility = 0
        self.maximal_entropy = 0
        self.maximal_time = 0
