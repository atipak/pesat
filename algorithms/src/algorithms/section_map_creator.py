import os
import copy
import yaml
import json
from geometry_msgs.msg import Pose
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
from scipy.cluster.hierarchy import fclusterdata
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Voronoi, voronoi_plot_2d
import cv2
from shapely.geometry import Polygon, LineString, MultiPolygon, Point, MultiPoint
import numpy as np
from pesat_utils.pesat_math import Math
from pesat_utils.base_map import BaseMap
from pesat_utils.camera_calculation import CameraCalculation
from pesat_utils.graph import Graph
import time
import triangle
import matplotlib

matplotlib.use('Agg')

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
from multiprocessing import Queue
import pickle as pck
from copyreg import pickle
from types import MethodType
from section_algorithm_utils import SectionUtils
from section_algorithm_utils import SectionMapObjectTypes
from section_algorithm_utils import SectionMapObject

import graph_tool.all as gt

prefix = "../../../"
logic_configuration_file = prefix + "pesat_resources/config/logic_configuration.yaml"
drone_configuration_file = prefix + "pesat_resources/config/drone_configuration.yaml"
target_configuration_file = prefix + "pesat_resources/config/target_configuration.yaml"
environment_configuration_file = prefix + "pesat_resources/config/environment_configuration.yaml"


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
        self.load_yaml_files()
        _camera_configuration = self._drone_configuration["drone_configuration"]["camera"]
        logic_configuration = self._logic_configuration["logic_configuration"]
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
        self.shapely_obstacles = []
        self.edges = np.array([[]])
        self.delimiters = np.array([])
        self.free_pixels, self.world_map, self.correct_triangles = 0, None, []
        self.rtc = None
        self.section_map_image = None
        self.section_objects = None
        self.iterations = 0

    def load_yaml_files(self):
        with open(drone_configuration_file, 'r') as stream:
            try:
                self._drone_configuration = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        with open(environment_configuration_file, 'r') as stream:
            try:
                self._environment_configuration = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        with open(target_configuration_file, 'r') as stream:
            try:
                self._target_configuration = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        with open(logic_configuration_file, 'r') as stream:
            try:
                self._logic_configuration = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def create_sections_regions_and_points(self, map_file):
        environment_configuration = self._environment_configuration["environment_configuration"]
        max_target_velocity = self._target_configuration["target_configuration"]["strategies"]["max_velocity"]
        begin = time.time()
        world_map = BaseMap.map_from_file(map_file, 2)
        self.world_map = world_map
        print("World map was loaded", "during time", time.time() - begin)
        begin = time.time()
        sections, obstacles = self.create_sections(world_map, environment_configuration)
        self.calculate_sections_transitions(sections, obstacles)
        print("Section were created:", len(sections), "during time", time.time() - begin)
        regions = self.create_regions(world_map, sections, obstacles)
        # regions = [section1: [cluster1: [point1: [x, y, z, vertical orientation of camera, yaw],
        # point2, ...], cluster2, ...], section2, ...]
        sections_objects = {}
        section_index = 0
        region_index = 0
        point_index = 0
        point_coordinates = np.zeros((world_map.width, world_map.height))
        point_occupancy = [[[] for _ in range(world_map.height)] for _ in range(world_map.width)]
        built_up_coefficient = world_map.map_details["built_up"]
        half_target_velocity = max_target_velocity / 2.0
        regions_id = []
        sections_id = []
        points_id = []
        points_into_voronoi = []
        # logging
        self.areas = []
        self.circumferences = []
        self.regions_centers_list = []
        self.points_centers_sections_list = []
        self.points_heights_sections_list = []
        self.obstacles_in_section = []
        # end
        for section in sections:
            section_object = SectionMapObject()
            section_object.type = SectionMapObjectTypes.Section
            section_object.object_id = section_index
            polygon = Polygon(section)
            self.areas.append(polygon.area)
            self.circumferences.append(polygon.length)
            if polygon.contains(polygon.centroid):
                section_object.centroid = np.array(polygon.centroid)
            else:
                [np_polygon, _] = nearest_points(polygon, polygon.centroid)
                section_object.centroid = np.array(np_polygon)
            sections_objects[section_object.object_id] = section_object
            a = np.dstack(polygon.exterior.xy)[0]
            d = np.max(distance.cdist(a, a))
            section_object.maximal_time = d / ((1 - built_up_coefficient) * half_target_velocity)
            region_visibility = 0
            # logging
            regions_centers = []
            points_centers_list = []
            points_heights_list = []
            region_obstacles = []
            # end logging
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
                        a = np.dstack(polygon.exterior.xy)[0]
                        d = np.max(distance.cdist(a, a))
                # region_object.maximal_time = d / ((1 - built_up_coefficient) * half_target_velocity)
                region_object.centroid = polygon.centroid
                regions_centers.append(np.array(polygon.centroid))
                # logging
                points_centers = []
                points_heights = []
                region_obstacles.append(len(region))
                # end logging
                section_object.objects[region_object.object_id] = region_object
                points_visibility = 0
                points_times = []
                points_times_centers = []
                for point in region:
                    position = Pose()
                    position.position.x = point[0]
                    position.position.y = point[1]
                    position.position.z = point[2]
                    position.orientation.y = point[3]
                    position.orientation.z = point[4]
                    coordinates = world_map.faster_rectangle_ray_tracing_3d(position, self.vfov, self.hfov,
                                                                            self.camera_range)
                    point_object = SectionMapObject()
                    point_object.type = SectionMapObjectTypes.Point
                    point_object.object_id = point_index
                    data_id = 0
                    for coor in coordinates:
                        if section_map[coor[1], coor[0]] == 1:  # and point_coordinates[coor[0], coor[1]] != 1:
                            point_object.objects[data_id] = [coor[0], coor[1]]
                            point_occupancy[coor[0]][coor[1]].append(point_object.object_id)
                            point_coordinates[coor[0], coor[1]] = 1
                            data_id += 1
                    point_object.centroid = point[:2]
                    points_centers.append(point[:2])
                    points_heights.append(point[2])
                    point_object.data = point
                    point_object.maximal_time = self.calculate_time([o for i, o in point_object.objects.items()],
                                                                    half_target_velocity)
                    points_times.append(point_object.maximal_time)
                    points_times_centers.append(point_object.centroid)
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
                region_object.maximal_time = np.sum(np.sort(points_times)[:2])
                if len(points_times) > 1:
                    indices = np.argsort(points_times)[:2]
                    region_object.maximal_time += Math.array_euclidian_distance(points_times_centers[indices[0]],
                                                                                points_times_centers[
                                                                                    indices[1]]) / half_target_velocity
                points_centers_list.append(points_centers)
                points_heights_list.append(points_heights)
            section_index += 1
            section_object.visibility = region_visibility * np.square(world_map.box_size)
            section_object.maximal_entropy = np.clip(np.log2(region_visibility), 0, None)
            section_coordinates = np.nonzero(section_map)
            rx, ry = [], []
            for ri, ro in section_object.objects.items():
                rx.append(ro.centroid.xy[0])
                ry.append(ro.centroid.xy[1])
            x, y = world_map.get_index_on_map(np.array(rx), np.array(ry))
            section_object.maximal_time = self.calculate_time(list(zip(section_coordinates[1], section_coordinates[0])),
                                                              half_target_velocity, list(zip(x, y)))
            self.points_centers_sections_list.append(points_centers_list)
            self.points_heights_sections_list.append(points_heights_list)
            self.regions_centers_list.append(regions_centers)
            self.obstacles_in_section.append(np.sum(region_obstacles) / 2 - 1)
        self.add_neighbours(points_into_voronoi, sections_objects, sections_id, regions_id, points_id, world_map)
        self.non_seen_visible_pixels = point_coordinates.size - np.count_nonzero(point_coordinates) - len(
            world_map.obstacles_pixels[0])
        self.all_pixels = point_coordinates.size - len(world_map.obstacles_pixels[0])
        # has to be before func fill_unfilled
        self.section_intersections, self.region_intersections, self.point_intersections = self.calculate_objects_intersections(
            point_occupancy, sections_objects)
        self.fill_unfilled(world_map, point_occupancy)
        self.calculate_points_transitions(point_occupancy, world_map, sections_objects)
        self.section_transition, self.region_transition, self.point_transition = self.calculate_transition(sections,
                                                                                                           obstacles,
                                                                                                           point_occupancy,
                                                                                                           world_map,
                                                                                                           sections_objects)
        self.sections_objects = sections_objects

    def fill_unfilled(self, world_map, point_occupancy):
        for i in range(world_map.width):
            for j in range(world_map.height):
                if world_map.target_obstacle_map[i, j] == 0:
                    point_occupancy[i][j].append(-1)
        for i in range(world_map.width):
            for j in range(world_map.height):
                stack = []
                points = []
                if len(point_occupancy[i][j]) == 0:
                    for i0 in [-1, 0, 1]:
                        for j0 in [-1, 0, 1]:
                            self.assign_into_right_stack(point_occupancy[i + i0][j + j0], [i + i0, j + j0], stack,
                                                         points,
                                                         [world_map.width, world_map.height])
                for point in points:
                    for s in stack:
                        point_occupancy[point[0]][point[1]].append(s)

    def assign_into_right_stack(self, value, point, stack, points, maxis):
        if point[0] >= maxis[0] or point[0] < 0 or point[1] >= maxis[1] or point[1] < 0:
            return
        if len(value) == 1 and value[0] == -1:
            return
        if len(value) == 0:
            if point not in points:
                points.append(point)
        if len(value) > 0:
            for v in value:
                if v not in stack:
                    stack.append(v)
        return

    def calculate_objects_intersections(self, point_occupancy, sections_objects):
        points = 0
        regions = 0
        mapping = {}
        points_size = {}
        region_size = {}
        for index, section in sections_objects.items():
            for region_index, region in section.objects.items():
                regions += 1
                region_index_string = str(region_index)
                if region_index_string not in region_size:
                    region_size[region_index_string] = 0
                for point_index, point in region.objects.items():
                    points += 1
                    point_index_string = str(point_index)
                    if point_index not in mapping:
                        mapping[point_index] = region_index
                    if point_index_string not in points_size:
                        points_size[point_index_string] = len(point.objects)
                    region_size[region_index_string] += len(point.objects)
        point_intersections = np.zeros((points, points), dtype=np.int32)
        region_intersections = [[[] for _ in range(regions)] for _ in range(regions)]
        for i in range(len(point_occupancy)):
            for j in range(len(point_occupancy[i])):
                for k in range(len(point_occupancy[i][j])):
                    for l in range(k, len(point_occupancy[i][j])):
                        point_intersections[
                            min(point_occupancy[i][j][k], point_occupancy[i][j][l]), max(point_occupancy[i][j][k],
                                                                                         point_occupancy[i][j][l])] += 1
                        rr = region_intersections[
                            min(mapping[point_occupancy[i][j][k]], mapping[point_occupancy[i][j][l]])][max(
                                mapping[point_occupancy[i][j][k]], mapping[point_occupancy[i][j][l]])]
                        if [min(i, j), max(i, j)] not in rr:
                            rr.append([min(i, j), max(i, j)])
        pi = {}
        ri = {}
        si = {}
        for i in range(len(sections_objects)):
            section_index = str(i)
            if section_index not in si:
                si[section_index] = {}
        for i in range(regions):
            for j in range(regions):
                if len(region_intersections[i][j]) > 0 and i != j:
                    region_i_index = str(i)
                    region_j_index = str(j)
                    if region_i_index not in ri:
                        ri[region_i_index] = {}
                    if region_j_index not in ri[region_i_index]:
                        ri[region_i_index][region_j_index] = len(region_intersections[i][j])
                    else:
                        print("Error {},{}".format(region_i_index, region_j_index))
                    if region_j_index not in ri:
                        ri[region_j_index] = {}
                    if region_i_index not in ri[region_j_index]:
                        ri[region_j_index][region_i_index] = len(region_intersections[i][j])
                    else:
                        print("Error {},{}".format(region_i_index, region_j_index))

        x, y = np.nonzero(point_intersections)
        for i in range(len(x)):
            if x[i] == y[i]:
                continue
            point_i_index = str(x[i])
            point_j_index = str(y[i])
            if point_i_index not in pi:
                pi[point_i_index] = {}
            if point_j_index not in pi[point_i_index]:
                pi[point_i_index][point_j_index] = 0

            if point_j_index not in pi:
                pi[point_j_index] = {}
            if point_i_index not in pi[point_j_index]:
                pi[point_j_index][point_i_index] = 0
            pi[point_i_index][point_j_index] += point_intersections[x[i], y[i]]
            pi[point_j_index][point_i_index] += point_intersections[x[i], y[i]]

        """
            region_i_index = str(mapping[x[i]])
            region_j_index = str(mapping[y[i]])
            if region_i_index not in ri:
                ri[region_i_index] = {}
            if region_j_index not in ri[region_i_index]:
                ri[region_i_index][region_j_index] = 0

            if region_j_index not in ri:
                ri[region_j_index] = {}
            if region_i_index not in ri[region_j_index]:
                ri[region_j_index][region_i_index] = 0
            ri[region_j_index][region_i_index] += point_intersections[x[i], y[i]]
            ri[region_i_index][region_j_index] += point_intersections[x[i], y[i]]
        """

        for region in ri:
            for region1 in ri[region]:
                if ri[region][region1] > 0 and region != region1:
                    ri[region][region1] = ri[region][region1] / region_size[region1]
                else:
                    raise Exception("This is exception")
        for i in range(regions):
            i_string = str(i)
            if i_string not in ri:
                ri[i_string] = {}
        for point in pi:
            for point1 in pi[point]:
                if pi[point][point1] > 0 and point1 != point:
                    pi[point][point1] = pi[point][point1] / points_size[point1]
                else:
                    raise Exception("This is exception")
        for i in range(points):
            i_string = str(i)
            if i_string not in pi:
                pi[i_string] = {}
        return si, ri, pi

    def calculate_sections_transitions(self, sections, obstacles):
        points = {}
        index = 0
        for section in sections:
            for point in section:
                if point[0] not in points:
                    points[point[0]] = {}
                if point[1] not in points[point[0]]:
                    points[point[0]][point[1]] = []
                points[point[0]][point[1]].append(index)
            index += 1
        transitions = np.zeros((len(sections), len(sections)))
        for section_index in range(len(sections)):
            section = sections[section_index]
            for index in range(len(section)):
                point = section[index % len(section)]
                next_point = section[(index + 1) % len(section)]
                line = LineString([Point(point[0], point[1]), Point(next_point[0], next_point[1])])
                if line.length <= 1:
                    continue
                intersection_indices = np.in1d(points[point[0]][point[1]], points[next_point[0]][next_point[1]])
                intersection = np.array(points[point[0]][point[1]])[intersection_indices]
                for obstacle in obstacles:
                    res = line.difference(obstacle)
                    for ii in intersection:
                        if ii != section_index:
                            transitions[section_index, ii] += res.length
        tt = (transitions.T / transitions.sum(axis=0)).T
        transition_dict = {}
        for i in range(len(transitions)):
            transition_dict[str(i)] = {}
            for j in range(len(transitions)):
                if tt[i, j] > 0:
                    transition_dict[str(i)][str(j)] = tt[i, j]
        return transition_dict

    def calculate_points_transitions(self, point_occupancy, world_map, sections_objects):
        occupancy = {}
        for i in range(world_map.width):
            for j in range(world_map.height):
                for value in point_occupancy[i][j]:
                    if value == -1:
                        continue
                    if value not in occupancy:
                        occupancy[value] = {}
                    for i0 in [-1, 0, 1]:
                        for j0 in [-1, 0, 1]:
                            if i + i0 >= world_map.width or i + i0 < 0 or j + j0 >= world_map.height or j + j0 < 0:
                                continue
                            if len(point_occupancy[i + i0][j + j0]) >= 1 and value not in point_occupancy[i + i0][
                                j + j0]:
                                for value1 in point_occupancy[i + i0][j + j0]:
                                    if value1 != -1:
                                        if value1 not in occupancy[value]:
                                            occupancy[value][value1] = []
                                        if [i + i0, j + j0] not in occupancy[value][value1]:
                                            occupancy[value][value1].append([i + i0, j + j0])
        point_borders = {}
        for point in occupancy:
            point_borders[point] = {}
            point_dict = occupancy[point]
            s = 0.0
            for p1 in point_dict:
                s += len(point_dict[p1])
            for p1 in point_dict:
                point_borders[point][p1] = len(point_dict[p1]) / s
        region_borders = {}
        regions = {}
        points = {}
        for index, section in sections_objects.items():
            for region_index, region in section.objects.items():
                for point_index, point in region.objects.items():
                    if region_index not in regions:
                        regions[region_index] = []
                    if point_index not in points:
                        points[point_index] = 0
                    regions[region_index].append(point_index)
                    points[point_index] = region_index
        for region in regions:
            if region not in region_borders:
                region_borders[region] = {}
            for point in regions[region]:
                if point in occupancy:
                    for point1 in occupancy[point]:
                        region1 = points[point1]
                        if region1 not in region_borders[region]:
                            region_borders[region][region1] = []
                        region_borders[region][region1].append(len(occupancy[point][point1]))
        for index, section in sections_objects.items():
            for region_index, region in section.objects.items():
                if region_index not in region_borders:
                    region_borders[region] = {}
                for point_index, point in region.objects.items():
                    if point_index not in point_borders:
                        point_borders[point_index] = {}
        for r in region_borders:
            s = 0.0
            for r1 in region_borders[r]:
                s += np.sum(region_borders[r][r1])
            for r1 in region_borders[r]:
                region_borders[r][r1] = np.sum(region_borders[r][r1]) / s
        return region_borders, point_borders

    def calculate_transition(self, sections, obstacles, point_occupancy, world_map, sections_objects):
        region_transition, point_transition = self.calculate_points_transitions(point_occupancy, world_map,
                                                                                sections_objects)
        section_transition = self.calculate_sections_transitions(sections, obstacles)
        return section_transition, region_transition, point_transition

    def add_neighbours(self, points_into_voronoi, sections_objects, sections_id, regions_id, points_id, world_map):
        vor = Voronoi(np.array(points_into_voronoi))
        point_d = {}
        for i in range(len(points_into_voronoi)):
            p = points_into_voronoi[i]
            if p[0] not in point_d:
                point_d[p[0]] = {}
            if p[1] not in point_d[p[0]]:
                point_d[p[0]][p[1]] = []
            point_d[p[0]][p[1]].append(i)
        for rp_index in range(len(vor.ridge_points)):
            ridge_point = vor.ridge_points[rp_index]
            if -1 in vor.ridge_vertices[rp_index]:
                if vor.ridge_vertices[rp_index][0] == -1:
                    index = vor.ridge_vertices[rp_index][1]
                else:
                    index = vor.ridge_vertices[rp_index][0]
                other_point = vor.vertices[index]
                if other_point[0] < -world_map.real_width / 2.0 or other_point[0] > world_map.real_width / 2.0 or \
                        other_point[1] < -world_map.real_height / 2.0 or other_point[1] > world_map.real_height / 2.0:
                    continue
            pp1 = points_into_voronoi[ridge_point[0]]
            pp2 = points_into_voronoi[ridge_point[1]]
            pp1_list = point_d[pp1[0]][pp1[1]]
            pp2_list = point_d[pp2[0]][pp2[1]]
            for i in pp1_list:
                for j in pp2_list:
                    self.create_neighbours([i, j], sections_objects, sections_id, regions_id, points_id)
        voronoi_plot_2d(vor)
        plt.show()
        for key_i in point_d:
            for key_j in point_d[key_i]:
                l = point_d[key_i][key_j]
                for i in range(len(l)):
                    for j in range(len(l)):
                        if i != j:
                            self.create_neighbours([l[i], l[j]], sections_objects, sections_id, regions_id, points_id)

    def create_neighbours(self, ridge_point, sections_objects, sections_id, regions_id, points_id):
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
            if [r1, r, p1, p] not in sections_objects[s1].neighbors[s]:
                sections_objects[s1].neighbors[s].append([r1, r, p1, p])
        if r1 != r:
            if r1 not in sections_objects[s].objects[r].neighbors:
                sections_objects[s].objects[r].neighbors[r1] = []
            if [p, p1] not in sections_objects[s].objects[r].neighbors[r1]:
                sections_objects[s].objects[r].neighbors[r1].append([p, p1])
            if r not in sections_objects[s1].objects[r1].neighbors:
                sections_objects[s1].objects[r1].neighbors[r] = []
            if [p1, p] not in sections_objects[s1].objects[r1].neighbors[r]:
                sections_objects[s1].objects[r1].neighbors[r].append([p1, p])
        if p != p1:
            if p1 not in sections_objects[s].objects[r].objects[p].neighbors:
                sections_objects[s].objects[r].objects[p].neighbors[p1] = []
            if [p, p1] not in sections_objects[s].objects[r].objects[p].neighbors[p1]:
                sections_objects[s].objects[r].objects[p].neighbors[p1].append([p, p1])
            if p not in sections_objects[s1].objects[r1].objects[p1].neighbors:
                sections_objects[s1].objects[r1].objects[p1].neighbors[p] = []
            if [p1, p] not in sections_objects[s1].objects[r1].objects[p1].neighbors[p]:
                sections_objects[s1].objects[r1].objects[p1].neighbors[p].append([p1, p])

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
        self.vertices = verticies
        obstacles = self.get_shapely_obstacles(verticies, delimiters)
        self.shapely_obstacles = obstacles
        print("Vertices for graph created:", len(verticies), "during time", time.time() - begin)
        # self.print_vertices(verticies, obstacles)
        begin = time.time()
        edges, unchanged_edges = self.create_edges(verticies, delimiters, world_map, environment_configuration)
        self.edges = edges
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
                lower_index, upper_index = int(lower_index), int(upper_index)
                path_upper_to_lower = Graph.get_path(predecessors, section[lower_index], section[upper_index])
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
        self.vertex_sections = vertex_sections
        # sections info logging
        self.borders_counts, self.borders_sums = self.compute_borders_price(unchanged_edges, sections)
        return vertex_sections, obstacles

    def compute_borders_price(self, edges, sections):
        borders_counts = []
        borders_sums = []
        for section in sections:
            rolled_section = np.roll(section, 1)
            edges_values = edges[rolled_section, section]
            mask = np.logical_and(edges_values > 0, edges_values != 1)
            borders_counts.append(np.count_nonzero(mask))
            borders_sums.append(np.sum(edges_values[mask]))
        return borders_counts, borders_sums

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
        if np.count_nonzero(under_vision_map) == 0:
            under_vision_map = None
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
                if under_vision_map is not None:
                    edges[index, v] = self.calculate_distance(verticies[index], verticies[v], under_vision_map,
                                                              world_map)
                else:
                    edges[index, v] = Math.array_euclidian_distance(verticies[index], verticies[v])
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
        edges_copy = copy.deepcopy(edges)
        self.recalculate_edges(edges)
        return edges, edges_copy

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
                    coordinates, _ = world_map.faster_rectangle_ray_tracing_3d(position, vfov, hfov, camera_range)
                    sectors_map = CameraCalculation.write_coordinates(coordinates, sectors_map)
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
                distance = Math.euclidian_distance(Math.Point(corners[current][0], corners[current][1]),
                                                   Math.Point(corners[next][0], corners[next][1]))
                if distance > 1:
                    vector = Math.normalize(vector)
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

    def calculate_time(self, coordinates, half_target_velocity, sources=None):
        mapping_coor_to_vertex = {}
        mapping_vertex_to_coor = []
        index = 0
        for o in coordinates:
            if o[0] not in mapping_coor_to_vertex:
                mapping_coor_to_vertex[o[0]] = {}
            if o[1] not in mapping_coor_to_vertex[o[0]]:
                mapping_coor_to_vertex[o[0]][o[1]] = index
                mapping_vertex_to_coor.append(o)
                index += 1
        c = np.sqrt(2) * 0.5
        graph = np.zeros((index, index))
        for v_index in range(index):
            coor = mapping_vertex_to_coor[v_index]
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == j:
                        continue
                    if (coor[0] + i) in mapping_coor_to_vertex and (coor[1] + j) in mapping_coor_to_vertex[
                        (coor[0] + i)]:
                        v1_index = mapping_coor_to_vertex[(coor[0] + i)][(coor[1] + j)]
                        if np.sum(np.abs([i, j])) == 2:
                            graph[v_index, v1_index] = c
                            graph[v1_index, v_index] = c
                        else:
                            graph[v_index, v1_index] = 0.5
                            graph[v1_index, v_index] = 0.5
        graph = csr_matrix(graph)
        dist_matrix = np.array(shortest_path(csgraph=graph, directed=False))
        if sources is None:
            indices = np.arange(0, index)
        else:
            indices = []
            for coordinate_index in range(len(coordinates)):
                coordinate = coordinates[coordinate_index]
                if coordinate in sources:
                    indices.append(coordinate_index)
            indices = np.array(indices)
        a = dist_matrix[indices]
        a = a[:, indices]
        a[a == np.inf] = 0
        maximum = 0
        if len(a) > 0:
            maximum = np.max(a)
        return maximum / half_target_velocity

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
            print("Section {} is not valid.".format(section))
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
            if not x.is_valid:
                print("Triangle {} is not valid. Skipped".format(x.xy))
                print(explain_validity(x))
            else:
                if section_polygon.contains(x):
                    correct_triangles.append(x)
        return correct_triangles, inner_obstacles

    def join_triangles_to_triangles(self, maximal_iterations, correct_triangles):
        if len(correct_triangles) > 0:
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
        if len(correct_triangles) > 0:
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
                    if i == len(used):
                        break
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
        MUTPB = 0.7
        NGEN = 100
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        m = multiprocessing.Manager()
        q = m.Queue()
        inp = [(section, obstacles, world_map, HEIGHTS_MIN, CAMERA_MIN, YAW_MIN, HEIGHTS_MAX, CAMERA_MAX, YAW_MAX,
                NGEN, MUTPB, CXPB, q) for section in sections]
        result = pool.map(self.multiproc, inp)
        self.after_reduction = []
        self.before_reduction = []
        section_points = []
        self.inner_obstacles = []
        for r in result:
            self.before_reduction.append(r[1])
            self.after_reduction.append(r[2])
            section_points.append(r[0])
            self.inner_obstacles.append(r[3])
        return section_points

    def multiproc(self, args):
        try:
            section, obstacles, world_map = args[0], args[1], args[2]
            HEIGHTS_MIN, CAMERA_MIN, YAW_MIN = args[3], args[4], args[5]
            HEIGHTS_MAX, CAMERA_MAX, YAW_MAX = args[6], args[7], args[8]
            NGEN, MUTPB, CXPB, q = args[9], args[10], args[11], args[12]
            correct_triangles, inner_obstacles = self.triangulate_section(section, obstacles)
            min_distance = 2.0
            maximal_iterations = 50
            before_reduction = len(correct_triangles)
            correct_triangles = self.reduce_triangles(maximal_iterations, min_distance, correct_triangles,
                                                      obstacles)
            correct_triangles = np.array(correct_triangles)
            after_reduction = len(correct_triangles)
            if after_reduction < 1:
                q.put(True)
                raise Exception("No triangles")
            section_map, section_map_without_obstacles = self.section_map(section, world_map)
            free_pixels = np.count_nonzero(section_map)
            # bounds = [(0, len(correct_triangles) - 1), (3, 15), (0, np.pi / 3), (0, 2 * np.pi)]
            # points_count = (len(inner_obstacles) + 1) * 2
            # all_bounds = self.create_bounds(bounds, points_count)
            cache = {}

            N_CYCLES = (len(inner_obstacles) + 1) * 2
            PLACES_MIN, PLACES_MAX = 0, len(correct_triangles) - 1
            if after_reduction == 1:
                PLACES_MAX = 1

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
                             correct_triangles=correct_triangles, section_map=section_map, print_info=False,
                             cache=cache)
            toolbox.decorate("mutate", self.checkBounds(MINS, MAXS))
            toolbox.register("attr_place", random.randint, PLACES_MIN, PLACES_MAX)
            toolbox.register("individual", tools.initCycle, creator.Individual,
                             (toolbox.attr_place, toolbox.attr_height, toolbox.attr_camera, toolbox.attr_yaw),
                             n=N_CYCLES)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            pop = toolbox.population(n=100)
            current_best = None
            current_best_value = float("inf")
            last_change = 0
            maximal_iteration_timeout = 20

            print("# individuals: " + str(N_CYCLES))
            print("# triangles: " + str(len(correct_triangles)))

            iteration = 0
            for g in range(NGEN):
                if not q.empty():
                    raise Exception("One of the process ended with exception.")
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
                offspring = list(map(toolbox.clone, offspring))

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
                    last_change = iteration
                iteration_timeout = np.abs(last_change - iteration)
                if iteration_timeout > maximal_iteration_timeout:
                    break
                print("Thread id {}, iteration {}, minimal value {}, time {}, timeout {}".format(
                    multiprocessing.current_process().pid, iteration, minimal_value,
                    time.time() - begin, iteration_timeout))

            print("individual:", current_best)
            print("minimal value", current_best_value)
            points = np.reshape(np.array(current_best), (-1, 4))
            final_set = self.remove_redundant(points, world_map, correct_triangles)
            points = np.array(final_set)
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
            return [clusters, before_reduction, after_reduction, len(inner_obstacles)]
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            raise e

    def remove_redundant(self, points, world_map, correct_triangles):
        before_removing_duplicates = len(points)
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
        after_removing_duplicates = len(points)
        all_xy = np.array([np.array(tri.centroid) for tri in correct_triangles])
        visible_points = []
        index = 0
        for point in points:
            position = Pose()
            position.position.x = all_xy[int(point[0]), 0]
            position.position.y = all_xy[int(point[0]), 1]
            position.position.z = point[1]
            position.orientation.y = point[2]
            position.orientation.z = point[3]
            coordinates = world_map.faster_rectangle_ray_tracing_3d(position, self.vfov, self.hfov,
                                                                    self.camera_range)
            visible_points.append((index, np.array(coordinates)))
            index += 1
        points_sorted = sorted(visible_points, key=lambda x: len(x[1]))[::-1]
        intersections = []
        removed = []
        added = []
        for i in range(len(points_sorted)):
            s = points_sorted[i]
            if s[0] not in removed:
                added.append(s[0])
                for j in range(i + 1, len(points_sorted)):
                    s1 = points_sorted[j]
                    if s1[0] not in removed:
                        intersection = self.compare_intersection_of_visible_points(s[1], s1[1])
                        if len(s1[1]) == 0:
                            intersections.append(0)
                        else:
                            intersections.append(intersection / len(s1[1]))
                        if len(s1[1]) == 0 or intersection / len(s1[1]) > 0.98:
                            removed.append(s1[0])
        print("Before removing duplicates {}, after removing duplicates {}, average intersaction value {}".format(
            before_removing_duplicates, after_removing_duplicates, np.average(intersections)))
        print("Max value in intersection {}, min value in intersection {}, median value in intesection {}".format(
            np.max(intersections), np.min(intersections), np.median(intersections)))
        print("Points after removing intersections points {}".format(len(added)))
        seen = None
        added_new = []
        for i in range(len(points_sorted)):
            s = points_sorted[i]
            if s[0] in added:
                if seen is None:
                    seen_new = s[1]
                    seen = []
                else:
                    seen_new = np.unique(np.concatenate((s[1], seen)), axis=0)
                if abs(len(seen) - len(seen_new)) > 5:
                    added_new.append(s[0])
                    seen = seen_new
        print("Points after removing redundant points {}".format(len(added_new)))
        final_set = []
        for i in added_new:
            final_set.append(points[i])
        return final_set

    def compare_intersection_of_visible_points(self, A, B):
        nrows, ncols = A.shape
        dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                 'formats': ncols * [A.dtype]}

        C = np.intersect1d(A.view(dtype), B.view(dtype))

        # This last bit is optional if you're okay with "C" being a structured array...
        C = C.view(A.dtype).reshape(-1, ncols)
        return len(C)

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
            end_vector = np.array(Math.rotate_2d_vector(points[p_index, 3], [1, 0]) + xy_point)
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

    def round_yaw(self, yaw):
        f = 0.0625
        return int(yaw / f)

    def round_vertical(self, vertical):
        v = 0.03125
        return int(vertical / v)

    def get_unseen_pixels(self, world_map, xy, points, free_pixels, section_map, cache):
        seen_pixels = np.zeros((world_map.width, world_map.height))
        # start = time.time()
        for triangle_index in range(len(xy)):
            v = self.round_vertical(points[triangle_index, 2])
            yaw = self.round_yaw(points[triangle_index, 3])
            if xy[triangle_index, 0] in cache and \
                    xy[triangle_index, 1] in cache[xy[triangle_index, 0]] and \
                    np.round(points[triangle_index, 1]) in cache[xy[triangle_index, 0]][xy[triangle_index, 1]] and \
                    v in cache[xy[triangle_index, 0]][xy[triangle_index, 1]][np.round(points[triangle_index, 1])] and \
                    yaw in cache[xy[triangle_index, 0]][xy[triangle_index, 1]][np.round(points[triangle_index, 1])][v]:
                coordinates = \
                    cache[xy[triangle_index, 0]][xy[triangle_index, 1]][np.round(points[triangle_index, 1])][v][yaw]
            else:
                position = Pose()
                position.position.x = xy[triangle_index, 0]
                position.position.y = xy[triangle_index, 1]
                position.position.z = np.round(points[triangle_index, 1])
                position.orientation.y = points[triangle_index, 2]
                position.orientation.z = points[triangle_index, 3]
                coordinates = world_map.faster_rectangle_ray_tracing_3d(position, self.vfov, self.hfov,
                                                                        self.camera_range)
                if xy[triangle_index, 0] not in cache:
                    cache[xy[triangle_index, 0]] = {}
                if xy[triangle_index, 1] not in cache[xy[triangle_index, 0]]:
                    cache[xy[triangle_index, 0]][xy[triangle_index, 1]] = {}
                if np.round(points[triangle_index, 1]) not in cache[xy[triangle_index, 0]][xy[triangle_index, 1]]:
                    cache[xy[triangle_index, 0]][xy[triangle_index, 1]][np.round(points[triangle_index, 1])] = {}
                if v not in cache[xy[triangle_index, 0]][xy[triangle_index, 1]][np.round(points[triangle_index, 1])]:
                    cache[xy[triangle_index, 0]][xy[triangle_index, 1]][np.round(points[triangle_index, 1])][v] = {}
                if yaw not in cache[xy[triangle_index, 0]][xy[triangle_index, 1]][np.round(points[triangle_index, 1])][
                    v]:
                    cache[xy[triangle_index, 0]][xy[triangle_index, 1]][np.round(points[triangle_index, 1])][v][
                        yaw] = coordinates
                # print(coordinates)
            seen_pixels[coordinates[:, 1], coordinates[:, 0]] = 1
        # print("Tracing {}".format(time.time() - start))
        # start = time.time()
        seen_pixels_from_section = np.logical_and(seen_pixels, section_map)
        # print("Logical and {}".format(time.time() - start))
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
                end_vector = np.array(Math.rotate_2d_vector(points[p_index, 3], [1, 0]) + xy_point)
                start_vector = xy_point
                # if tri.exterior.distance(tri.centroid) > median:
                plt.plot([start_vector[0], end_vector[0]], [start_vector[1], end_vector[1]])
            plt.show()
            plt.close()
        # start = time.time()
        unseen_pixels = free_pixels - np.count_nonzero(seen_pixels_from_section)
        # print("Count nonzero {}".format(time.time() - start))
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

    def fitness_function(self, original_points, free_pixels, world_map, correct_triangles, section_map, print_info,
                         cache):
        # free_pixels, world_map, correct_triangles, section_map, print_info = args
        # start = time.time()
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
        # print("Clusters {}".format(time.time() - start))
        # start = time.time()
        # not seen areas
        unseen_pixels = self.get_unseen_pixels(world_map, xy, points, free_pixels, section_map, cache)
        if print_info:
            print("Sum of distance of points inside clusters: " + str(clusters_points_distance))
            print("Sum of distance clusters: " + str(clusters_distance))
            print("Unseen pixels in section: " + str(unseen_pixels))
            print("Free pixels: " + str(free_pixels))
        fitness += unseen_pixels * free_pixels
        # print("Visibilioty {}".format(time.time() - start))
        return (fitness,)

    def compute_statistics(self):
        sections = []
        for i in range(len(self.areas)):
            section = {}
            section["area"] = self.areas[i]
            section["perimeter"] = self.circumferences[i]
            section["obstacles"] = self.obstacles_in_section[i]
            section["clusters_count"] = len(self.regions_centers_list[i])
            if len(self.regions_centers_list[i]) == 1:
                section["region_distance"] = 0
            else:
                d = distance.pdist(np.array(self.regions_centers_list[i]))
                section["region_distance"] = np.average(d)
            points_distances = []
            points_centers_list = self.points_centers_sections_list[i]
            for j in range(len(points_centers_list)):
                if len(np.array(points_centers_list[j])) == 1:
                    points_distances.append(0)
                else:
                    points_distances.append(np.mean(distance.pdist(np.array(points_centers_list[j]))))
            section["points_distance"] = points_distances
            section["points_before_reduction"] = self.before_reduction[i]
            section["points_after_reduction"] = self.after_reduction[i]
            section["points_heights"] = self.points_heights_sections_list[i]
            section["unseen_visible_pixels"] = self.non_seen_visible_pixels
            section["all_pixels"] = self.all_pixels
            section["borders_sums"] = self.borders_sums[i]
            section["borders_counts"] = self.borders_counts[i]
            sections.append(section)
        return sections

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

    #########
    def pickle_sections(self, section_file):
        if self.sections_objects is not None:
            print("section_objects", self.sections_objects)
            d = {
                "objects": self.sections_objects,
                "additional": {
                    "transitions": {
                        "sections": self.section_transition,
                        "regions": self.region_transition,
                        "points": self.point_transition
                    },
                    "intersections": {
                        "sections": self.section_intersections,
                        "regions": self.region_intersections,
                        "points": self.point_intersections
                    }

                }
            }
            with open(section_file, "wb") as fp:  # Pickling
                pck.dump(d, fp, protocol=2)
        else:
            raise Exception("Sections not found, did you create them?")

    def print_vertices(self, path_to_file):
        plt.figure()
        self.draw_obstacles()
        self.draw_verticies()
        plt.savefig(path_to_file)
        # plt.savefig("vertices_map.png")
        plt.close()
        return None

    def draw_verticies(self):
        plt.scatter(self.vertices[:, 0], self.vertices[:, 1], 3, "k")

    def draw_edges(self, max_level_dim_zero, max_level_dim_one):
        if max_level_dim_zero <= -1 or max_level_dim_zero >= len(self.edges):
            max_level_dim_zero = len(self.edges)
        if max_level_dim_one <= -1 or max_level_dim_one >= len(self.edges[0]):
            max_level_dim_one = len(self.edges[0])
        for v1 in range(max_level_dim_zero):
            for v2 in range(max_level_dim_one):
                if self.edges[v1, v2] > 0:
                    plt.plot([self.vertices[v1, 0], self.vertices[v2, 0]], [self.vertices[v1, 1], self.vertices[v2, 1]],
                             "k")

    def print_edges(self, path_to_file, max_level_dim_zero, max_level_dim_one):
        plt.figure()
        self.draw_obstacles()
        self.draw_edges(max_level_dim_zero, max_level_dim_one)
        plt.savefig(path_to_file)
        # plt.savefig("edges_map.png")
        plt.close()
        return None

    def draw_section(self):
        patterns = ['-', '+', 'x', '\\', '*', 'o', 'O', '.']
        for section in self.vertex_sections:
            pattern = np.random.choice(patterns, 1)[0]
            plt.fill(section[:, 0], section[:, 1], fill=True, hatch=pattern)

    def draw_obstacles(self):
        for o in self.shapely_obstacles:
            x, y = o.exterior.xy
            plt.fill(x, y, color=(0.2, 0.2, 0.2, 0.3))

    def draw_list_obstacles(self):
        obstacles = self.obstacles_corners()
        for o in obstacles:
            plt.fill(o[:, 0], o[:, 1], color=(0.2, 0.2, 0.2, 0.3))

    def print_section(self, path_to_file):
        plt.figure()
        self.draw_obstacles()
        self.draw_section()
        plt.savefig(path_to_file)
        # plt.savefig("sections_map.png")
        plt.close()

    def print_orientation_points(self, path_to_file, path_to_file_visible_points):
        plt.figure()
        self.draw_list_obstacles()
        self.draw_orientation_points()
        plt.savefig(path_to_file)
        # plt.savefig("orientation_point_map.png")
        plt.close()
        self.draw_visible_pixels(path_to_file_visible_points)

    def draw_orientation_points(self):
        angle = np.deg2rad(self._logic_configuration["logic_configuration"]["tracking"]["camera_max_angle"])
        for (_, section) in self.sections_objects.items():
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
                    end_vector = np.array(Math.rotate_2d_vector(yaw, [1, 0]) * a + start_vector)
                    plt.plot([start_vector[0], end_vector[0]], [start_vector[1], end_vector[1]])

    def draw_visible_pixels(self, path_to_file):
        m = np.logical_not(np.copy(self.world_map.target_obstacle_map))
        for (_, section) in self.sections_objects.items():
            for region_index in section.objects:
                region = section.objects[region_index]
                for point_index in region.objects:
                    point = region.objects[point_index]
                    for (di, d) in point.objects.items():
                        m[d[0], d[1]] = 1
        m = m.astype(np.uint8)
        m[m > 0] = 255
        cv2.imwrite(path_to_file, m)
        # cv2.imwrite("visibility_map.png", m)

    def obstacles_corners(self):
        corners = copy.copy(self.world_map.corners)
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


def create_section_from_file(map_folder_path):
    if not os.path.exists(map_folder_path):
        raise IOError("File {} not found".format(map_folder_path))
    section_map_creator = SectionMap()
    allright = True
    print(os.listdir(map_folder_path))
    for f1 in os.listdir(map_folder_path):
        if "section_file.pickle" in os.listdir(map_folder_path):
            pass
        if f1.endswith(".json") and f1.startswith("sworld"):
            f1_path = map_folder_path + "/" + f1
            (size, built, height, position, d_obstacles, orientation, map_id) = read_map_parameters(f1)
            it = 0
            while True:
                it += 1
                try:
                    section_map_creator.create_sections_regions_and_points(f1_path)
                    break
                except Exception as e:
                    print("Error occured!")
                    print(e)
                    print("Restarting algorihtm!")
                    if it > 5:
                        raise e
            sections_statisctics = section_map_creator.compute_statistics()
            stats = statistics(sections_statisctics)
            section_file_path = map_folder_path + "/" + "section_file.pickle"
            section_json_file_path = map_folder_path + "/" + "section_file.json"
            section_information = map_folder_path + "/" + "section_statics.json"
            orientation_file_path = map_folder_path + "/orientation_points.png"
            points_file_path = map_folder_path + "/points.png"
            edges_file_path = map_folder_path + "/edges.png"
            # visibility_file_path = map_folder_path + "/visibility_points.png"
            visibility_pixels_file_path = map_folder_path + "/visibility_pixels.png"
            sections_file_path = map_folder_path + "/sections.png"
            try:
                section_map_creator.pickle_sections(section_file_path)
                convert(section_file_path, section_json_file_path)
                sec = SectionUtils.unpickle_sections(section_file_path)
                if sec is None:
                    raise Exception("Problem with pickling")
            except Exception as e:
                print(e)
                allright = False
            try:
                section_map_creator.print_orientation_points(orientation_file_path, visibility_pixels_file_path)
            except Exception as e:
                print(e)
                allright = False
            try:
                section_map_creator.print_vertices(points_file_path)
            except Exception as e:
                print(e)
                allright = False
            try:
                section_map_creator.print_edges(edges_file_path, 1, -1)
            except Exception as e:
                print(e)
                allright = False
            try:
                section_map_creator.print_section(sections_file_path)
            except Exception as e:
                print(e)
                allright = False
            try:
                with open(section_information, "w") as file:
                    json.dump(stats, file, indent=2)
            except Exception as e:
                print(e)
                allright = False
            break
    return allright


def create_section_from_files_in_folder(folder_name):
    for f in os.listdir(folder_name):
        if f.startswith("sworld-10-"):
            f_path = folder_name + "/" + f
            create_section_from_file(f_path)
            break


def read_map_parameters(map_file):
    import re
    t = re.findall("sworld-([^-]+)-([^-]+)-(-1|[^-]+)-([^-]+)-(-1|[^-]+)-([^-]+)-([^-]+).json", map_file)
    return t[0]


def statistics(sections_statisctics):
    st = {}
    st["areas"] = [section["area"] for section in sections_statisctics]
    st["obstacles"] = [section["obstacles"] for section in sections_statisctics]
    st["clusters_on_size"] = [section["clusters_count"] / section["area"] for section in sections_statisctics]
    st["clusters_distances"] = [section["region_distance"] for section in sections_statisctics]
    st["points_distances"] = [np.mean(section["points_distance"]) for section in sections_statisctics]
    st["reduction_effectivity"] = [
        np.array(section["points_after_reduction"]) / np.array(section["points_before_reduction"])
        for section in sections_statisctics]
    st["average_height"] = np.mean(
        [np.mean([np.mean([np.mean(a) for a in s]) for s in section["points_heights"]]) for section in
         sections_statisctics])
    st["borders_sums"] = [section["borders_sums"] / section["perimeter"] for section in sections_statisctics]
    st["borders_counts"] = [section["borders_counts"] for section in sections_statisctics]
    st["visibility"] = sections_statisctics[0]["unseen_visible_pixels"] / sections_statisctics[0]["all_pixels"]
    return st


def draw_images(section_file_name):
    sections = SectionUtils.unpickle_sections(section_file_name)
    SectionUtils.show_sections(sections)


def convert(pickle_file, json_file):
    SectionUtils.convert_pickle_file_into_json(pickle_file, json_file)
    print("==========")
    sections = SectionUtils.unjson_sections(json_file)
    print(sections)


def show_voronoi(json_file):
    matplotlib.use('TkAgg')
    sections = SectionUtils.unjson_sections(json_file)
    points, ids = SectionUtils.get_all_points(sections)
    point_d = {}
    for i in range(len(points)):
        p = points[i]
        if p[0] not in point_d:
            point_d[p[0]] = {}
        if p[1] not in point_d[p[0]]:
            point_d[p[0]][p[1]] = []
        point_d[p[0]][p[1]].append(i)
    for pp1 in point_d:
        for pp2 in point_d[pp1]:
            print(point_d[pp1][pp2])
    vor = Voronoi(np.array(points))
    voronoi_plot_2d(vor)
    showed = {}
    for i in range(len(ids)):
        object_id = ids[i]
        point = points[i]
        if point[0] not in showed:
            showed[point[0]] = {}
        if point[1] not in showed[point[0]]:
            showed[point[0]][point[1]] = True
            plt.text(point[0], point[1], str(object_id), fontsize=12)
    reshaped = np.array(vor.ridge_points).reshape(-1)
    print(np.unique(reshaped))
    plt.show()


if __name__ == '__main__':
    # create_section_from_files_in_folder("/home/patik/Diplomka/dp_ws/src/pesat_resources/worlds")
    # print(SectionUtils.unpickle_sections("/home/patik/Diplomka/dp_ws/src/pesat_resources/worlds/sworld-100-0_2995--1-f--1-n-22_46_25/section_file.pickle"))
    # pickle_file = "/home/patik/Diplomka/dp_ws/src/pesat_resources/worlds/sworld-100-0_2999--1-f--1-n-10_7_32/section_file.pickle"
    json_file = "/home/patik/Diplomka/dp_ws/src/pesat_resources/worlds/sworld-100-0_2999--1-f--1-n-10_7_32/section_file.json"
    # convert(pickle_file, json_file)
    # SectionUtils.show_sections(SectionUtils.unjson_sections(json_file), False)
    SectionUtils.show_sections(SectionUtils.unjson_sections(json_file)["objects"], True)
    # show_voronoi(json_file)
    #create_section_from_file(
    #    "/home/patik/Diplomka/dp_ws/src/pesat_resources/worlds/sworld-100-0_2999--1-f--1-n-10_7_32")
    pass
