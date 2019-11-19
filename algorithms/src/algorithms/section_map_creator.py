import rospy
import os
import copy
from geometry_msgs.msg import Pose
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.cluster.hierarchy import fclusterdata
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import cv2
from shapely.geometry import Polygon, LineString, MultiPolygon, Point, MultiPoint
import numpy as np
import helper_pkg.utils as utils
import time
import triangle
import matplotlib.pyplot as plt
from shapely.ops import triangulate, cascaded_union
from shapely.strtree import STRtree
from scipy.optimize import differential_evolution


class SectionMap(object):
    def __init__(self):
        super(SectionMap, self).__init__()
        _camera_configuration = rospy.get_param("drone_configuration")["camera"]
        self.hfov = _camera_configuration["hfov"]
        self.image_height = _camera_configuration["image_height"]
        self.image_width = _camera_configuration["image_width"]
        self.focal_length = self.image_width / (2.0 * np.tan(self.hfov / 2.0))
        self.vfov = 2 * np.arctan2(self.image_height / 2, self.focal_length)
        self.camera_range = _camera_configuration["camera_range"]
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
        begin = time.time()
        world_map = utils.Map.map_from_file(map_file, 2)
        print("World map was loaded", "during time", time.time() - begin)
        begin = time.time()
        sections = self.create_sections(world_map, environment_configuration)
        print("Section were created:", len(sections), "during time", time.time() - begin)
        regions = self.create_regions(world_map, sections)
        # regions = [section1: [cluster1: [point1: [x, y, z, vertical orientation of camera, yaw],
        # point2, ...], cluster2, ...], section2, ...]
        sections_objects = []
        section_index = 0
        point_coordinates = np.zeros((world_map.width, world_map.height))
        coordinates_of_points = {}
        for section in sections:
            section_object = SectionMapObject()
            section_object.type = SectionMapObjectTypes.Section
            section_object.object_id = section_index
            polygon = Polygon(section)
            section_object.centroid = polygon.centroid
            sections_objects.append(section_object)
            region_index = 0
            sections_regions = regions[section_index]
            section_map, section_map_without_obstacles = self.section_map(section, world_map)
            for region in sections_regions:
                region_object = SectionMapObject()
                region_object.type = SectionMapObjectTypes.Region
                region_object.object_id = region_index
                points = []
                for r in region:
                    points.append(r[:2])
                polygon = MultiPoint(points).convex_hull
                region_object.centroid = polygon.centroid
                section_object.objects.append(region_object)
                point_index = 0
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
                    for coor in coordinates:
                        if coor[0] not in coordinates_of_points:
                            coordinates_of_points[coor[0]] = {}
                        if coor[1] not in coordinates_of_points[coor[0]]:
                            coordinates_of_points[coor[0]][coor[1]] = []
                        coordinates_of_points[coor[0]][coor[1]].append([section_index, region_index, point_index])
                        if section_map[coor[1], coor[0]] == 1 and point_coordinates[coor[0], coor[1]] != 1:
                            point_object.objects.append([coor[1], coor[0]])
                            point_coordinates[coor[0], coor[1]] = 1
                    point_object.centroid = point[:2]
                    point_object.neighbors = [[i, i] for i in range(len(region)) if point_index != i]
                    point_object.data = point
                    region_object.objects.append(point_object)
                    point_index += 1
                region_index += 1
            section_index += 1
        for i in range(world_map.map.shape[0]):
            for j in range(world_map.map.shape[1]):
                if i in coordinates_of_points and j in coordinates_of_points[i]:
                    for cell in coordinates_of_points[i][j]:
                        s = cell[0]
                        r = cell[1]
                        p = cell[2]
                        for i1 in range(-1, 1):
                            for j1 in range(-1, 1):
                                if i1 + i in coordinates_of_points and j1 + j in coordinates_of_points[i1 + i]:
                                    for cell1 in coordinates_of_points[i1 + i][j1 + j]:
                                        s1 = cell1[0]
                                        r1 = cell1[1]
                                        p1 = cell[2]
                                        if s1 != s:
                                            if s1 not in sections_objects[s].neighbors:
                                                sections_objects[s].neighbors[s1] = []
                                            if [r, r1, p, p1] not in sections_objects[s].neighbors[s1]:
                                                sections_objects[s].neighbors[s1].append([r, r1, p, p1])
                                        if r1 != r or (s1 != s and r1 == r):
                                            if r1 not in sections_objects[s].objects[r].neighbors:
                                                sections_objects[s].objects[r].neighbors[r1] = []
                                            if [p, p1] not in sections_objects[s].objects[r].neighbors[r1]:
                                                sections_objects[s].objects[r].neighbors[r1].append([p, p1])
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
        print("Vertices for graph created:", len(verticies), "during time", time.time() - begin)
        vertices_map = self.print_vertices(verticies, world_map)
        begin = time.time()
        edges = self.create_edges(verticies, delimiters, world_map, environment_configuration)
        print("Edges for graph created:", np.count_nonzero(edges), "during time", time.time() - begin)
        edges_map = self.print_edges(verticies, edges, vertices_map, world_map)
        # first section is map
        verticies_indices = range(len(verticies))
        sections_for_reduction = [verticies_indices[delimiters[len(delimiters) - 2] + 1:]]
        # print(sections_for_reduction)
        sections = []
        maximum = np.max(edges)
        print("Dijsktra was successful")
        print("Launching algorithm for making sections")
        while len(sections_for_reduction) > 0:
            section = np.array(sections_for_reduction.pop())
            rolled_section = np.roll(section, 1)
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
                    correct_path = True
                    break
            if correct_path:
                path_lower_to_upper = path_upper_to_lower[::-1]
                print("vertices", verticies[section[lower_index]], verticies[section[upper_index]])
                print("paths", path_upper_to_lower, path_lower_to_upper)
                section_one = [section[i] for i in range(lower_index + 1, upper_index)]
                section_two = [section[i] for i in range(upper_index + 1, len(section))]
                section_two_part_one = [section[i] for i in range(0, lower_index)]
                section_two.extend(section_two_part_one)
                section_one.extend(path_upper_to_lower)
                section_two.extend(path_lower_to_upper)
                if self.get_section_area(section_one, verticies) > self._max_section_area:
                    sections_for_reduction.append(section_one)
                else:
                    sections.append(section_one)
                if self.get_section_area(section_two, verticies) > self._max_section_area:
                    sections_for_reduction.append(section_two)
                else:
                    sections.append(section_two)
            else:
                sections.append(section)
        vertex_sections = []
        for s in sections:
            vertex_section = []
            for v in s:
                vertex_section.append(verticies[v])
            vertex_sections.append(np.array(vertex_section))
        vertex_sections = np.array(vertex_sections)
        self.print_section(vertex_sections, edges_map, world_map)
        return vertex_sections

    def print_vertices(self, vertices, world_map):
        x, y = world_map.get_index_on_map(vertices[:, 0], vertices[:, 1])
        x = np.interp(x, (0, world_map.width), (0, 5000))
        y = np.interp(y, (0, world_map.height), (0, 5000))
        map_points = np.array(zip(x, y), dtype=np.int32)
        vertices_map = np.zeros((5000, 5000, 4), np.uint8)
        vertices_map.fill(255)
        for map_point in map_points:
            cv2.circle(vertices_map, tuple(map_point), 20, (0, 0, 0, 255), -1)
        cv2.imwrite("vertex_map.png", vertices_map)
        return vertices_map

    def print_edges(self, vertices, edges, verticies_map, world_map):
        indices = np.argwhere(edges > 0)
        x, y = world_map.get_index_on_map(vertices[:, 0], vertices[:, 1])
        x = np.interp(x, (0, world_map.width), (0, 5000))
        y = np.interp(y, (0, world_map.height), (0, 5000))
        map_points = np.array(zip(x, y), dtype=np.int32)
        for index in indices:
            start = map_points[index[1]]
            end = map_points[index[0]]
            cv2.line(verticies_map, (start[0], start[1]), (end[0], end[1]), (0, 0, 0, 255), 4)
        cv2.imwrite("edges_map.png", verticies_map)
        return verticies_map

    def print_section(self, sections, edges_map, world_map):
        sections_map = np.zeros((5000, 5000, 4), np.uint8)
        for section in sections:
            x, y = world_map.get_index_on_map(section[:, 0], section[:, 1])
            x = np.interp(x, (0, world_map.width), (0, 5000))
            y = np.interp(y, (0, world_map.height), (0, 5000))
            map_points = np.array(zip(x, y), dtype=np.int32)
            # print(map_points)
            cv2.fillConvexPoly(sections_map, map_points, [np.random.randint(0, 255, 1)[0],
                                                          np.random.randint(0, 255, 1)[0],
                                                          np.random.randint(0, 255, 1)[0],
                                                          120])
        edges_map[edges_map == 255] = sections_map[edges_map == 255]
        cv2.imwrite("s_map.png", edges_map)

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
        maximum = np.max(edges)
        edges[np.logical_and(edges > 0, edges != 1.0)] = 1.5 * edges[np.logical_and(edges > 0, edges != 1.0)] + maximum
        return edges

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

    def get_shapely_obstacles(self):
        obstacles = []
        points = []
        rectangles_index = 0
        for index in range(len(self.vertices)):
            if rectangles_index == len(self.delimiters) - 2:
                break
            if index == self.delimiters[rectangles_index]:
                if rectangles_index - 1 < 0:
                    points.append(self.vertices[0])
                else:
                    points.append(self.vertices[self.delimiters[rectangles_index - 1] + 1])
                obstacles.append(Polygon(points))
                points = []
                rectangles_index += 1
            else:
                points.append(self.vertices[index])
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
        inner_obstacles_count = 0
        for obstacle_index in range(len(obstacles)):
            print("==============")
            obstacle = obstacles[obstacle_index]
            if section_polygon.contains(obstacle):
                # obstacle = self.simplify_polygon(obstacle)
                # obstacle = obstacle.simplify(0.0, True)
                section_polygon = section_polygon.difference(obstacle)
                inner_obstacles_count += 1
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
        return correct_triangles, inner_obstacles_count

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
                for i in ii[1:]:
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

    def remove_small_triangles(self, correct_triangles, min_distance=0.5):
        new_triangles = []
        for tri in correct_triangles:
            dd, ii = self.vertices_tree.query(np.array(tri.centroid), 2)
            if dd[1] >= min_distance:
                new_triangles.append(tri)
        return new_triangles

    def reduce_triangles(self, maximal_iterations, min_distance, correct_triangles):
        correct_triangles = self.join_triangles_to_triangles(maximal_iterations, correct_triangles)
        correct_triangles = self.join_triangles_to_rectangles(correct_triangles)
        correct_triangles = self.remove_small_triangles(correct_triangles, min_distance)
        return correct_triangles

    def callback_evolution(self, xk, convergence=None):
        print("xk", np.reshape(xk, (-1, 4)))
        self.fitness_function(xk, self.free_pixels, self.world_map, self.correct_triangles, self.section_map_image,
                              True)
        print("convergence", convergence)
        if self.iterations > 5:
            return True
        self.iterations += 1
        return True

    def create_bounds(self, bounds, count):
        all_bounds = []
        for c in range(count):
            for b in bounds:
                all_bounds.append(b)
        return all_bounds

    def create_regions(self, world_map, sections):
        section_points = []
        obstacles = self.get_shapely_obstacles()
        for section in sections:
            correct_triangles, inner_obstacles = self.triangulate_section(section, obstacles)
            min_area = 0.5
            maximal_iterations = 50
            correct_triangles = self.reduce_triangles(maximal_iterations, min_area, correct_triangles)
            correct_triangles = np.array(correct_triangles)
            if False:
                plt.figure()
                for tri in correct_triangles:
                    x, y = tri.exterior.xy
                    # if tri.exterior.distance(tri.centroid) > median:
                    c = np.array(tri.centroid)
                    plt.scatter(c[0], c[1])
                    plt.plot(x, y)
                for o in obstacles:
                    x, y = o.exterior.xy
                    plt.fill(x, y, color=(0.1, 0.2, 0.5, 0.3))
                plt.show()
            section_map, section_map_without_obstacles = self.section_map(section, world_map)
            free_pixels = np.count_nonzero(section_map)
            bounds = [(0, len(correct_triangles) - 1), (3, 15), (0, -np.pi / 2), (0, 2 * np.pi)]
            points_count = (inner_obstacles + 1) * 2
            all_bounds = self.create_bounds(bounds, points_count)
            self.iterations = 0
            self.free_pixels, self.world_map, self.correct_triangles = free_pixels, world_map, correct_triangles
            self.section_map_image = section_map
            result = differential_evolution(self.fitness_function, all_bounds,
                                            args=(free_pixels, world_map, correct_triangles, section_map, False),
                                            disp=True, callback=self.callback_evolution, polish=True, mutation=0.2,
                                            recombination=0.05)
            print(result.success)
            print(result.x)
            points = np.reshape(np.array(result.x), (-1, 4))
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

            section_points.append(clusters)
        return section_points

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
            coordinates = world_map.rectangle_ray_tracing_3d(position, self.vfov, self.hfov, self.camera_range)
            seen_pixels[coordinates[:, 1], coordinates[:, 0]] = 1
        seen_pixels_from_section = np.logical_and(seen_pixels, section_map)
        unseen_pixels = free_pixels - np.count_nonzero(seen_pixels_from_section)
        return unseen_pixels

    def fitness_function(self, original_points, *args):
        free_pixels, world_map, correct_triangles, section_map, print_info = args
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
        return fitness

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


class SectionMapObjectTypes:
    Section = 0
    Region = 1
    Point = 2


class SectionMapObject(object):
    def __init__(self):
        super(SectionMapObject, self).__init__()
        self.object_id = 0
        self.score = 0
        self.entropy = 0
        self.neighbors = {}
        self.centroid = [0, 0]
        self.objects = []
        self.type = SectionMapObjectTypes.Section
        self.data = None
