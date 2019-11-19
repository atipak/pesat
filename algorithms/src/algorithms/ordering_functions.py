from abc import abstractmethod
import numpy as np
import helper_pkg.utils as utils
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from section_map_creator import SectionMapObject
import rospy
from collections import defaultdict


class ExtendedOrderingFunction(object):
    def __init__(self):
        super(ExtendedOrderingFunction, self).__init__()

    @abstractmethod
    def extended_select(self, selected_objects, scored_objects, objects, properties, start_position_index,
                        end_position_index=None):
        pass


class NeighboursPath(ExtendedOrderingFunction):

    def extended_select(self, selected_objects, scored_objects, objects, properties, start_position_index,
                        end_position_index=None):
        # calculate distances for neighbours
        dist = np.zeros((properties.shape[0], properties.shape[1]))
        for object_index in range(len(objects)):
            neighbours = np.array([key for key in objects[object_index].neighbors])
            dist[object_index, neighbours] = properties[object_index, neighbours]
        graph = csr_matrix(dist)
        dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True)
        # where we start and the way we will go
        lasts = [start_position_index]
        way = [start_position_index]
        # if end position is not None, then we have to end up in this position
        back_way = []
        if end_position_index is not None:
            lasts.append(end_position_index)
            back_way.append(end_position_index)
        rest = selected_objects[:]
        # find better part for complete way from start to end
        # we calculate for both sides of way and we take the better one
        while len(rest) > 0:
            maximum = []
            maximum_last = lasts[0]
            for last in lasts:
                for x in rest:
                    path = utils.Graph.get_path(predecessors, last, x)[::-1]
                    intersection = np.intersect1d(path, rest)
                    if len(maximum) < len(intersection):
                        maximum = path
                        maximum_last = last
            if maximum_last == lasts[0]:
                way.extend(maximum[1:])
                lasts[0] = maximum[-1]
            else:
                back_way.extend(maximum[1:])
                lasts[1] = maximum[-1]
            rest = rest[np.bitwise_not(np.in1d(rest, maximum))]
        way.extend(np.flip(back_way))
        return way


class DirectPath(ExtendedOrderingFunction):

    def extended_select(self, selected_objects, scored_objects, objects, properties, start_position_index,
                        end_position_index=None):
        # where we start and the way we will go
        lasts = [start_position_index]
        way = [start_position_index]
        # if end position is not None, then we have to end up in this position
        back_way = []
        if end_position_index is not None:
            lasts.append(end_position_index)
            back_way.append(end_position_index)
        rest = selected_objects[:]
        while len(rest) > 0:
            minimum = None
            minimum_last = lasts[0]
            minimum_index = 0
            for last in lasts:
                distances = properties[last, np.in1d(properties[last], rest)]
                m = np.min(distances)
                if minimum is None or minimum > m:
                    minimum = m
                    minimum_index = np.argmin(distances)
            mask = np.ones((rest))
            mask[minimum_index] = False
            if minimum_last == lasts[0]:
                way.extend(rest[minimum_index])
                lasts[0] = rest[minimum_index]
            else:
                back_way.extend(rest[minimum_index])
                lasts[1] = rest[minimum_index]
            rest = rest[mask]
        way.extend(np.flip(back_way))
        return way


class Graph:

    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices

        # default dictionary to store graph
        self.graph = defaultdict(list)

        # function to add an edge to graph

    def addEdge(self, u, v):
        self.graph[u].append(v)

    '''A recursive function to print all paths from 'u' to 'd'. 
    visited[] keeps track of vertices in current path. 
    path[] stores actual vertices and path_index is current 
    index in path[]'''

    def printAllPathsUtil(self, u, d, visited, path, paths, max_iter):

        if len(paths) >= max_iter:
            return

            # Mark the current node as visited and store in path
        visited[u] = True
        path.append(u)

        # If current vertex is same as destination, then print
        # current path[]
        if u in d:
            paths[u].append(np.copy(path))
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.graph[u]:
                if not visited[i]:
                    self.printAllPathsUtil(i, d, visited, path, paths, max_iter)

                    # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u] = False

    # Prints all paths from 's' to 'd'
    def printAllPaths(self, s, d, max_iter):

        # Mark all the vertices as not visited
        visited = np.zeros(self.V)

        # Create an array to store paths
        path = []
        paths = defaultdict(list)

        # Call the recursive helper function to print all paths
        self.printAllPathsUtil(s, d, visited, path, paths, max_iter)
        return paths


class PathPlaning(object):
    def __init__(self):
        super(PathPlaning, self).__init__()
        drone_control = rospy.get_param("drone_configuration")["control"]
        self.max_speed = drone_control["video_max_horizontal_speed"]
        self.score_coef = -4
        self.entropy_coef = -3
        self.time_coef = 2
        self.space_coef = -1
        self.total_score_coef = 2
        self.color_coef = -1
        self.max_iterations = 100

    def extended_select(self, selected_objects, scored_objects, objects, properties, start_position_index,
                        end_position_index=None):
        if end_position_index is None or not isinstance(end_position_index, list) or len(end_position_index) == 0:
            return []
        distances = np.zeros(len(objects))
        for k in range(len(objects)):
            o = objects[k]
            for i in range(len(o.objects)):
                o1 = o.objects[i]
                if not isinstance(o1, SectionMapObject):
                    break
                for j in range(i, len(o.objects)):
                    o2 = o.objects[j]
                    d = utils.Math.euclidian_distance(utils.Math.Point(*o1.centroid), utils.Math.Point(*o2.centroid))
                    distances[k] += d
        properties += distances
        properties = properties[properties != 0]
        properties /= self.max_speed
        properties /= np.max(properties)
        visible_space = np.zeros(len(objects))
        for k in range(len(objects)):
            o = objects[k]
            visible_space[k] = self.divide_and_sum(o)
        visible_space /= np.max(visible_space)
        entropies = np.array([o.entropy for o in objects])
        vertices = np.zeros(len(objects))
        vertices += self.score_coef * scored_objects + self.entropy_coef * entropies + self.space_coef * visible_space
        properties *= 2
        properties += vertices
        g = self.create_graph(properties)
        paths = g.printAllPaths(start_position_index, end_position_index, self.max_iterations)
        return paths[self.choose_best(paths, objects, properties)]

    def create_graph(self, properties):
        g = Graph(properties.shape[0])
        for i in range(properties.shape[0]):
            for j in range(properties.shape[1]):
                if properties[i, j] != 0:
                    g.addEdge(i, j)
        return g

    def divide_and_sum(self, object):
        s = 0
        for o in object.objects:
            if isinstance(o, SectionMapObject):
                s += self.divide_and_sum(o)
            else:
                s += 1
        return s

    def get_all_paths(self, g, start, target):
        return []

    def evaluate_path(self, path, objects, properties):
        value = 0
        for i in range(len(path) - 1):
            value += properties[i, i + 1]
        return [value, self.coloring(path, objects)]

    def choose_best(self, paths, objects, properties):
        values = np.empty((paths, 2))
        for i in range(len(paths)):
            path = paths[i]
            values[i] = self.evaluate_path(path, objects, properties)
        values[:, 1] /= np.max(values[:, 1])
        evaluation = self.total_score_coef * values[:, 0] - self.color_coef * values[:, 1]
        return np.argmax(evaluation)

    def coloring(self, path, objects):
        colors = np.zeros(len(objects))
        current_color = 1
        for p in path:
            colors[p] = current_color
        while np.count_nonzero(colors) < len(objects):
            unprocessed_objects = []
            current_color += 1
            for i in range(len(colors)):
                if colors[i] == 0:
                    unprocessed_objects.append(i)
                    break
            while len(unprocessed_objects) > 0:
                o_index = unprocessed_objects.pop()
                colors[o_index] = current_color
                for os in objects[o_index].neighbors:
                    if colors[os] == 0:
                        unprocessed_objects.append(os)
        return colors


class DirectPathPlanning(ExtendedOrderingFunction):
    def __init__(self):
        super(DirectPathPlanning, self).__init__()
        self.path_planing = PathPlaning()

    def extended_select(self, selected_objects, scored_objects, objects, properties, start_position_index,
                        end_position_index=None):
        properties_copy = np.copy(properties)
        return self.path_planing.extended_select(selected_objects, scored_objects, objects, properties_copy,
                                                 start_position_index, end_position_index)


class NeighboursPathPlanning(ExtendedOrderingFunction):
    def __init__(self):
        super(NeighboursPathPlanning, self).__init__()
        self.path_planing = PathPlaning()

    def extended_select(self, selected_objects, scored_objects, objects, properties, start_position_index,
                        end_position_index=None):
        # calculate distances for neighbours
        dist = np.zeros((properties.shape[0], properties.shape[1]))
        for object_index in range(len(objects)):
            neighbours = np.array([key for key in objects[object_index].neighbors])
            dist[object_index, neighbours] = properties[object_index, neighbours]
        return self.path_planing.extended_select(selected_objects, scored_objects, objects, dist,
                                                 start_position_index, end_position_index)
