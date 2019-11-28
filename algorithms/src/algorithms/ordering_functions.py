from abc import abstractmethod
import numpy as np
import helper_pkg.utils as utils
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from section_map_creator import SectionMapObject
import rospy
from collections import defaultdict
import graph_tool.all as gt


class ExtendedOrderingFunction(object):
    def __init__(self):
        super(ExtendedOrderingFunction, self).__init__()

    @abstractmethod
    def extended_select(self, selected_objects, scored_objects, objects, properties, start_position_index,
                        end_position_index=None):
        pass

    def remove_start_end_from_selected(self, selected, start, end):
        pass


class NeighboursPath(ExtendedOrderingFunction):

    def extended_select(self, selected_objects, scored_objects, objects, properties, start_position_index,
                        end_position_index=[]):
        if len(end_position_index) > 1:
            return []
        # calculate distances for neighbours
        distances = properties["distances"]
        dist = np.zeros((distances.shape[0], distances.shape[1]))
        ids = np.sort(np.array([i for (i, object) in objects.items()]))
        i = 0
        for object_index in ids:
            neighbours = np.array([key for key in objects[object_index].neighbors if key in ids])
            neighbours = np.in1d(ids, neighbours).nonzero()[0]
            if len(neighbours) > 0:
                dist[i, neighbours] = distances[i, neighbours]
            i += 1
        graph = csr_matrix(dist)
        dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True)
        # where we start and the way we will go
        lasts = [start_position_index]
        way = [start_position_index]
        # if end position is not None, then we have to end up in this position
        back_way = []
        if len(end_position_index) > 0 and end_position_index[0] != start_position_index:
            lasts.append(end_position_index[0])
            back_way.extend(end_position_index)
        rest = selected_objects[:]
        if start_position_index in rest:
            i = list(rest).index(start_position_index)
            ii = np.r_[0:i, i + 1: len(rest)]
            rest = rest[ii]
        if end_position_index in rest:
            i = list(rest).index(end_position_index)
            ii = np.r_[0:i, i + 1:len(rest)]
            rest = rest[ii]
        # find better part for complete way from start to end
        # we calculate for both sides of way and we take the better one
        rest = np.array(rest)
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
                        end_position_index=[]):
        object_distances = properties["distances"]
        # where we start and the way we will go
        lasts = [start_position_index]
        way = [start_position_index]
        # if end position is not None, then we have to end up in this position
        back_way = []
        if len(end_position_index) > 0 and end_position_index[0] != start_position_index:
            lasts.append(end_position_index[0])
            back_way.extend(end_position_index)
        rest = np.array(selected_objects[:])
        if start_position_index in rest:
            i = list(rest).index(start_position_index)
            ii = np.r_[0:i, i + 1: len(rest)]
            rest = rest[ii]
        if end_position_index in rest:
            i = list(rest).index(end_position_index)
            ii = np.r_[0:i, i + 1:len(rest)]
            rest = rest[ii]
        while len(rest) > 0:
            minimum = None
            minimum_last = lasts[0]
            minimum_index = 0
            for last in lasts:
                distances = object_distances[last, np.in1d(range(len(object_distances[last])), rest)]
                m = np.min(distances)
                if minimum is None or minimum > m:
                    minimum = m
                    minimum_index = np.argmin(distances)
            mask = np.ones((len(rest)), dtype=np.bool)
            mask[minimum_index] = False
            if minimum_last == lasts[0]:
                way.append(rest[minimum_index])
                lasts[0] = rest[minimum_index]
            else:
                back_way.append(rest[minimum_index])
                lasts[1] = rest[minimum_index]
            rest = rest[mask]
        way.extend(np.flip(back_way))
        return way


class AStarSearch(ExtendedOrderingFunction):
    def __init__(self):
        super(AStarSearch, self).__init__()
        drone_control = rospy.get_param("drone_configuration")["control"]
        self.max_speed = drone_control["video_max_horizontal_speed"]
        self.max_battery_life_in_flight = drone_control["max_battery_life_in_flight"]
        self.score_coef = -4
        self.entropy_coef = -3
        self.time_coef = 2
        self.space_coef = -1
        self.total_score_coef = 2

    class Visitor(gt.AStarVisitor):

        def __init__(self, touched_v, touched_e, target, weight, v_inner_distances, e_distances, v_time, v_score,
                     v_maximal_time, v_entropy, v_maximal_entropy, v_required, v_visibilities, v_required_path,
                     requested_count, battery_life, predecessors, g):
            super(AStarSearch.Visitor, self).__init__()
            self.score_coef = 3
            self.entropy_coef = 2
            self.space_coef = 1
            self.unrequired_compensation = len(touched_v.get_array())
            self.touched_v = touched_v
            self.touched_e = touched_e
            self.target = target
            self.weight = weight
            self.v_inner_distances = v_inner_distances
            self.e_distances = e_distances
            self.v_time = v_time
            self.v_score = v_score
            self.v_maximal_time = v_maximal_time
            self.v_entropy = v_entropy
            self.v_maximal_entropy = v_maximal_entropy
            self.v_required = v_required
            self.v_visibilities = v_visibilities
            self.v_required_path = v_required_path
            self.requested_count = requested_count
            self.target_vertex = None
            self.battery_life = battery_life
            self.predecessors = predecessors
            self.g = g
            self.visibility = 0
            for v in self.g.vertices():
                self.visibility += self.v_visibilities[v]

        def discover_vertex(self, u):
            self.touched_v[u] = True

        def examine_edge(self, e):
            target_v = e.target()
            source_v = e.source()
            if not self.touched_v[target_v]:
                self.set_new_relaxed_weights(e)
                self.predecessors[target_v] = int(source_v)
            self.touched_e[e] = True

        def examine_vertex(self, u):
            if int(u) in self.target:
                if self.v_required_path[u] == self.requested_count:
                    self.target_vertex = u
                    raise gt.StopSearch()

        def edge_relaxed(self, e):
            target_v = e.target()
            source_v = e.source()
            self.set_new_relaxed_weights(e)
            self.predecessors[target_v] = int(source_v)

        def time_score_function(self, time, maximal_time, value, min_value):
            return value - np.clip(time / float(maximal_time), 0, 1) * (value - min_value)

        def time_entropy_function(self, time, maximal_time, value, max_value):
            return value + np.clip(time / float(maximal_time), 0, 1) * (max_value - value)

        def set_new_relaxed_weights(self, relaxed_edge):
            self.set_new_time(relaxed_edge)
            target_v = relaxed_edge.target()
            v = self.predecessors[relaxed_edge.source()]
            path = [int(relaxed_edge.source())]
            while v != -1:
                path.append(v)
                v = self.predecessors[self.g.vertex(v)]
            entropy, score, visibility = self.compute_from_path(path, relaxed_edge)
            for e in target_v.out_edges():
                self.weight[e] = self.calculate_weight(e, entropy, score, visibility)

        def compute_from_path(self, path, e):
            source_v = int(e.source())
            entropy = 0
            score = 0
            visibility = 0
            time_in_target = self.v_inner_distances[source_v] + self.e_distances[e] + self.v_time[source_v]
            for v in self.g.vertices():
                if int(v) not in path:
                    s = self.time_score_function(time_in_target, self.v_maximal_time[v], self.v_score[v],
                                                 0.33 * self.v_score[v])
                    score += (self.v_score[v] - s)
                    entropy += self.time_entropy_function(time_in_target, self.v_maximal_time[v], self.v_entropy[v],
                                                          self.v_maximal_entropy[v])
                    visibility += self.v_visibilities[v]
            return entropy, score, visibility

        def calculate_weight(self, e, entropy, score, visibility):
            source_v = e.source()
            target_v = e.source()
            weight = 0
            # time
            time_in_target = self.v_inner_distances[source_v] + self.e_distances[e] + self.v_time[source_v]
            time_score = self.time_score_function(time_in_target, self.v_maximal_time[target_v], self.v_score[target_v],
                                                  0.33 * self.v_score[target_v])
            time_entropy = self.time_entropy_function(time_in_target, self.v_maximal_time[target_v],
                                                      self.v_entropy[source_v], self.v_maximal_entropy[source_v])
            target_entropy = entropy - time_entropy
            target_score = score - (self.v_score[target_v] - time_score)
            target_visibility = visibility - self.v_visibilities[target_v]
            # set required path
            weight += self.unrequired_compensation if not self.v_required[target_v] else 0
            weight += self.score_coef * target_score + self.entropy_coef * target_entropy + self.space_coef * target_visibility
            weight += self.unrequired_compensation * self.unrequired_compensation if time_in_target > self.battery_life else 0
            weight = np.clip(weight, 0, None)
            return weight

        def set_new_time(self, e):
            source_v = e.source()
            target_v = e.source()
            if not self.v_required[target_v]:
                self.v_required_path[target_v] = self.v_required_path[source_v]
            else:
                self.v_required_path[target_v] = 1 + self.v_required_path[source_v]
            self.v_time[target_v] = self.v_inner_distances[source_v] + self.e_distances[e] + self.v_time[source_v]

    def h(self, v, dist):
        return dist[int(v)]

    def extended_select(self, selected_objects, scored_objects, objects, properties, start_position_index,
                        end_position_index=[]):
        distances = np.copy(properties["distances"])
        inner_distances = np.copy(properties["inner_distances"]).astype(np.float32)
        battery_life = properties["battery_life"] * self.max_battery_life_in_flight
        ids = np.sort(np.array([i for (i, object) in objects.items()]))
        visibility = np.array([objects[i].visibility for i in ids])
        distances /= self.max_speed
        distances /= np.max(distances)
        inner_distances /= self.max_speed
        inner_distances /= np.max(inner_distances)
        entropies = np.array([objects[i].entropy for i in ids])
        visibility /= np.max(visibility)
        g = gt.Graph()
        _ = g.add_vertex(len(objects))

        # vertex, edge properties definition
        ########################
        # amount of visible places in object
        v_visibilities = g.new_vertex_property("double")
        # estimation of time spent in object
        v_inner_distances = g.new_vertex_property("double")
        # distances between vertices
        e_distances = g.new_edge_property("double")
        # entropy
        v_entropy = g.new_vertex_property("double")
        # score
        v_score = g.new_vertex_property("double")
        # estimated maximal time which is spent by target in object
        v_maximal_time = g.new_vertex_property("double")
        # estimated maximal entropy which is spent by target in object
        v_maximal_entropy = g.new_vertex_property("double")
        # time
        v_time = g.new_vertex_property("double")
        # required
        v_required = g.new_vertex_property("bool")
        # required vertices on path
        v_required_path = g.new_vertex_property("int")
        # predecessors
        predecessors = g.new_vertex_property("int")
        touch_v = g.new_vertex_property("bool")
        touch_e = g.new_edge_property("bool")
        weight = g.new_edge_property("double")
        ####################

        # vertices properties
        ###################
        start_index = np.in1d(ids, [start_position_index]).nonzero()[0][0]
        for v_index in range(len(objects)):
            v = g.vertex(v_index)
            v_visibilities[v] = visibility[v_index]
            v_inner_distances[v] = inner_distances[v_index]
            v_entropy[v] = entropies[v_index]
            v_score[v] = scored_objects[v_index]
            v_maximal_time[v] = objects[ids[v_index]].maximal_time
            v_maximal_entropy[v] = objects[ids[v_index]].maximal_entropy
            v_time[v] = 0
            v_required[v] = True if v_index in selected_objects else False
            v_required_path[v] = 1 if v_index == start_index and v_index in selected_objects else 0
        # there is no predecessor for start vertex
        predecessors[g.vertex(start_index)] = -1
        shadow_start_index = start_index
        if start_position_index in end_position_index:
            v = g.add_vertex()
            shadow_start_index = int(v)
            v_visibilities[v] = visibility[start_index]
            v_inner_distances[v] = inner_distances[start_index]
            v_entropy[v] = entropies[start_index]
            v_score[v] = scored_objects[start_index]
            v_maximal_time[v] = objects[ids[start_index]].maximal_time
            v_maximal_entropy[v] = objects[ids[start_index]].maximal_entropy
            v_time[v] = 0
            v_required[v] = False
            v_required_path[v] = 0
        ###################

        # edges
        ###################
        i = 0
        for object_index in ids:
            o = objects[object_index]
            for key in o.neighbors:
                ind = np.in1d(ids, [key]).nonzero()[0]
                if len(ind) > 0:
                    if ind[0] == start_index:
                        add_vertex =  shadow_start_index
                    else:
                        add_vertex = ind[0]
                    #print(str(i) + "->" + str(add_vertex))
                    g.add_edge(i,add_vertex)
            i += 1

        # edges properties
        ###################
        for e in g.edges():
            if int(e.target()) == shadow_start_index:
                e_distances[e] = distances[int(e.source()), start_index]
            else:
                e_distances[e] = distances[int(e.source()), int(e.target())]
        # Euclidean distances
        for e in g.edges():
            weight[e] = e_distances[e]
        ###################

        # end position vertices mapping
        ###################
        verticies_position_indices = []
        if len(end_position_index) > 0:
            for index in end_position_index:
                ind = np.in1d(ids, [index]).nonzero()[0]
                if len(ind) > 0:
                    if ind[0] == start_index:
                        verticies_position_indices.append(shadow_start_index)
                    else:
                        verticies_position_indices.append(ind[0])
        ###################

        # heuristic function
        ###################
        extra_vertex = 0
        if shadow_start_index != start_index:
            extra_vertex = 1
        h_dist = np.zeros(len(distances) + extra_vertex)
        for v1 in range(len(distances)):
            min_v1 = float("inf")
            for v in verticies_position_indices:
                if v == shadow_start_index:
                    d = distances[v1, start_index]
                    if d < min_v1:
                        min_v1 = d
                else:
                    d = distances[v1, v]
                    if 0 < d < min_v1:
                        min_v1 = d
            h_dist[v1] = min_v1
        ###################

        visitor = AStarSearch.Visitor(touch_v, touch_e, verticies_position_indices, weight, v_inner_distances,
                                      e_distances, v_time, v_score, v_maximal_time, v_entropy, v_maximal_entropy,
                                      v_required, v_visibilities, v_required_path, len(selected_objects), battery_life,
                                      predecessors, g)
        dist, pred = gt.astar_search(g, g.vertex(start_index), weight, visitor, heuristic=lambda v: self.h(v, h_dist))
        if visitor.target_vertex is not None:
            v = visitor.target_vertex
        else:
            d = dist.a[verticies_position_indices]
            min = np.min(d)
            argmin = np.where(dist.a == min)[0][0]
            v = g.vertex(argmin)
        path = []
        while v != g.vertex(start_index):
            path.append(int(v))
            v = g.vertex(pred[v])
        path = path[::-1]
        return ids[path]


################################################
# END

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
        objects_distances = properties
        objects_distances += distances
        objects_distances = objects_distances[objects_distances != 0]
        objects_distances /= self.max_speed
        objects_distances /= np.max(objects_distances)
        visible_space = np.zeros(len(objects))
        for k in range(len(objects)):
            o = objects[k]
            visible_space[k] = self.divide_and_sum(o)
        visible_space /= np.max(visible_space)
        entropies = np.array([o.entropy for o in objects])
        vertices = np.zeros(len(objects))
        vertices += self.score_coef * scored_objects + self.entropy_coef * entropies + self.space_coef * visible_space
        objects_distances *= 2
        objects_distances += vertices
        g = self.create_graph(objects_distances)
        paths = g.printAllPaths(start_position_index, end_position_index, self.max_iterations)
        return paths[self.choose_best(paths, objects, objects_distances)]

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
        properties_copy = np.copy(properties["distances"])
        return self.path_planing.extended_select(selected_objects, scored_objects, objects, properties_copy,
                                                 start_position_index, end_position_index)


class NeighboursPathPlanning(ExtendedOrderingFunction):
    def __init__(self):
        super(NeighboursPathPlanning, self).__init__()
        self.path_planing = PathPlaning()

    def extended_select(self, selected_objects, scored_objects, objects, properties, start_position_index,
                        end_position_index=None):
        # calculate distances for neighbours
        distances = properties["distances"]
        dist = np.zeros((distances.shape[0], distances.shape[1]))
        for object_index in range(len(objects)):
            neighbours = np.array([key for key in objects[object_index].neighbors])
            dist[object_index, neighbours] = properties[object_index, neighbours]
        return self.path_planing.extended_select(selected_objects, scored_objects, objects, dist,
                                                 start_position_index, end_position_index)
