#!/usr/bin/env python
from builtins import super

import rospy
import numpy as np
import os
import sys
from geometry_msgs.msg import Pose
import helper_pkg.utils as utils
from helper_pkg.utils import Constants
import helper_pkg.PredictionManagement as pm
from scipy.spatial import distance, cKDTree
from scipy.stats import entropy
from replaning_functions import NAllReplaning, NNeighboursMAllReplaning, NNeighboursReplaning, NoReplaning, \
    ProbabilityChangeReplanning
from ordering_functions import TSPath
from selection_functions import AboveAverage, BigDrop, AboveMinimum
from fitness_functions import Neighbour, Maximum, RestrictedProbability
from section_algorithm_utils import SectionMapObjectTypes
from section_algorithm_utils import SectionUtils



class ModeTypes:
    FITNESS = 0
    SELECTION = 1
    ORDERING = 2
    PLANING = 3


np.set_printoptions(threshold=sys.maxsize)


class SectionsAlgrotihms(object):
    # init
    def __init__(self, sections, world_map, properties):
        super(SectionsAlgrotihms, self).__init__()
        self.level = 0
        self.sections = sections
        self._transitions = properties["transitions"]
        self._intersections = properties["intersections"]
        self._true_intersections = properties["intersections"]
        self._empty_intersections = self.create_empty_intersections(self._intersections)
        self.world_map = world_map
        self._region_plan, self._region_index, self._region_plan_score = [], 0, []
        self._section_plan, self._section_index, self._section_plan_score = [], 0, []
        self._point_plan, self._point_index, self._point_plan_score = [], 0, []
        self._score_algorithm_index = 0
        self._selection_algorithm_index = 0
        self._extended_selection_algorithm_index = 0
        self._score_algorithms = [Maximum(), RestrictedProbability(), Neighbour()]
        self._score_algorithms_names = ["Maximum", "RestrictedProbability", "Neighbour"]
        self._selection_algorithms = [AboveMinimum(0.05), AboveAverage(), BigDrop()]
        self._extended_selection_algorithms = [TSPath()]
        self._selection_algorithms_names = [ "AboveMinimum", "AboveAverage", "BigDrop" ]
        self._extended_selection_algorithms_names = ["TSP"]
        self._replaning_algorithms = [NNeighboursReplaning(3), NAllReplaning(3), NNeighboursMAllReplaning(3, 5),
                                      NoReplaning(), ProbabilityChangeReplanning()]
        self._modes = [(0, 0, 1, 4), (1, 4, 0, 4)]
        self._probability_map = None
        self.coordinates = self.create_coordinates(sections)
        self._distances, self._inner_distances, self._points_tree, self._points_region_section_mapping, self._object_times = self.distances()
        self._battery_life = 1.0
        self.drone_last_position = None
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        self._region_closest = (None, drone_region)
        self._region_end_position = []
        self._region_start_position = drone_region
        self._point_closest = (None, drone_point)
        self._point_start_position = drone_point
        self._point_end_position = []
        self._last_drone_coordinates = None
        self._last_planned_section = -1
        self._last_planned_region = -1
        self._objects_modes = {SectionMapObjectTypes.Section: self.get_section_mode,
                               SectionMapObjectTypes.Region: self.get_region_mode,
                               SectionMapObjectTypes.Point: self.get_point_mode}
        self._iterations_section = 1
        self._iterations_region = 1
        self._iterations_point = 1
        self.replan_section_check = False
        self.replan_region_check = False
        self.replan_point_check = False
        self._point_starts = {}
        self._region_starts = {}
        self._section_starts = {}
        self.drone_localization_map = self.create_drone_localization_map()
        self._silent = False
        self._last_returned_point_indices = None
        self._last_returned_yaw = None
        self._is_target_seen = False
	self._need_to_update = True

    # inner states
    def update_probability_map(self, target_positions):
        self._probability_map = target_positions

    def update_target_information(self, target_information):
        if target_information is not None and target_information.quotient > 0:
            self._is_target_seen = True


    def update_drone_position(self, drone_positions):
        self.drone_last_position = drone_positions

    def create_coordinates(self, sections):
        coordinates = {}
        for section_index in range(len(sections)):
            section = sections[section_index]
            for (region_index, _) in section.objects.items():
                region = section.objects[region_index]
                for (point_index, _) in region.objects.items():
                    point = region.objects[point_index]
                    for (_, coor) in point.objects.items():
                        if coor[0] not in coordinates:
                            coordinates[coor[0]] = {}
                        if coor[1] not in coordinates[coor[0]]:
                            coordinates[coor[0]][coor[1]] = []
                        coordinates[coor[0]][coor[1]].append([section_index, region_index, point_index])
        return coordinates

    def create_empty_intersections(self, intersections):
        empty_intersections = {}
        for key in intersections:
            empty_intersections[key] = {}
            for key1 in intersections[key]:
                empty_intersections[key][key1] = {}
        return empty_intersections

    def calculate_entropy(self, value):
        return np.clip(np.nan_to_num(-np.log2(value)), 0, None) * value

    def update_scores(self, sections=None):
        if self._probability_map is None:
            return
        if sections is not None:
            for (section_index, _) in sections.items():
                section = self.sections[section_index]
                section.score = 0
                section.entropy = 0
                for (region_index, _) in section.objects.items():
                    region = section.objects[region_index]
                    region.score = 0
                    region.entropy = 0
                    for (point_index, _) in region.objects.items():
                        point = region.objects[point_index]
                        point.score = 0
                        point.entropy = 0
                        for (_, coor) in point.objects.items():
                            point.score += self._probability_map[coor[0], coor[1]]
                            if self._probability_map[coor[0], coor[1]] > 0:
                                point.entropy += self.calculate_entropy(self._probability_map[coor[0], coor[1]])
                        region.score += point.score
                        region.entropy += point.entropy
                    section.score += region.score
                    section.entropy += region.entropy
        else:
            for (section_index, _) in self.sections.items():
                section = self.sections[section_index]
                section.entropy = 0
                section.score = 0
                for (region_index, _) in section.objects.items():
                    region = section.objects[region_index]
                    region.score = 0
                    region.entropy = 0
                    for (point_index, _) in region.objects.items():
                        point = region.objects[point_index]
                        point.score = 0
                        point.entropy = 0
            for coor_x in self.coordinates:
                for coor_y in self.coordinates[coor_x]:
                    for point in self.coordinates[coor_x][coor_y]:
                        # score
                        self.sections[point[0]].objects[point[1]].objects[point[2]].score += self._probability_map[
                            coor_x, coor_y]
                        self.sections[point[0]].objects[point[1]].score += self._probability_map[coor_x, coor_y]
                        self.sections[point[0]].score += self._probability_map[coor_x, coor_y]
                        # entropy
                        if self._probability_map[coor_x, coor_y] > 0:
                            en = np.clip(
                                -np.log2(self._probability_map[coor_x, coor_y]) * self._probability_map[coor_x, coor_y],
                                0, None)
                            self.sections[point[0]].objects[point[1]].objects[point[2]].entropy += en
                            self.sections[point[0]].objects[point[1]].entropy += en
                            self.sections[point[0]].entropy += en

    def distances(self):
        dc = rospy.get_param("drone_configuration")
        drone_horizontal_speed = dc["control"]["video_max_horizontal_speed"]
        drone_rotation_speed = dc["control"]["video_max_rotation_speed"]
        sections_ids = np.sort(np.array([i for (i, _) in self.sections.items()]))
        regions_ids = [np.sort(np.array([ri for (ri, _) in self.sections[sections_ids[si]].objects.items()])) for si in
                       range(len(sections_ids))]
        points_ids = [
            [np.sort(np.array(
                [pi for (pi, _) in self.sections[sections_ids[si]].objects[regions_ids[si][ri]].objects.items()])) for
                ri in range(len(regions_ids[si]))] for si in range(len(sections_ids))]

        sections_centroids = np.array(
            [np.array(self.sections[sections_ids[si]].centroid) for si in range(len(sections_ids))])

        regions_centroids = [np.array(
            [np.array(self.sections[sections_ids[si]].objects[regions_ids[si][ri]].centroid) for ri in
             range(len(regions_ids[si]))]) for si in range(len(sections_ids))]

        points_centroids = [
            [np.array(
                [self.sections[sections_ids[si]].objects[regions_ids[si][ri]].objects[points_ids[si][ri][pi]].centroid
                 for pi in
                 range(len(points_ids[si][ri]))]) for ri in
                range(len(regions_ids[si]))] for si in range(len(sections_ids))]
        section_distances = np.array(distance.cdist(sections_centroids, sections_centroids))
        region_distances = np.array([distance.cdist(regions, regions) for regions in regions_centroids])
        point_distances = np.array(
            [np.array([distance.cdist(points, points) for points in regions]) for regions in points_centroids])
        inner_section_distances = np.array([region.sum(axis=1).min() for region in region_distances])
        inner_region_distances = np.array(
            [[point.sum(axis=1).min() for point in regions] for regions in point_distances])
        inner_point_distances = np.array([[[1 for _ in point] for point in regions] for regions in point_distances])
        rd = {}
        for si in range(len(sections_ids)):
            rd[sections_ids[si]] = region_distances[si]
        ird = {}
        for si in range(len(sections_ids)):
            ird[sections_ids[si]] = inner_region_distances[si]
        pd = {}
        for si in range(len(sections_ids)):
            pd[sections_ids[si]] = {}
            for ri in range(len(regions_ids[si])):
                pd[sections_ids[si]][regions_ids[si][ri]] = point_distances[si][ri]
        ipd = {}
        for si in range(len(sections_ids)):
            ipd[sections_ids[si]] = {}
            for ri in range(len(regions_ids[si])):
                ipd[sections_ids[si]][regions_ids[si][ri]] = inner_point_distances[si][ri]
        point_distances = pd
        region_distances = rd
        inner_region_distances = ird
        inner_point_distances = ipd
        point_centers = []
        point_mapping = []
        for si in range(len(sections_ids)):
            for ri in range(len(regions_ids[si])):
                for pi in range(len(points_ids[si][ri])):
                    point_centers.append(self.sections[sections_ids[si]].objects[regions_ids[si][ri]].objects[
                                             points_ids[si][ri][pi]].centroid)
                    point_mapping.append([sections_ids[si], regions_ids[si][ri], points_ids[si][ri][pi]])
        points_centroids_tree = cKDTree(point_centers)
        sections_times = np.copy(section_distances) / drone_horizontal_speed * 1.2
        region_times = {}
        for si in region_distances:
            region_times[si] = region_distances[si] / drone_horizontal_speed * 1.2
        points_times = {}
        for i in range(len(sections_ids)):
            si = sections_ids[i]
            s = self.sections[si]
            section_array = {}
            for ii in range(len(regions_ids[i])):
                ri = regions_ids[i][ii]
                r = s.objects[ri]
                region_array = []
                for iii in range(len(points_ids[i][ii])):
                    pi = points_ids[i][ii][iii]
                    p = r.objects[pi]
                    point_array = []
                    for iiii in range(len(points_ids[i][ii])):
                        pi1 = points_ids[i][ii][iiii]
                        p1 = r.objects[pi1]
                        if pi1 == pi:
                            point_array.append(0.0)
                        else:
                            value = self.compute_time(p, p1, drone_horizontal_speed, drone_rotation_speed)
                            point_array.append(value)
                    region_array.append(point_array)
                section_array[ri] = np.array(region_array)
            points_times[si] = section_array
        return {"sections": section_distances, "regions": region_distances, "points": point_distances}, {
            "sections": inner_section_distances, "regions": inner_region_distances,
            "points": inner_point_distances}, points_centroids_tree, point_mapping, {
                   "sections": sections_times, "regions": region_times, "points": points_times}

    def compute_time(self, pos1, pos2, horizontal_speed, rotation_speed):
        t = utils.Math.euclidian_distance(utils.Math.Point(*pos1.centroid),
                                          utils.Math.Point(*pos2.centroid)) / horizontal_speed
        t += np.abs(pos1.data[4] - pos2.data[4]) / rotation_speed
        return t

    def get_properties(self, section=None, region=None):
        if section is None:
            distances, inner_distances = self._distances["sections"], self._inner_distances["sections"]
            transitions = self._transitions["sections"]
            intersections = self._intersections["sections"]
            times = self._object_times["sections"]
        elif region is None:
            distances, inner_distances = self._distances["regions"][section], self._inner_distances["regions"][section]
            times = self._object_times["regions"][section]
            intersections = self._intersections["regions"]
            transitions = self._transitions["regions"]
        else:
            distances, inner_distances = self._distances["points"][section][region], \
                                         self._inner_distances["points"][section][region]
            times = self._object_times["points"][section][region]
            intersections = self._intersections["points"]
            transitions = self._transitions["points"]
        return {"distances": distances, "inner_distances": inner_distances, "battery_life": self._battery_life,
                "transitions": transitions, "times": times, "intersections": intersections, "is_target_visible": self._is_target_seen}

    # help functions
    def get_selection_mode(self, section_object_type, mode_type):
        return self._modes[self._objects_modes[section_object_type]()][mode_type]

    def get_section_mode(self):
        return 0

    def get_region_mode(self):
        return 1

    def get_point_mode(self):
        return 1

    def create_drone_localization_map(self):
        drone_localization_map = np.zeros((self.world_map.width, self.world_map.height, 3), dtype=np.int32)
        for x in xrange(self.world_map.width):
            for y in xrange(self.world_map.height):
                real_x, real_y = self.world_map.get_coordination_from_map_indices_vectors(np.array([x]), np.array([y]))
                dd, ii = self._points_tree.query([[real_x[0], real_y[0]]], 1)
                [s, r, p] = self._points_region_section_mapping[ii[0]]
                if ii[0] != p:
                    self.log_info("diff {},{}".format(ii[0], p))
                drone_localization_map[x, y, :] = [s, r, p]
                """
                if x in self.coordinates and y in self.coordinates[x]:
                    drone_localization_map[x, y, :] = np.array(self.coordinates[x][y][0]).astype(np.int32)
                else:
                    x_key_array = np.array([key for key in self.coordinates])
                    idx_arg = (np.abs(x_key_array - x)).argmin()
                    idx = x_key_array[idx_arg]
                    y_key_array = np.array([key for key in self.coordinates[idx]])
                    idy_arg = (np.abs(y_key_array - y)).argmin()
                    idy = y_key_array[idy_arg]
                    drone_localization_map[x, y, :] = np.array(self.coordinates[idx][idy][0]).astype(np.int32)
                """
        return drone_localization_map

    def get_drone_section_region_point(self):
        if self.drone_last_position is not None:
            map_point = self.world_map.map_point_from_real_coordinates(self.drone_last_position[0],
                                                                       self.drone_last_position[1], 0)
            srp = self.drone_localization_map[map_point.x, map_point.y]
            self.log_info("Drone loc: {}".format(srp))
            self.log_info("Centroid {}".format(self.sections[srp[0]].objects[srp[1]].objects[srp[2]].centroid))
            return self.drone_localization_map[map_point.x, map_point.y]
        else:
            self.log_info("Drone loc wasnt found: {}".format("None"))
            return [None, None, None]

    def init_section_closest(self):
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        self.log_info("Init section closest point {}".format(drone_section))
        return (None, drone_section)

    def init_region_closest(self):
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        self.log_info("Init region closest point {}".format(drone_region))
        return (None, drone_region)

    def init_point_closest(self):
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        self.log_info("Init point closest point {}".format(drone_point))
        return (None, drone_point)

    def get_start_id(self, object_type, object_id):
        if object_type == SectionMapObjectTypes.Point:
            return self._point_starts[object_id]
        elif object_type == SectionMapObjectTypes.Region:
            return self._region_starts[object_id]
        else:
            return self._section_starts[object_id]

    def set_start_id(self, object_type, object_id, value):
        if object_type == SectionMapObjectTypes.Point:
            self._point_starts[object_id] = value
        elif object_type == SectionMapObjectTypes.Region:
            self._region_starts[object_id] = value
        else:
            self._section_starts[object_id] = value

    def find_closest(self, object1, object2):
        centroids1 = np.array([np.array(o.centroid) for (_, o) in object1.objects.items()])
        centroids2 = np.array([np.array(o.centroid) for (_, o) in object2.objects.items()])
        ids1 = np.array([i for i in object1.objects])
        ids2 = np.array([i for i in object2.objects])
        dist = distance.cdist(centroids1, centroids2, 'euclidean')
        m = np.max(dist[0, :])
        m_index = (0, 0)
        for index in range(len(centroids1)):
            # indices_without_current_index = np.r_[0:index, index + 1:len(centroids2)]
            # if len(indices_without_current_index) > 0:
            # minimum = np.min(dist[index, np.r_[0:index, index + 1:len(centroids2)]])
            minimum = np.min(dist[index])
            if minimum < m:
                m = minimum
                # m_index = (index, np.argmin(dist[index, np.r_[0:index, index + 1:len(centroids2)]]))
                m_index = (index, np.argmin(dist[index]))
        res = [ids1[m_index[0]], ids2[m_index[1]]]
        self.log_info("Find closest res {}".format(res))
        return res

    def find_closest_for_centroid(self, centroid, next_object):
        centroids1 = np.array([np.array(o.centroid) for (key, o) in next_object.objects.items()])
        ids = np.array([key for key in next_object.objects])
        dist = distance.cdist(np.array([np.array(centroid)]), centroids1, 'euclidean')
        res = ids[np.argmin(dist)]
        self.log_info("Find closest for centroids res {}".format(res))
        return res

    def find_end_points(self, map_object, next_map_object):
        if map_object.object_id == next_map_object.object_id:
            ids = [i for i in map_object.objects]
            return np.random.choice(ids, 1)
        if next_map_object.object_id in map_object.neighbors:
            res = np.unique(
                [item[0] for item in map_object.neighbors[next_map_object.object_id] if item[0] in map_object.objects])
            self.log_info("Find end points res {}".format(res))
            return res
        else:
            res = [self.find_closest(map_object, next_map_object)[0]]
            self.log_info("Find end points res {}".format(res))
            return res

    def find_start_point(self, last_map_object, current_map_object, end_point):
        self.log_info(
            "Find start point {}, {}, {}".format(last_map_object.object_id, current_map_object.object_id, end_point))
        if current_map_object.object_id in last_map_object.neighbors:
            for item in last_map_object.neighbors[current_map_object.object_id]:
                if end_point == item[0]:
                    self.log_info("Find start point item {}".format(item[1]))
                    return item[1]
	else:
            return self.find_closest_for_centroid(last_map_object.objects[end_point].centroid, current_map_object)

    def decide_right_mechanism(self, objects, end_positions, object_type):
	if self._need_to_update:
	    self.update_scores()
	self._need_to_update = False
        first_obstacles = 4
        scores = [o.score for (_, o) in objects.items()]
        sorted = np.sort(scores)[::-1]
        s = np.sum(scores)
        # TSPath()
        ordering_function = 0
        # Maximum(), RestrictedProbability(), Neighbour()
        # AboveMinimum(0.05), AboveAverage(), BigDrop()
        if s < 0.1: # 10%
            score_function = 0
            selection_function = 0
        else:
            ids = np.sort(np.array([i for (i, object) in objects.items()]))
            entropies = np.array([objects[i].entropy for i in ids])
            max_entropies = np.array([objects[i].maximal_entropy for i in ids])
            sort_indices = np.argsort(scores)[::-1]
            avg = np.average(scores)
            mask = []
            for ii in sort_indices:
                if scores[ii] < avg:
                    break
                mask.append(ii)
            mask = np.array(mask)
            #mask = sort_indices[:int(len(sort_indices)/ 2)]
            self.log_info(
                "First obstacles {}, mask {}".format(sorted[:first_obstacles], mask))
            selected_entropies = entropies[mask]
            selected_max_entropies = max_entropies[mask]
            if np.sum(selected_entropies) > 2 / 3 * np.sum(selected_max_entropies):
                score_function = 2
                selection_function = 2
            else:
                score_function = 1
                selection_function = 1

        # print("Mechanism for " + SectionMapObjectTypes.names(object_type))
        # print("Score: " + self._score_algorithms_names[score_function] + ", selection: " +
        #      self._selection_algorithms_names[selection_function] + ", ordering: " +
        #      self._extended_selection_algorithms_names[ordering_function])
        # print("End positions ", end_positions)
        if score_function >= 1:
            self._intersections = self._empty_intersections
        else:
            self._intersections = self._true_intersections
        self.log_info(
            "Mechanism score fun {}, select fun {}, ordering fun {}".format(score_function, selection_function,
                                                                            ordering_function))
        return score_function, selection_function, ordering_function

    def section_map_object_selection(self, objects, properties, object_type, start_position, end_position):
        self.log_info("Object type {}, objects ids {}, start_position {}, end position {}".format(object_type,
                                                                                                  [o for o in objects],
                                                                                                  start_position,
                                                                                                  end_position))
        score_function, selection_function, ordering_function = self.decide_right_mechanism(objects, end_position,
                                                                                            object_type)
        scored_objects = self._score_algorithms[score_function].score(objects, properties)
        self.log_info("Scored {}".format(scored_objects))
        selected_objects = self._selection_algorithms[selection_function].select(objects, properties, scored_objects)
        self.log_info("Select {}".format(selected_objects))
        exteded_objects = self._extended_selection_algorithms[ordering_function].extended_select(selected_objects,
                                                                                                 scored_objects,
                                                                                                 objects, properties,
                                                                                                 start_position,
                                                                                                 end_position)
        self.log_info("Ordering {}".format(exteded_objects))
        return exteded_objects

    # replannig

    def replan_section(self):
        if not self.replan_section_check and len(self._section_plan) > 0:
            if not self._silent:
                self.log_info("Section replaning")
            self.replan_section_check = True
            props = self.get_properties()
            props["original_score"] = self._section_plan_score

            replan, sections = self.replaning(SectionMapObjectTypes.Section, self.sections, self._section_plan,
                                              props, self._iterations_section)
            if replan:
                self.update_scores(sections)
                drone_section, _, _ = self.get_drone_section_region_point()
                self._section_plan, self._section_index = self.section_map_object_selection(
                    self.sections, props, SectionMapObjectTypes.Section, drone_section, []), 0
                if not self._silent:
                    self.log_info("Sections replanned.")
                return True
        return False

    def replan_region(self):
        if not self.replan_region_check and len(self._region_plan) > 0:
            if not self._silent:
                self.log_info("Region replaning")
            self.replan_region_check = True
            section = self.plan_i_section(0)
            current_section = self.sections[section]
            props = self.get_properties(section)
            props["original_score"] = self._region_plan_score
            replan, _ = self.replaning(SectionMapObjectTypes.Region, current_section.objects, self._region_plan,
                                       props, self._iterations_region)
            if replan:
                s, r, p = self.replan_start(section)
                if r is None:
                    r = self._region_start_position
                self._region_plan, self._region_index = self.section_map_object_selection(
                    self.sections[section].objects, props, SectionMapObjectTypes.Region,
                    r, self._region_end_position), 0
                if not self._silent:
                    self.log_info("Regions replanned.")
                return True
        return False

    def replan_point(self):
        if not self.replan_point_check and len(self._point_plan) > 0:
            if not self._silent:
                self.log_info("Point replaning")
            self.replan_point_check = True
            region, section = self.plan_i_region(0)
            current_region = self.sections[section].objects[region]
            props = self.get_properties(section, region)
            props["original_score"] = self._point_plan_score
            replan, _ = self.replaning(SectionMapObjectTypes.Point, current_region.objects, self._point_plan,
                                       props, self._iterations_point)
            if replan:
                s, r, p = self.replan_start(section, region)
                if p is None:
                    p = self._point_start_position
                self._point_plan, self._point_index = self.section_map_object_selection(
                    current_region.objects, props, SectionMapObjectTypes.Point,
                    p, self._point_end_position), 0
                if not self._silent:
                    self.log_info("Points replanned.")
                return True
        return False

    def replaning(self, section_map_object_type, objects, chosen_objects, properties, iteration):
        planing_alg = self._replaning_algorithms[self.get_selection_mode(section_map_object_type, ModeTypes.PLANING)]
        return planing_alg.replaning(objects, chosen_objects, properties, iteration)

    # planning
    def plan_i_section(self, i, save_section_variables=False):
        if not self._silent:
            self.log_info("Request for section with index {}".format(i), 1)
        if i < 0:
            return None
        # print("plan " + str(i) + " section")
        section_plan = list(self._section_plan)
        index = self._section_index + i
        if len(self._section_plan) == 0:
            start_section = self.init_section_closest()[1]
        else:
            start_section = self._section_plan[self._section_index]
        while index >= len(section_plan):
            s_plan = self.section_map_object_selection(self.sections, self.get_properties(),
                                                       SectionMapObjectTypes.Section, start_section, [])
            section_plan.extend(list(s_plan))
            start_section = section_plan[len(section_plan) - 1]
        if save_section_variables or len(self._section_plan) == 0:
            self._iterations_section += 1
            self.replan_section_check = False
            self._section_plan = np.array(section_plan)
            self._section_plan_score = [self.sections[i].score for i in self._section_plan]
            self._section_index = index
            if not self._silent:
                self.log_info("Section saved values:")
                self.log_info("Current section plan iterations: {}".format(self._iterations_section))
                self.log_info("Already planned: {}".format(self.replan_section_check))
                self.log_info("Section plan: {}".format(self._section_plan))
                self.log_info("Section index: {}".format(self._section_index))
                self.log_info("Section plan score: {}".format(self._section_plan_score))
        if not self._silent:
            self.log_info("End of request for section with index {}".format(i), -1)
        return section_plan[index]

    def plan_i_region(self, i, save_region_variables=False):
        if not self._silent:
            self.log_info("Request for region with index {}".format(i), 1)
        if i < 0:
            return None, None
        # print("plan " + str(i) + " region")
        index = self._region_index + i
        region_plan = self._region_plan
        region_start_position = self.init_region_closest()[1]
        region_end_position = self._region_end_position
        section = self.plan_i_section(0)
        section_index = 0
        if section != self._last_planned_section:
            section_index = -1
        while index >= len(region_plan):
            index -= len(region_plan)
            section_1 = self.plan_i_section(section_index, False)
            section = self.plan_i_section(section_index + 1, save_region_variables)
            current_section = self.sections[section]
            region_end_position = self.region_end_point(section_index, current_section)
            region_start_position = self.region_start_point(section_1, current_section, region_plan,
                                                            region_start_position)
            region_plan, region_index = self.section_map_object_selection(
                current_section.objects, self.get_properties(section), SectionMapObjectTypes.Region,
                region_start_position, region_end_position), 0
            section_index += 1
        if save_region_variables or len(self._region_plan) == 0:
            self._iterations_region += 1
            self.replan_region_check = False
            self._last_planned_section = section
            self._region_end_position = region_end_position
            self._region_start_position = region_start_position
            self._region_plan = region_plan
            self._region_plan_score = [self.sections[self._last_planned_section].objects[i].score for i in
                                       self._region_plan]
            self._region_index = index
            if not self._silent:
                self.log_info("Region saved values:")
                self.log_info("Current regions plan iterations: {}".format(self._iterations_region))
                self.log_info("Already planned: {}".format(self.replan_region_check))
                self.log_info("Last planned section: {}".format(self._last_planned_section))
                self.log_info("Region end positions: {}".format(self._region_end_position))
                self.log_info("Region start position: {}".format(self._region_start_position))
                self.log_info("Region plan: {}".format(self._region_plan))
                self.log_info("Region index: {}".format(self._region_index))
                self.log_info("Region plan score: {}".format(self._region_plan_score))
        if not self._silent:
            self.log_info("End of request for region with index {}".format(i), -1)
        return region_plan[index], section

    def plan_i_point(self, i, save_point_variables=False):
        if not self._silent:
            self.log_info("Request for point with index {}".format(i), 1)
        if i < 0:
            return None, None, None
        # print("plan " + str(i) + " point")
        index = self._point_index + i
        point_plan = self._point_plan
        point_start_position = self.init_point_closest()[1]
        point_end_position = self._point_end_position
        region, section = self.plan_i_region(0)
        region_index = 0
        if region_index != self._last_planned_region:
            region_index = -1
        while index >= len(point_plan):
            index -= len(point_plan)
            region_1, section_1 = self.plan_i_region(region_index, False)
            region, section = self.plan_i_region(region_index + 1, save_point_variables)
            current_region = self.sections[section].objects[region]
            point_end_position = self.point_end_point(region_index, current_region)
            point_start_position = self.point_start_point(region_1, section_1, current_region, point_plan,
                                                          point_start_position)
            point_plan, point_index = self.section_map_object_selection(
                current_region.objects, self.get_properties(section, region), SectionMapObjectTypes.Point,
                point_start_position, point_end_position), 0
            region_index += 1
        if save_point_variables or len(self._point_plan) == 0:
            self._iterations_point += 1
            self.replan_point_check = False
            self._last_planned_region = region_index
            self._point_plan = point_plan
            self._point_index = index
            self._point_end_position = point_end_position
            self._point_start_position = point_start_position
            self._point_plan_score = [self.sections[section].objects[region].objects[i].score for i in self._point_plan]
            if not self._silent:
                self.log_info("Point saved values:")
                self.log_info("Current points plan iterations: {}".format(self._iterations_point))
                self.log_info("Already planned: {}".format(self.replan_point_check))
                self.log_info("Last planned region: {}".format(self._last_planned_region))
                self.log_info("Point end positions: {}".format(self._point_end_position))
                self.log_info("Point start position: {}".format(self._point_start_position))
                self.log_info("Point plan: {}".format(self._point_plan))
                self.log_info("Point index: {}".format(self._point_index))
                self.log_info("Point plan score: {}".format(self._point_plan_score))
        if not self._silent:
            self.log_info("End of request for point with index {}".format(i), -1)
        return point_plan[index], region, section

    def delete_plan(self):
        self._iterations_section = 0
        self.replan_section_check = False
        self._section_plan = np.array([])
        self._section_plan_score = []
        self._section_index = 0
        self.delete_after_section_replan()

    def delete_after_section_replan(self):
        self.log_info("Delete after section replan")
        self._last_planned_section = -1
        self._region_plan = []
        self._region_index = 0
        self._region_end_position = []
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        self._region_start_position = drone_region
        self.delete_after_region_replan()

    def delete_after_region_replan(self):
        self.log_info("Delete after region replan")
        self._last_planned_region = -1
        self._point_plan = []
        self._point_index = 0
        self._point_end_position = []
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        self._point_start_position = drone_point

    def point_start_point(self, region_1, section_1, current_region, point_plan, point_start_position):
        self.log_info("Point start point {}, {}, {}, {}".format(region_1, section_1, point_plan, point_start_position))
        if region_1 is not None:
            last_region = self.sections[section_1].objects[region_1]
        else:
            last_region = None
        if len(point_plan) > 0 and last_region is not None:
            point_start_position = self.find_start_point(last_region, current_region, point_plan[- 1])
        self.log_info("Planned point start position {}".format(point_start_position))
        return point_start_position

    def replan_start(self, section=None, region=None):
        [s, r, p] = self.get_drone_section_region_point()
        self.log_info("Replanned start {}, {}, {}".format(s, r, p))
        if section is not None and s == section and region is not None and region == r:
            return s, r, p
        if section is not None and s == section and region is None:
            return s, r, p
        if section is None and region is None:
            return s, r, p
        return None, None, None

    def find_current_drone_closest_point(self, section=None, region=None):
        dd, ii = self._points_tree.query(self.drone_last_position[:2], 100)
        for i in ii:
            [s, r, _] = self._points_region_section_mapping[i]
            if section is not None and s != section:
                continue
            if region is not None and r != region:
                continue
            self.log_info("Current drone closest_point {},{},{}".format(s, r, i))
            return s, r, i
        self.log_info("Current drone closest_point {}".format("None"))
        return None, None, None

    def point_end_point(self, i, current_region):
        region2, section2 = self.plan_i_region(i + 2)
        next_region = self.sections[section2].objects[region2]
        self.log_info("Point end point region2 {}, section2 {}, next_region {}".format(region2, section2, next_region))
        return self.find_end_points(current_region, next_region)

    def region_end_point(self, i, current_section):
        next_section = self.sections[self.plan_i_section(i + 2)]
        self.log_info("Point end point next_section {}".format(next_section))
        return self.find_end_points(current_section, next_section)

    def region_start_point(self, section_1, current_section, region_plan, region_start_position):
        if section_1 is not None:
            last_section = self.sections[section_1]
        else:
            last_section = None
        if len(region_plan) > 0 and last_section is not None:
            region_start_position = self.find_start_point(last_section, current_section, region_plan[- 1])
        self.log_info("Region start point region_start_position {}".format(region_start_position))
        return region_start_position

    def are_close(self, drone_position, goal_position):
        dif = np.abs(goal_position - drone_position)
        self.log_info(
            "drone difference: x: {}, y: {}, z: {}, verical: {}, yaw: {}".format(dif[0], dif[1], dif[2], dif[3],
                                                                                 dif[4]))
        if dif[0] < 0.1 and dif[1] < 0.1 and dif[2] < 0.1 and dif[4] < 0.1:
            self.log_info("next")
            return True
        return False

    def next_point(self, current_drone_position):
        if current_drone_position is not None:
            current_drone_position = np.array(current_drone_position)
            self.update_drone_position(current_drone_position)
        if not self._silent:
            self.log_info("################################")
        save = False
        if len(self._point_plan) == 0:
            if not self._silent:
                self.log_info("Saving calculation: zero point plan")
            save = True
        point, region, section = self.plan_i_point(0, save)
        try:
            print("Point {}, region {}, section {}".format(point, region, section))
            current_point = np.copy(self.sections[section].objects[region].objects[point].data)
            current_point_score = self.sections[section].objects[region].objects[point].score
            if current_point_score == 0:
                if self._last_returned_point_indices is None or self._last_returned_point_indices[0] != section or \
                        self._last_returned_point_indices[1] != region or self._last_returned_point_indices[
                    2] != point or self._last_returned_yaw is None:
                    next_predicted_point, next_predicted_region, next_predicted_section = self.plan_i_point(1, False)
                    next_predicted_point_data = \
                    self.sections[next_predicted_section].objects[next_predicted_region].objects[
                        next_predicted_point].data
                    self._last_returned_point_indices = [section, region, point]
                    self._last_returned_yaw = utils.Math.calculate_yaw_from_points(current_point[0], current_point[1],
                                                                                   next_predicted_point_data[0],
                                                                                   next_predicted_point_data[1])
                # current_point[4] = self._last_returned_yaw
        except Exception as e:
            print("Returning current position because an error '{}' was encountered.".format(e))
            self.delete_after_section_replan()
            self.plan_i_point(0, True)
            return current_drone_position
        self.log_info("INFO")
        self.log_info(
            "Drone last position {}".format(self.drone_last_position[np.r_[:3, 4:len(self.drone_last_position)]]))
        self.log_info("Current point {}".format(current_point))
        self.log_info(
            "Are close {}".format(
                self.are_close(self.drone_last_position[np.r_[:3, 4:len(self.drone_last_position)]], current_point)))
        # (x position, y position, z position, vertical rotation of camera, yaw)
        converted_last_position = self.convert_to_correct_world_joints(
            self.drone_last_position[np.r_[:3, 4:len(self.drone_last_position)]])
        converted_current_point = self.convert_to_correct_world_joints(current_point)
        if self.are_close(converted_current_point, converted_last_position):
	    self.update_scores()
            if self.replan_section():
                self.delete_after_section_replan()
                if not self._silent:
                    self.log_info("Saving calculation: sections replaning")
                self.plan_i_point(0, True)
            elif self.replan_region():
                self.delete_after_region_replan()
                if not self._silent:
                    self.log_info("Saving calculation: regions replaning")
                self.plan_i_point(0, True)
            else:
                self.replan_point()
            if not self._silent:
                self.log_info("Saving calculation: point replaning")
            point, region, section = self.plan_i_point(1, True)
            current_point = self.sections[section].objects[region].objects[point].data
            self._is_target_seen = False
        # print("section index", self._section_index)
        # print("region index", self._region_index)
        # print("point index", self._point_index)
        # print("section plan", self._section_plan)
        # print("region plan", self._region_plan)
        # print("point plan", self._point_plan)
        self.log_info(
            "section index: {}, region index: {}, point index {}".format(self._section_index, self._region_index,
                                                                         self._point_index))
        self.level = 0
        return self.convert_to_correct_world_joints(np.insert(current_point, 3, 0))

    def convert_to_correct_world_joints(self, point):
        yaw = point[-1] % (2 * np.pi)
        if yaw > np.pi:
            yaw -= (2 * np.pi)
        point[-1] = yaw
        return point

    def log_info(self, string, level=0):
        if level < 0:
            self.level += level
        rospy.loginfo("{}{}".format("".join(["  " for _ in range(self.level)]), str(string)))
        if level > 0:
            self.level += level


class SectionAlgorithm(pm.PredictionAlgorithm):
    DRONE_POSITIONS = 1
    TARGET_POSITIONS = 1
    INPUT_TYPE = Constants.InputOutputParameterType.pointcloud
    OUTPUT_TYPE = Constants.InputOutputParameterType.pointcloud
    MAP_PRESENCE = Constants.MapPresenceParameter.yes

    def __init__(self, mapa):
        super(SectionAlgorithm, self).__init__()
        environment_configuration = rospy.get_param("environment_configuration")
        section_file = environment_configuration["map"]["section_file"]
        try:
            d = SectionUtils.unjson_sections(section_file)
            self._sections = d["objects"]
            properties = d["additional"]
            self._algorithm = SectionsAlgrotihms(self._sections, mapa, properties)
            rospy.loginfo("Section algorithm was loaded.")
        except IOError as ioerror:
            print(ioerror)
            self._sections = None

        self._first_update = False
        self._saved_values = None
        file_name = os.path.split(self._obstacles_file_path)[1][7:]
        self.f = open("searching_test/drone_log_file_{}.txt".format(file_name), "w")

    def state_variables(self, data, drone_positions, target_positions, map, **kwargs):
        return {}

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        drone_position = drone_positions[0]
        target_position = target_positions[0]
        if self._sections is not None:
            # current_drone_position = utils.DataStructures.pointcloud2_to_pose_stamped(drone_position)
            # current_drone_position = np.array([
            #    current_drone_position.pose.position.x,
            #    current_drone_position.pose.position.y,
            #    current_drone_position.pose.position.z,
            #    current_drone_position.pose.orientation.x,
            #    current_drone_position.pose.orientation.y,
            #    current_drone_position.pose.orientation.z
            # ])
            target_probability_map = utils.DataStructures.array_to_image(
                utils.DataStructures.pointcloud2_to_array(target_position), map.real_width, map.real_height,
                map.resolution)
            self._algorithm.update_probability_map(target_probability_map)
            #if not self._first_update:
            #    self._algorithm.update_scores()
            current_drone_position = None
            if "current_position" in kwargs:
                current_drone_position = kwargs["current_position"]
            position = self._algorithm.next_point(current_drone_position)
            self.log(kwargs, target_probability_map, current_drone_position, position)
	    self._algorithm._need_to_update = True
            return np.array([position])
        else:
            current_drone_position = None
            if "current_position" in kwargs:
                current_drone_position = kwargs["current_position"]
                current_drone_position.append(0)
                current_drone_position = np.array(current_drone_position)
            else:
                current_drone_position = drone_position
            self.log(kwargs, None, current_drone_position, current_drone_position)
	    self._algorithm._need_to_update = True
            return current_drone_position

    def log(self, kwargs, target_probability_map, drone_position, target_point):
        ar = []
        ar.append(rospy.Time.now().to_sec())
        ent = 0
        if target_probability_map is not None:
            reshaped = target_probability_map.reshape(-1)
            ent = np.sum(np.clip(np.nan_to_num(-np.log2(reshaped)), 0, None) * reshaped)
        ar.append(ent)
        plan_values = [[], [], []]
        if (self._saved_values is not None and (
                np.any(self._saved_values[0] != self._algorithm._section_plan) or np.any(self._saved_values[
                                                                                             1] != self._algorithm._region_plan) or np.any(
            self._saved_values[2] != self._algorithm._point_plan))) or self._saved_values is None:
            plan_values = [self._algorithm._section_plan, self._algorithm._region_plan, self._algorithm._point_plan]
            self._saved_values = plan_values
        pv = []
        for p in plan_values:
            pv.append(self.convert_array_to_string(p, ":"))
        ar.append(self.convert_array_to_string(pv))
        current_drone_position = np.array([0, 0, 0, 0, 0, 0])
        if "current_position" in kwargs:
            current_drone_position = kwargs["current_position"]
            current_drone_position.append(0)
            current_drone_position = np.array(current_drone_position)
        ar.append(self.convert_array_to_string(current_drone_position))
        target_destination = np.array([0, 0, 0, 0, 0, 0])
        if target_point is not None:
            target_destination = target_point
        ar.append(self.convert_array_to_string(target_destination))
        self.f.write("{}\n".format(self.convert_array_to_string(ar, ",")))

    def convert_array_to_string(self, array, delimiter=";"):
        str_array = [str(a) for a in array]
        s = delimiter.join(str_array)
        return s

    def restart_algorithm(self):
        self._algorithm.delete_plan()