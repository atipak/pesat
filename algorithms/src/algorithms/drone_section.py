#!/usr/bin/env python
from builtins import super

import rospy
import numpy as np
from geometry_msgs.msg import Pose
import helper_pkg.utils as utils
from helper_pkg.utils import Constants
import helper_pkg.PredictionManagement as pm
from scipy.spatial import distance
from replaning_functions import NAllReplaning, NNeighboursMAllReplaning, NNeighboursReplaning, NoReplaning
from ordering_functions import DirectPath, NeighboursPath, DirectPathPlanning, NeighboursPathPlanning, AStarSearch
from selection_functions import RatioElements, NElements, AboveAverage, BigDrop, AboveMinimum
from fitness_functions import DistanceScore, Neighbour, Maximum, Median
from section_map_creator import SectionMapObjectTypes


class ModeTypes:
    FITNESS = 0
    SELECTION = 1
    ORDERING = 2
    PLANING = 3


class SectionsAlgrotihms(object):
    # init
    def __init__(self, sections, world_map):
        super(SectionsAlgrotihms, self).__init__()
        self.sections = sections
        self.world_map = world_map
        self._region_plan, self._region_index = [], 0
        self._section_plan, self._section_index = [], 0
        self._point_plan, self._point_index = [], 0
        self._score_algorithm_index = 0
        self._selection_algorithm_index = 0
        self._extended_selection_algorithm_index = 0
        self._score_algorithms = [Median(), Maximum(), Neighbour(), DistanceScore()]
        self._score_algorithms_names = ["Median", "Maximum", "Neighbour", "DistanceScore"]
        self._selection_algorithms = [BigDrop(), AboveAverage(), NElements(int(len(sections) / 4)), RatioElements(0.2),
                                      AboveMinimum(0.05)]
        self._extended_selection_algorithms = [NeighboursPath(), DirectPath(), AStarSearch()]
        self._selection_algorithms_names = ["BigDrop", "AboveAverage", "NElements", "RatioElements", "AboveMinimum"]
        self._extended_selection_algorithms_names = ["NeighboursPath", "DirectPath", "AStarSearch"]
        self._replaning_algorithms = [NNeighboursReplaning(3), NAllReplaning(3), NNeighboursMAllReplaning(3, 5),
                                      NoReplaning()]
        self._modes = [(0, 0, 1, 0), (1, 4, 0, 0)]
        self._probability_map = None
        self.coordinates = self.create_coordinates(sections)
        self._distances, self._inner_distances = self.distances()
        self._battery_life = 1.0
        self.drone_last_position = None
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        self._region_closest = (None, drone_region)
        self._region_end_position = []
        self._point_closest = (None, drone_point)
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

    # inner states
    def update_probability_map(self, target_positions):
        self._probability_map = target_positions

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
                                point.entropy += np.clip(np.log2(self._probability_map[coor[0], coor[1]]), 0, None) * \
                                                 self._probability_map[
                                                     coor[0], coor[1]]
                        region.score += point.score
                        region.entropy += point.entropy
                    section.score += region.score
                    section.entropy += region.entropy
        else:
            for (section_index, _) in self.sections.items():
                section = self.sections[section_index]
                for (region_index, _) in section.objects.items():
                    region = section.objects[region_index]
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
                            en = np.clip(np.log2(self._probability_map[coor_x, coor_y]) * self._probability_map[coor_x, coor_y], 0, None)
                            self.sections[point[0]].objects[point[1]].objects[point[2]].entropy += en
                            self.sections[point[0]].objects[point[1]].entropy += en
                            self.sections[point[0]].entropy += en

    def distances(self):
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
            [[distance.cdist(points, points) for points in regions] for regions in points_centroids])
        inner_section_distances = np.array([region.sum(axis=1).min() for region in region_distances])
        inner_region_distances = np.array(
            [[point.sum(axis=1).min() for point in regions] for regions in point_distances])
        inner_point_distances = np.array([[[1 for _ in point] for point in regions] for regions in point_distances])
        return {"sections": section_distances, "regions": region_distances, "points": point_distances}, {
            "sections": inner_section_distances, "regions": inner_region_distances, "points": inner_point_distances}

    def get_properties(self, section=None, region=None):
        distances = self._distances["sections"]
        inner_distances = self._inner_distances["sections"]
        if section is None:
            distances, inner_distances = self._distances["sections"], self._inner_distances["sections"]
        elif region is None:
            distances, inner_distances = self._distances["regions"][section], self._inner_distances["regions"][section]
        else:
            distances, inner_distances = self._distances["points"][section][region], \
                                         self._inner_distances["points"][section][region]
        return {"distances": distances, "inner_distances": inner_distances, "battery_life": self._battery_life}

    # help functions
    def get_selection_mode(self, section_object_type, mode_type):
        return self._modes[self._objects_modes[section_object_type]()][mode_type]

    def get_section_mode(self):
        return 0

    def get_region_mode(self):
        return 1

    def get_point_mode(self):
        return 1

    def get_drone_section_region_point(self):
        if self.drone_last_position is not None:
            map_point = self.world_map.map_point_from_real_coordinates(self.drone_last_position[0],
                                                                       self.drone_last_position[1], 0)
            if map_point.x in self.coordinates and map_point.y in self.coordinates[map_point.x]:
                coor = self.coordinates[map_point.x][map_point.y]
                self._last_drone_coordinates = coor[0]
            else:
                if self._last_drone_coordinates is None:
                    idx = (np.abs(np.array([key for key in self.coordinates]) - map_point.y)).argmin()
                    idy = (np.abs(np.array([key for key in self.coordinates[idx]]) - map_point.x)).argmin()
                    self._last_drone_coordinates = self.coordinates[idx][idy][0]
            return self._last_drone_coordinates
        else:
            return [None, None, None]

    def init_section_closest(self):
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        return (None, drone_section)

    def init_region_closest(self):
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        return (None, drone_region)

    def init_point_closest(self):
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
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
        dist = distance.cdist(centroids1, centroids2, 'euclidean')
        m = np.max(dist[0, :])
        m_index = (0, 0)
        for index in range(len(centroids1)):
            indices_without_current_index = np.r_[0:index, index + 1:len(centroids2)]
            if len(indices_without_current_index) > 0:
                minimum = np.min(dist[index, np.r_[0:index, index + 1:len(centroids2)]])
                if minimum < m:
                    m = minimum
                    m_index = (index, np.argmin(dist[index, np.r_[0:index, index + 1:len(centroids2)]]))
        return m_index

    def find_closest_for_centroid(self, centroid, next_object):
        centroids1 = np.array([np.array(o.centroid) for (key, o) in next_object.objects.items()])
        dist = distance.cdist(np.array([np.array(centroid)]), centroids1, 'euclidean')
        return np.argmin(dist)

    def find_end_points(self, map_object, next_map_object):
        if next_map_object.object_id in map_object.neighbors:
            return np.unique(
                [item[0] for item in map_object.neighbors[next_map_object.object_id] if item[0] in map_object.objects])
        else:
            return [self.find_closest(map_object, next_map_object)[0]]

    def find_start_point(self, map_object, next_map_object, end_point):
        if next_map_object.object_id in map_object.neighbors:
            for item in map_object.neighbors[next_map_object.object_id]:
                if end_point == item[0]:
                    return item[1]
        else:
            return self.find_closest_for_centroid(map_object.objects[end_point].centroid, next_map_object)

    def decide_right_mechanism(self, objects, end_positions, object_type):
        first_obstacles = 4
        minimum_score = 0.7
        minimum_battery = 0.2
        # NeighboursPath(), DirectPath(), AStarSearch()
        if len(end_positions) == 0:
            ordering_function = 1
        else:
            if len(objects) < 5 and len(end_positions) == 1:
                ordering_function = 0
            else:
                ordering_function = 2
        scores = [o.score for (_, o) in objects.items()]
        sorted = np.sort(scores)
        s = np.sum(sorted[:first_obstacles])
        # Median(), Maximum(), Neighbour(), DistanceScore()
        if s > minimum_score:
            entropies = [o.entropy for o in objects]
            max_entropies = [o.maximal_entropy for o in objects]
            mask = scores > sorted[first_obstacles]
            selected_entropies = entropies[mask]
            selected_max_entropies = max_entropies[mask]
            if np.sum(selected_entropies) < 1 / 3 * np.sum(selected_max_entropies):
                score_function = 2
            else:
                score_function = 1
        else:
            if ordering_function < 2:
                score_function = 3
            else:
                score_function = 0
        # BigDrop(), AboveAverage(), NElements(int(len(sections) / 4)), RatioElements(0.2), AboveMinimum(0.05)
        if self._battery_life < minimum_battery:
            selection_function = 3
        if score_function < 3:
            selection_function = 0
        else:
            selection_function = 4
        #print("Mechanism for " + SectionMapObjectTypes.names(object_type))
        #print("Score: " + self._score_algorithms_names[score_function] + ", selection: " +
        #      self._selection_algorithms_names[selection_function] + ", ordering: " +
        #      self._extended_selection_algorithms_names[ordering_function])
        #print("End positions ", end_positions)
        return score_function, selection_function, ordering_function

    def section_map_object_selection(self, objects, properties, object_type, start_position, end_position):
        score_function, selection_function, ordering_function = self.decide_right_mechanism(objects, end_position,
                                                                                            object_type)
        scored_objects = self._score_algorithms[score_function].score(objects, properties)
        #print("scored", scored_objects)
        selected_objects = self._selection_algorithms[selection_function].select(objects, properties, scored_objects)
        #print("select", selected_objects)
        exteded_objects = self._extended_selection_algorithms[ordering_function].extended_select(selected_objects,
                                                                                                 scored_objects,
                                                                                                 objects, properties,
                                                                                                 start_position,
                                                                                                 end_position)
        #print("ordering", exteded_objects)
        return exteded_objects

    # replannig

    def replan_section(self):
        if not self.replan_section_check and len(self._section_plan) > 0:
            # print("replan section")
            self.replan_section_check = True
            replan, sections = self.replaning(SectionMapObjectTypes.Section, self.sections, self._section_plan,
                                              self.get_properties(), self._iterations_section)
            if replan:
                self.update_scores(sections)
                drone_section, _, _ = self.get_drone_section_region_point()
                self._section_plan, self._section_index = self.section_map_object_selection(self.sections,
                                                                                            self.get_properties(),
                                                                                            SectionMapObjectTypes.Section,
                                                                                            drone_section,
                                                                                            []), 0

    def replan_region(self):
        if not self.replan_region_check and len(self._region_plan) > 0:
            # print("replan region")
            self.replan_region_check = True
            section = self.plan_i_section(0)
            current_section = self.sections[section]
            replan, _ = self.replaning(SectionMapObjectTypes.Region, current_section.objects, self._region_plan,
                                       self.get_properties(section), self._iterations_region)
            if replan:
                section = self.plan_i_section(0)
                self._point_plan, self._point_index = self.section_map_object_selection(
                    self.sections[section].objects, self.get_properties(section), SectionMapObjectTypes.Region,
                    self.init_region_closest()[1], self._region_end_position), 0

    def replan_point(self):
        if not self.replan_point_check and len(self._point_plan) > 0:
            # print("replan point")
            self.replan_point_check = True
            region, section = self.plan_i_region(0)
            current_region = self.sections[section].objects[region]
            replan, _ = self.replaning(SectionMapObjectTypes.Point, current_region.objects, self._point_plan,
                                       self.get_properties(section, region), self._iterations_point)
            if replan:
                self._point_plan, self._point_index = self.section_map_object_selection(current_region.objects,
                                                                                        self.get_properties(section,
                                                                                                            region),
                                                                                        SectionMapObjectTypes.Point,
                                                                                        self.init_point_closest()[1],
                                                                                        self._point_end_position), 0

    def replaning(self, section_map_object_type, objects, chosen_objects, properties, iteration):
        planing_alg = self._replaning_algorithms[self.get_selection_mode(section_map_object_type, ModeTypes.PLANING)]
        return planing_alg.replaning(objects, chosen_objects, properties, iteration)

    # planning
    def plan_i_section(self, i, save_section_variables=False):
        #print("plan " + str(i) + " section")
        self.replan_section()
        section_plan = self._section_plan
        index = self._section_index + i
        if len(self._section_plan) == 0:
            start_section = self.init_section_closest()[1]
        else:
            start_section = self._section_plan[self._section_index]
        while index >= len(section_plan):
            s_plan = self.section_map_object_selection(self.sections, self.get_properties(),
                                                       SectionMapObjectTypes.Section, start_section, [])
            section_plan.extend(s_plan)
            start_section = section_plan[len(section_plan) - 1]
        if save_section_variables or len(self._section_plan) == 0:
            self._iterations_section += 1
            self.replan_section_check = False
            self._section_plan = section_plan
            self._section_index = index
        return section_plan[index]

    def plan_i_region(self, i, save_region_variables=False):
        #print("plan " + str(i) + " region")
        self.replan_region()
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
            section = self.plan_i_section(section_index + 1, save_region_variables)
            current_section = self.sections[section]
            next_section = self.sections[self.plan_i_section(section_index + 2)]
            region_end_position = self.find_end_points(current_section, next_section)
            if len(region_plan) > 0:
                region_start_position = self.find_start_point(current_section, next_section, region_plan[- 1])
            region_plan, region_index = self.section_map_object_selection(current_section.objects,
                                                                          self.get_properties(section),
                                                                          SectionMapObjectTypes.Region,
                                                                          region_start_position, region_end_position), 0
            section_index += 1
        if save_region_variables or len(self._region_plan) == 0:
            self._iterations_region += 1
            self.replan_region_check = False
            self._last_planned_section = section
            self._region_end_position = region_end_position
            self._region_plan = region_plan
            self._region_index = index
        return region_plan[index], section

    def plan_i_point(self, i, save_point_variables=False):
        #print("plan " + str(i) + " point")
        self.replan_point()
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
            region, section = self.plan_i_region(region_index + 1, save_point_variables)
            region2, section2 = self.plan_i_region(region_index + 2)
            current_region = self.sections[section].objects[region]
            next_region = self.sections[section2].objects[region2]
            point_end_position = self.find_end_points(current_region, next_region)
            point_plan, point_index = self.section_map_object_selection(
                current_region.objects,
                self.get_properties(self.plan_i_section(self._section_index), region),
                SectionMapObjectTypes.Point,
                point_start_position, point_end_position), 0
            point_start_position = self.find_start_point(current_region, next_region, point_plan[- 1])
            region_index += 1
        if save_point_variables or len(self._point_plan) == 0:
            self._iterations_point += 1
            self.replan_point_check = False
            self._last_planned_region = region_index
            self._point_plan = point_plan
            self._point_index = index
            self._point_end_position = point_end_position
        return point_plan[index], region, section

    def next_point(self, current_drone_position):
        current_drone_position = np.array(current_drone_position)
        self.update_drone_position(current_drone_position)
        save = False
        if len(self._point_plan) == 0:
            save = True
        point, region, section = self.plan_i_point(0, save)
        current_point = self.sections[section].objects[region].objects[point].data
        # (x position, y position, z position, vertical rotation of camera, yaw)
        if np.allclose(current_drone_position[np.r_[:3, 4:len(current_drone_position)]], current_point):
            point, region, section = self.plan_i_point(1, True)
            current_point = self.sections[section].objects[region].objects[point].data
        #print("section index", self._section_index)
        #print("region index", self._region_index)
        #print("point index", self._point_index)
        #print("section plan", self._section_plan)
        #print("region plan", self._region_plan)
        #print("point plan", self._point_plan)
        return np.insert(current_point, 3, 0)


class SectionAlgorithm(pm.PredictionAlgorithm):
    DRONE_POSITIONS = 1
    TARGET_POSITIONS = 1
    INPUT_TYPE = Constants.InputOutputParameterType.pose
    OUTPUT_TYPE = Constants.InputOutputParameterType.pose
    MAP_PRESENCE = Constants.MapPresenceParameter.yes

    def __init__(self):
        super(SectionAlgorithm, self).__init__()

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        drone_position = drone_positions[0]
        target_position = target_positions[0]

        # get time
        # get current position
        # from plan compute position in given time
        # return position
        p = Pose()
        return p
