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
from ordering_functions import DirectPath, NeighboursPath, DirectPathPlanning, NeighboursPathPlanning
from selection_functions import RatioElements, NElements, AboveAverage, BigDrop, AboveMinimum
from fitness_functions import DistanceScore, Neighbour, Maximum, Median
from section_map_creator import SectionMapObjectTypes


class ModeTypes:
    FITNESS = 0
    SELECTION = 1
    ORDERING = 2
    PLANING = 3


class SectionsAlgrotihms(object):
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
        self._selection_algorithms = [BigDrop(), AboveAverage(), NElements(int(len(sections) / 4)), RatioElements(0.2),
                                      AboveMinimum(0.05)]
        self._extended_selection_algorithms = [NeighboursPath(), DirectPath(), DirectPathPlanning,
                                               NeighboursPathPlanning]
        self._replaning_algorithms = [NNeighboursReplaning(3), NAllReplaning(3), NNeighboursMAllReplaning(3, 5),
                                      NoReplaning()]
        self._modes = [(0, 0, 1, 0), (1, 4, 0, 0)]
        self._probability_map = None
        self.coordinates = self.create_coordinates(sections)
        self.properties = self.create_graphs()
        self.drone_last_position = None
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        self._region_closest = (None, drone_region)
        self._region_end_position = None
        self._point_closest = (None, drone_point)
        self._point_end_position = None
        self._last_drone_coordinates = None
        self._objects_modes = {SectionMapObjectTypes.Section: self.get_section_mode,
                               SectionMapObjectTypes.Region: self.get_region_mode(),
                               SectionMapObjectTypes.Point: self.get_point_mode()}

    def get_drone_section_region_point(self):
        if self.drone_last_position is not None:
            map_point = self.world_map.map_point_from_real_coordinates(self.drone_last_position[0],
                                                                       self.drone_last_position[1])
            if map_point.x in self.coordinates and map_point.y in self.coordinates[map_point.x]:
                coor = self.coordinates[self.drone_last_position[0]][self.drone_last_position[1]]
                self._last_drone_coordinates = coor[0]
            else:
                if self._last_drone_coordinates is None:
                    idx = (np.abs(np.array([key for key in self.coordinates]) - map_point.x)).argmin()
                    idy = (np.abs(np.array([key for key in self.coordinates[idx]]) - map_point.y)).argmin()
                    self._last_drone_coordinates = self.coordinates[idx][idy][0]
            return self._last_drone_coordinates
        else:
            return [None, None, None]

    def init_region_closest(self):
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        return (None, drone_region)

    def init_point_closest(self):
        drone_section, drone_region, drone_point = self.get_drone_section_region_point()
        return (None, drone_point)

    def create_coordinates(self, sections):
        coordinates = {}
        for section_index in range(len(sections)):
            section = sections[section_index]
            for region_index in range(len(section.objects)):
                region = section.objects[region_index]
                for point_index in range(len(region.objects)):
                    point = region.objects[point_index]
                    for coor in point.objects:
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
            for section_index in sections:
                section = self.sections[section_index]
                section.score = 0
                section.entropy = 0
                for region_index in range(len(section.objects)):
                    region = section.objects[region_index]
                    region.score = 0
                    region.entropy = 0
                    for point_index in range(len(region.objects)):
                        point = region.objects[point_index]
                        point.score = 0
                        point.entropy = 0
                        for coor in point.objects:
                            point.score += self._probability_map[coor[0], coor[1]]
                            point.entropy += np.log2(self._probability_map[coor[0], coor[1]]) * self._probability_map[
                                coor[0], coor[1]]
                        region.score += point.score
                        region.entropy += point.entropy
                    section.score += region.score
                    section.entropy += region.entropy
        else:
            for section_index in range(len(self.sections)):
                section = self.sections[section_index]
                for region_index in range(len(section.objects)):
                    region = section.objects[region_index]
                    for point_index in range(len(region.objects)):
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
                        en = np.log2(self._probability_map[coor_x, coor_y]) * self._probability_map[coor_x, coor_y]
                        self.sections[point[0]].objects[point[1]].objects[point[2]].entropy += en
                        self.sections[point[0]].objects[point[1]].entropy += en
                        self.sections[point[0]].entropy += en

    def create_graphs(self):
        sections_edges = np.zeros((len(self.sections), len(self.sections)))
        regions_edges = []
        point_edges = []
        for section_index in range(len(self.sections)):
            section = self.sections[section_index]
            for section_index2 in range(len(self.sections)):
                section2 = self.sections[section_index2]
                distance = utils.Math.euclidian_distance(utils.Math.Point(*section.centroid),
                                                         utils.Math.Point(*section2.centroid))
                sections_edges[section_index, section_index2] = distance
            r_edges = np.zeros((len(section.objects), len(section.objects)))
            point_edges.append([])
            for obj_index in range(len(section.objects)):
                region = section.objects[obj_index]
                for obj_index2 in range(len(section.objects)):
                    region2 = section.objects[obj_index2]
                    distance = utils.Math.euclidian_distance(utils.Math.Point(*region.centroid),
                                                             utils.Math.Point(*region2.centroid))
                    r_edges[obj_index, obj_index2] = distance
                p_edges = np.zeros((len(region.objects), len(region.objects)))
                for point_index in range(len(region.objects)):
                    point = section.objects[point_index]
                    for point_index2 in range(len(region.objects)):
                        point2 = section.objects[point_index2]
                        distance = utils.Math.euclidian_distance(utils.Math.Point(*point.centroid),
                                                                 utils.Math.Point(*point2.centroid))
                        p_edges[point_index, point_index2] = distance
                point_edges[section_index].append(p_edges)
            regions_edges.append(r_edges)
        return {"sections": sections_edges, "regions": regions_edges, "points": point_edges}

    def find_closest(self, object1, object2):
        centroids1 = np.array([o.centroid for o in object1.objects])
        centroids2 = np.array([o.centroid for o in object2.objects])
        dist = distance.cdist(centroids1, centroids2, 'euclidean')
        m = np.max(dist[0, :])
        m_index = (0, 0)
        for index in range(len(centroids1)):
            minimum = np.min(dist[index, np.r_[0:index, index + 1:len(centroids2)]])
            if minimum < m:
                m = minimum
                m_index = (index, np.argmin(dist[index, np.r_[0:index, index + 1:len(centroids2)]]))
        return m_index

    def find_closest_for_centroid(self, centroid, next_object):
        centroids1 = np.array([o.centroid for o in next_object.objects])
        dist = distance.cdist(np.array([centroid]), centroids1, 'euclidean')
        return np.argmin(dist)

    def find_end_points(self, map_object, next_map_object):
        if next_map_object.object_id in map_object.neigbours:
            return [item[0] for item in map_object.neigbours[next_map_object.object_id]]
        else:
            return [self.find_closest(map_object, next_map_object)[0]]

    def find_start_point(self, map_object, next_map_object, end_point):
        if next_map_object.object_id in map_object.neigbours:
            for item in map_object.neigbours[next_map_object.object_id]:
                if end_point == item[0]:
                    return item[1]
        else:
            return self.find_closest_for_centroid(map_object.objects[end_point].centroid, next_map_object)

    def update_probability_map(self, target_positions):
        # TODO create probability map
        self._probability_map = ...

    def update_drone_position(self, drone_positions):
        # TODO create drone position
        self.drone_last_position = ...

    def section_map_object_selection(self, objects, properties, object_type, start_position, end_position):
        scored_objects = self._score_algorithms[self.get_selection_mode(object_type, ModeTypes.FITNESS)].score(objects,
                                                                                                               properties)
        selected_objects = self._selection_algorithms[self.get_selection_mode(object_type, ModeTypes.SELECTION)].select(
            objects, properties, scored_objects)
        # TODO decide algortihm by end_position parameter
        exteded_objects = self._extended_selection_algorithms[
            self.get_selection_mode(object_type, ModeTypes.ORDERING)].extended_select(
            selected_objects, scored_objects, objects, properties, start_position, end_position)
        return exteded_objects

    def replan_section(self):
        replan, sections = self._replaning_algorithms[
            self.get_selection_mode(SectionMapObjectTypes.Section, ModeTypes.PLANING)]
        if replan:
            self.update_scores(sections)
            drone_section, _, _ = self.get_drone_section_region_point()
            self._section_plan, self._section_index = self.section_map_object_selection(self.sections,
                                                                                        self.properties["sections"],
                                                                                        SectionMapObjectTypes.Section,
                                                                                        drone_section,
                                                                                        None), 0

    def replan_region(self):
        replan, _ = self._replaning_algorithms[self.get_selection_mode(SectionMapObjectTypes.Region, ModeTypes.PLANING)]
        if replan:
            section = self.plan_i_section(0)
            self._point_plan, self._point_index = self.section_map_object_selection(
                self.sections[section].objects, self.properties["regions"][section], SectionMapObjectTypes.Region,
                self.init_region_closest()[1], self._region_end_position), 0

    def replan_point(self):
        replan, _ = self._replaning_algorithms[self.get_selection_mode(SectionMapObjectTypes.Point, ModeTypes.PLANING)]
        if replan:
            region, section = self.plan_i_region(0)
            current_region = self.sections[section].objects[region]
            self._point_plan, self._point_index = self.section_map_object_selection(current_region.objects,
                                                                                    self.properties["points"][section][
                                                                                        region],
                                                                                    SectionMapObjectTypes.Point,
                                                                                    self.init_point_closest()[1],
                                                                                    self._point_end_position), 0

    def get_selection_mode(self, section_object_type, mode_type):
        return self._modes[self._objects_modes[section_object_type]()][mode_type]

    def get_section_mode(self):
        return 0

    def get_region_mode(self):
        return 1

    def get_point_mode(self):
        return 1

    def plan_i_section(self, i, save_section_variables=False):
        if self._section_index + i >= len(self._section_plan):
            self._section_plan, section_index = self.section_map_object_selection(self.sections,
                                                                                  self.properties["sections"],
                                                                                  SectionMapObjectTypes.Section,
                                                                                  self._section_plan[
                                                                                      self._section_index],
                                                                                  None), 0
        if self._section_index + i >= len(self._section_plan):
            return None
        if save_section_variables:
            self._section_index += i
        return self._section_plan[self._section_index + i]

    def plan_i_region(self, i, save_region_variables=False):
        index = self._region_index + i
        region_plan = self._region_plan
        region_start_position = self.init_region_closest()[1]
        region_end_position = self._region_end_position
        section_index = 0
        section = self.plan_i_section(0)
        while index >= len(region_plan):
            index -= len(region_plan)
            section = self.plan_i_section(section_index + 1, save_region_variables)
            current_section = self.sections[section]
            next_section = self.sections[self.plan_i_section(section_index + 2)]
            region_end_position = self.find_end_points(current_section, next_section)
            region_plan, region_index = self.section_map_object_selection(current_section.objects,
                                                                          self.properties["regions"][section],
                                                                          SectionMapObjectTypes.Region,
                                                                          region_start_position, region_end_position), 0
            region_start_position = self.find_start_point(current_section, next_section, region_plan[- 1])
            section_index += 1
        if save_region_variables:
            self._region_end_position = region_end_position
            self._region_plan = region_plan
            self._region_index = index
        return region_plan[index], section

    def plan_i_point(self, i, save_point_variables=False):
        index = self._point_index + i
        point_plan = self._point_plan
        point_start_position = self.init_point_closest()[1]
        point_end_position = self._point_end_position
        region, section = self.plan_i_region(0)
        region_index = 0
        section_index = 0
        while index >= len(point_plan):
            index -= len(point_plan)
            region, section = self.plan_i_region(region_index + 1, save_point_variables)
            region2, section2 = self.plan_i_region(region_index + 2)
            current_region = self.sections[section].objects[region]
            next_region = self.sections[section2].objects[region2]
            point_end_position = self.find_end_points(current_region, next_region)
            point_plan, point_index = self.section_map_object_selection(
                current_region.objects,
                self.properties["points"][self.plan_i_section(section_index + self._section_index)][region],
                SectionMapObjectTypes.Point,
                point_start_position, point_end_position), 0
            point_start_position = self.find_start_point(current_region, next_region, point_plan[- 1])
            region_index += 1
        if save_point_variables:
            self._point_plan = point_plan
            self._point_index = index
            self._point_end_position = point_end_position
        return point_plan[index], region, section

    def next_point(self, current_drone_position):
        point, region, section = self.plan_i_point(0)
        current_point = self.sections[section].objects[region].objects[point].data
        # (x position, y position, z position, vertical rotation of camera, yaw)
        if np.allclose(current_drone_position[np.r_[:3, 4:]], current_point):
            point, region, section = self.plan_i_point(1, True)
            current_point = self.sections[section].objects[region].objects[point].data
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
