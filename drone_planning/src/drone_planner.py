#!/usr/bin/env python
import rospy
import numpy as np
import helper_pkg.utils as utils
from algorithms.drone_section import SectionsAlgrotihms
from algorithms.section_map_creator import SectionMap

if __name__ == '__main__':
    environment_configuration = rospy.get_param("environment_configuration")
    map_file = environment_configuration["map"]["obstacles_file"]
    world_map = utils.Map.map_from_file(map_file)
    section_file = environment_configuration["map"]["section_file"]
    sections = SectionMap.unpickle_sections(section_file)
    SectionMap.show_sections(sections)
    planner = SectionsAlgrotihms(sections, world_map)
    # map update
    pr_map = np.zeros((world_map.width, world_map.height))
    samples_count = 10000
    x = np.random.uniform(0, world_map.real_width, samples_count)
    y = np.random.uniform(0, world_map.real_height, samples_count)
    map_x, map_y = world_map.get_index_on_map(x, y)
    pr_map[map_y, map_x] += 1
    pr_map /= samples_count
    planner.update_probability_map(pr_map)
    planner.update_scores()
    point = [0, 0, 0.2, 0, 0, 0]
    for i in xrange(10):
        print("==========" + str(i) + "=================")
        point = planner.next_point(point)
        print("POINT",  point)
