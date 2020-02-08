#!/usr/bin/env python
import rospy
import numpy as np
import helper_pkg.utils as utils
from algorithms.drone_section import SectionsAlgrotihms
from algorithms.section_map_creator import SectionMap


def generate_probability_map(world_map):
    pr_map = np.zeros((world_map.width, world_map.height))
    samples_count = 10000
    x = np.random.uniform(-world_map.real_width / 2.0, world_map.real_width / 2.0, samples_count)
    y = np.random.uniform(- world_map.real_height / 2.0, world_map.real_height / 2.0, samples_count)
    map_x, map_y = world_map.get_index_on_map(x, y)
    values, counts = np.unique(zip(map_x, map_y), return_counts=True, axis=0)
    pr_map[values[:, 0], values[:, 1]] += counts
    pr_map /= samples_count
    return pr_map

def generate_point(world_map, returned_point):
    if np.random.rand() < 0.3:
        return returned_point
    x = np.random.uniform(-world_map.real_width / 2.0, world_map.real_width / 2.0, 1)[0]
    y = np.random.uniform(- world_map.real_height / 2.0, world_map.real_height / 2.0, 1)[0]
    z = np.random.uniform(3, 15, 1)[0]
    v_camera = np.random.uniform(0, np.pi/3, 1)[0]
    yaw = np.random.uniform(0, 2*np.pi, 1)[0]
    return [x, y, z, 0, v_camera, yaw]



if __name__ == '__main__':
    environment_configuration = rospy.get_param("environment_configuration")
    map_file = environment_configuration["map"]["obstacles_file"]
    world_map = utils.Map.map_from_file(map_file)
    section_file = environment_configuration["map"]["section_file"]
    sections = SectionMap.unpickle_sections(section_file)
    SectionMap.show_sections(sections)
    planner = SectionsAlgrotihms(sections, world_map)
    # map update
    pr_map = generate_probability_map(world_map)
    planner.update_probability_map(pr_map)
    planner.update_scores()
    SectionMap.section_score(sections)
    SectionMap.print_orientation_points(sections, world_map)
    point = [0, 0, 0.2, 0, 0, 0]
    for i in xrange(10):
        print("==========" + str(i) + "=================")
        returned_point = planner.next_point(point)
        point = generate_point(world_map, returned_point)
        print("POINT",  returned_point)
        if i % 4 == 0:
            pr_map = generate_probability_map(world_map)
            planner.update_probability_map(pr_map)
            planner.update_scores()



