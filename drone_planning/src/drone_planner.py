#!/usr/bin/env python
import rospy
import numpy as np
import helper_pkg.utils as utils
from algorithms.drone_section import SectionsAlgrotihms
from algorithms.section_map_creator import SectionMap
import cv2

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

def admissible_map(map_file):
    map_file = environment_configuration["map"]["obstacles_file"]
    world_map = utils.Map.map_from_file(map_file, 10)
    imgray = world_map.target_obstacle_map.astype(np.uint8)
    im2, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros((world_map.target_obstacle_map.shape[0], world_map.target_obstacle_map.shape[0], 3),
                   dtype=np.uint8)
    mask = np.zeros(imgray.shape, np.uint8)
    free_places = 0
    for contour in contours:
        mask[...] = 0
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean = cv2.mean(imgray, mask)[0]
        if mean > 0.5:
            cv2.drawContours(img, [contour], -1, np.random.randint(0, 255, 3), thickness=cv2.FILLED)
            free_places += 1
        else:
            cv2.drawContours(img, [contour], -1, np.random.randint(0, 1, 3), thickness=cv2.FILLED)
    cp = world_map.target_obstacle_map.astype(np.uint8)
    cp[cp > 0] = 255
    print(free_places)
    cv2.imwrite("orig_image.png", cp)
    cv2.imwrite("image.png", img)

def planning(map_file):
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
    logic_configuration = rospy.get_param("logic_configuration")
    SectionMap.print_orientation_points(sections, world_map, logic_configuration)
    point = [0, 0, 0.2, 0, 0, 0]
    for i in xrange(10):
        print("==========" + str(i) + "=================")
        returned_point = plannvisibility_file_pather.next_point(point)
        point = generate_point(world_map, returned_point)
        print("POINT", returned_point)
        if i % 4 == 0:
            pr_map = generate_probability_map(world_map)
            planner.update_probability_map(pr_map)
            planner.update_scores()

if __name__ == '__main__':
    environment_configuration = rospy.get_param("environment_configuration")
    map_file = environment_configuration["map"]["obstacles_file"]
    admissible_map(map_file)



