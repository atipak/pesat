#!/usr/bin/env python
import rospy
from algorithms.section_map_creator import SectionMap
import pickle


if __name__ == '__main__':
    environment_configuration = rospy.get_param("environment_configuration")
    map_file = environment_configuration["map"]["obstacles_file"]
    section_file = environment_configuration["map"]["section_file"]
    section_map_creator = SectionMap()
    section_objects = section_map_creator.create_sections_regions_and_points(map_file)
    with open(section_file, "wb") as fp:  # Pickling
        pickle.dump(section_objects, fp)

