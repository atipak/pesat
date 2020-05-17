import pickle as pck
import numpy as np
import json
import sys
from shapely.geometry import Point





class SectionUtils():
    @staticmethod
    def unjson_sections(section_file):
        with open(section_file, "r") as fp:  # Unpickling
            dic =  json.load(fp)
            return SectionUtils.deserialize_dict(dic)

    @staticmethod
    def deserilize(d):
        obj = SectionMapObject()
        obj.object_id = d["object_id"]
        obj.score = d["score"]
        obj.entropy = d["entropy"]
        obj.neighbors = SectionUtils.deserialize_dict(d['neighbors'])
        obj.centroid = np.array(d["centroid"])
        obj.objects = SectionUtils.deserialize_dict(d['objects'])
        obj.type = d["type"]
        if d["data"] is None:
            obj.data = None
        else:
            obj.data = np.array(d["data"])
        obj.visibility = d["visibility"]
        obj.maximal_entropy = d["maximal_entropy"]
        obj.maximal_time = d["maximal_time"]
        return obj

    @staticmethod
    def deserialize_dict(d):
        new_d = {}
        for key in d:
            resolved_key = SectionUtils.resolve(key)
            new_d[resolved_key] = SectionUtils.resolve(d[key])
        return new_d


    @staticmethod
    def resolve(e):
        if isinstance(e, dict):
            try:
                return SectionUtils.deserilize(e)
            except:
                return SectionUtils.deserialize_dict(e)
        elif isinstance(e, int) or isinstance(e, float):
            return e
        elif isinstance(e, list):
            return SectionUtils.deserialize_list(e)
        else:
            if (sys.version_info > (3, 0)):
                # Python 3 code in this block
                string_test = str
            else:
                string_test = (str, unicode)
            if isinstance(e, string_test):
                try:
                    return int(e)
                except:
                    return str(e)

    @staticmethod
    def deserialize_list(l):
        new_l = []
        only_numbers = True
        for e in l:
            resolved_e = SectionUtils.resolve(e)
            if not isinstance(resolved_e, int) and not isinstance(resolved_e, float):
                only_numbers = False
            new_l.append(resolved_e)
        if only_numbers:
            new_l = np.array(new_l)
        return new_l

    @staticmethod
    def serialize(obj):
        """JSON serializer for objects not serializable by default json code"""

        if isinstance(obj, np.ndarray):
            serial = obj.tolist()
        elif isinstance(obj, Point):
            serial = SectionUtils.serialize(np.array(obj))
        elif isinstance(obj, np.int32):
            serial = int(obj)

        elif isinstance(obj, np.float32):
            serial = float(obj)

        elif isinstance(obj, SectionMapObject):
            serial = obj.__dict__
        else:
            serial = obj.__dict__
        return serial

    @staticmethod
    def convert_pickle_file_into_json(section_pickle_file, section_json_file):
        sections = SectionUtils.unpickle_sections(section_pickle_file)
        with open(section_json_file, "w") as fp:  # Unpickling
            return json.dump(sections, fp, indent=4, default=SectionUtils.serialize)

    @staticmethod
    def unpickle_sections(section_file):
        with open(section_file, "rb") as fp:  # Unpickling
            return pck.load(fp)

    @staticmethod
    def get_all_points(sections):
        points = []
        points_ids = []
        for (_, section) in sections.items():
            for region_index in section.objects:
                region = section.objects[region_index]
                for point_index in region.objects:
                    point = region.objects[point_index]
                    points.append(np.array(point.centroid))
                    points_ids.append(point.object_id)
        return points, points_ids

    @staticmethod
    def remove_neighbours(sections):
        for (_, section) in sections.items():
            section.neighbors = {}
            for region_index in section.objects:
                region = section.objects[region_index]
                region.neighbors = {}
                for point_index in region.objects:
                    point = region.objects[point_index]
                    point.neighbors = {}
        return sections


    @staticmethod
    def show_sections(sections, with_neighbours=True):
        print("SECTIONS")
        for (_, section) in sections.items():
            print("-section " + str(section.object_id))
            if with_neighbours:
                print("    NEIGHBORS")
                for neighbor in section.neighbors:
                    print("    -neighbor " + str(neighbor) + " : " + str(section.neighbors[neighbor]))
            print("    REGIONS")
            for region_index in section.objects:
                region = section.objects[region_index]
                print("    -region " + str(region.object_id))
                if with_neighbours:
                    print("       NEIGHBORS")
                    for neighbor in region.neighbors:
                        print("       -neighbor " + str(neighbor) + " : " + str(region.neighbors[neighbor]))
                print("       POINTS")
                for point_index in region.objects:
                    point = region.objects[point_index]
                    print("       -point " + str(point.object_id))
                    if with_neighbours:
                        print("          NEIGHBORS")
                        for neighbor in point.neighbors:
                            print("          -neighbor " + str(neighbor) + " : " + str(point.neighbors[neighbor]))

    @staticmethod
    def section_score(sections):
        print("SECTIONS")
        for (_, section) in sections.items():
            print("-section " + str(section.object_id) + "-- score: " + str(section.score) + ", entropy " + str(
                section.entropy) + ", visibility: " + str(section.visibility))
            print("    REGIONS")
            for region_index in section.objects:
                region = section.objects[region_index]
                print("    -region " + str(region.object_id) + "-- score: " + str(region.score) + ", entropy " + str(
                    region.entropy) + ", visibility: " + str(region.visibility))
                print("       POINTS")
                for point_index in region.objects:
                    point = region.objects[point_index]
                    print(
                        "       -point " + str(point.object_id) + "-- score: " + str(point.score) + ", entropy: " + str(
                            point.entropy) + ", visibility: " + str(point.visibility))


class SectionMapObjectTypes:
    Section = 0
    Region = 1
    Point = 2

    NAMES = ["Section", "Region", "Point"]

    @staticmethod
    def names(object_type):
        return SectionMapObjectTypes.NAMES[object_type]


class SectionMapObject(object):
    def __init__(self):
        super(SectionMapObject, self).__init__()
        self.object_id = 0
        self.score = 0
        self.entropy = 0
        self.neighbors = {}
        self.centroid = [0, 0]
        self.objects = {}
        self.type = SectionMapObjectTypes.Section
        self.data = None
        self.visibility = 0
        self.maximal_entropy = 0
        self.maximal_time = 0

