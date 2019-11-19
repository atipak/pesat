import json
from collections import namedtuple
import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors

box = namedtuple("box", ["center_x", "center_y", "width", "length", "height", "orientation"])
loaded_box = None
loaded_skeleton = None
loaded_ogre_script = None
loaded_script = None
loaded_color = None
loaded_face = None
loaded_face_box = None


class Building(object):

    def __init__(self):
        brick_height = 0.075  # mm
        self.texture_names = ["grey_bricks.png", "red_bricks.png"]
        bricks_count_in_column = [41, 16]
        images_pixels_width = [1233, 1024]
        images_pixels_heigth = [3600, 1024]
        self.textures_widths = []
        self.textures_heights = []
        for count in bricks_count_in_column:
            self.textures_heights.append(count * brick_height)
        for i in range(len(images_pixels_width)):
            self.textures_widths.append((self.textures_heights[i] / images_pixels_heigth[i]) * images_pixels_width[i])
        path = os.path.realpath(__file__)
        path = os.path.dirname(path)
        src_path = path + '/../../../'
        texts_path = src_path + "pesat_resources/texts/"
        self.plane_file_path = texts_path + "box_face"
        self.static_box_file_path = texts_path + "static_box"
        self.script_file_path = texts_path + "script_block"
        self.ogre_script_file_path = texts_path + "ogre_script"
        self.color_file_path = texts_path + "color_block"
        self.box_path = texts_path + "box"
        self.skeleton_path = texts_path + "skeleton"
        self.world_path = src_path + "pesat_resources/worlds/"
        self.textures_path = src_path + "pesat_resources/textures"
        self.maximal_iteration = 10000

    @classmethod
    def grid_map(cls, vertical_boxes_count=6, horizontal_boxes_count=6, world_name="grid_box_world"):
        world_file_name = world_name + ".world"
        if vertical_boxes_count % 2 == 1:
            print("Count of vertical count of boxes hat to be even number. Set to default 6")
            vertical_boxes_count = 6
        if horizontal_boxes_count % 2 == 1:
            print("Count of horizontal count of boxes hat to be even number. Set to default 6")
            horizontal_boxes_count = 6
        width = horizontal_boxes_count * 3 + (horizontal_boxes_count + 1) * 2
        height = vertical_boxes_count * 3 + (vertical_boxes_count + 1) * 2
        scale = 1
        b = Building()
        arcs = b.get_arcs(3, 3, 3, 3, 3, 3, 0, 0)
        start_horizontal_coor = -1 * (3 * horizontal_boxes_count / 2 + 1 + (horizontal_boxes_count / 2 - 1) * 2) + 1.5
        start_vertical_coor = -1 * (3 * vertical_boxes_count / 2 + 1 + (vertical_boxes_count / 2 - 1) * 2) + 1.5
        objects = []
        built_up = (3 * 3 * horizontal_boxes_count * vertical_boxes_count) / (width * height)
        for vertical in range(vertical_boxes_count):
            for horizontal in range(horizontal_boxes_count):
                original_box = b.get_i(arcs)
                alternated_box = [box(start_horizontal_coor, start_vertical_coor, original_box[0].width,
                                      original_box[0].length, original_box[0].height, original_box[0].orientation)]
                start_horizontal_coor += 3 + 2
                objects.append(alternated_box)
            start_vertical_coor += 3 + 2
            start_horizontal_coor = -1 * (
                    3 * horizontal_boxes_count / 2 + 1 + (horizontal_boxes_count / 2 - 1) * 2) + 1.5
        b.prepare_directories(world_name)
        boxes = b.create_world_file(world_file_name, objects, scale, width, height)
        world_info = b.create_world_info(world_name, width, height, built_up, 3, 0, 3, 0, 3, 0, 0, 0)
        json_text = {"world": world_info, "objects": boxes}
        map = Building.create_map_from_objects(objects, width, height, 20, 2)
        b.create_json_file(world_name, json_text)
        b.create_map_file(world_name, map, width, height)

    @classmethod
    def generate_random_map(cls, world_name="box_world", width=10, height=10, built_up=0.6,
                            maximal_width=20.0, minimal_width=2, maximal_height=20.0, minimal_height=2,
                            maximal_orientation=1.54, minimal_orientation=0, map_resolution=100, drone_size=1.0,
                            drone_start_position=[0, 0], target_size=1.5, target_start_position=[5, 0]):
        pixel_width = width * map_resolution
        pixel_height = height * map_resolution
        pixels_count = pixel_width * pixel_height
        map = np.zeros((pixel_width, pixel_height), dtype=np.uint8)
        image = np.zeros((pixel_width, pixel_height), dtype=np.uint8)
        # insert places for target and drone
        drone_position = [
            box(drone_start_position[0] + width / 2, drone_start_position[1] + height / 2, drone_size, drone_size,
                drone_size, 0)]
        target_position = [
            box(target_start_position[0] + width / 2, target_start_position[1] + height / 2, target_size, target_size,
                target_size, 0)]
        drone_position_points = (Building.split_to_points(drone_position) / (1.0 / map_resolution)).astype('int64')
        target_position_points = (Building.split_to_points(target_position) / (1.0 / map_resolution)).astype('int64')
        cv2.fillConvexPoly(image, drone_position_points[0], 255)
        cv2.fillConvexPoly(image, target_position_points[0], 255)

        objects = []
        building = Building()
        iteration = 0
        while iteration < building.maximal_iteration:
            iteration += 1
            shadow = np.zeros((pixel_width, pixel_height), dtype=np.uint8)
            shadow_map = np.zeros((pixel_width, pixel_height), dtype=np.uint8)
            center_x = float(np.random.uniform(0, width, 1)[0])
            center_y = float(np.random.uniform(0, height, 1)[0])
            object = building.random_object(maximal_height, minimal_height, maximal_width, minimal_width,
                                            maximal_width, minimal_width, maximal_orientation, minimal_orientation,
                                            center_x, center_y)
            points = Building.split_to_points(object)
            axis = Building.calculate_center_of_object(object)
            # all points inside map
            out = False
            for ps in points:
                for p in ps:
                    if p[0] < 0 or p[0] > width or p[1] < 0 or p[1] > height:
                        out = True
            if out:
                continue
            int_points = (points / (1.0 / map_resolution)).astype('int64')
            for i in range(len(points)):
                cv2.fillConvexPoly(shadow, int_points[i], 255)
                cv2.fillConvexPoly(shadow_map, int_points[i], 255)
            rotated_shifted_object = []
            centers = []
            for o in object:
                centers.append([o.center_x, o.center_y])
            new_centers = np.array(
                Building.shift_and_rotate_points(axis, centers, [-width / 2, -height / 2], object[0].orientation),
                dtype=np.float32)
            for i in range(len(centers)):
                rotated_shifted_object.append(
                    box(float(new_centers[i][0]), float(new_centers[i][1]), object[i].width, object[i].length,
                        object[i].height, object[i].orientation))
            if cv2.countNonZero(cv2.bitwise_and(image, shadow)) > 0:
                continue
            check_map = cv2.bitwise_or(map, shadow_map)
            if cv2.countNonZero(check_map) / pixels_count > built_up:
                continue
            objects.append(rotated_shifted_object)
            print(rotated_shifted_object)
            map = cv2.bitwise_or(map, shadow_map)
            image = cv2.bitwise_or(image, shadow)
        map = 255 - map
        building.prepare_directories(world_name)
        tringles, connectivity = building.triangulation_connectivity(objects,
                                                                     [box(0, 0, width, height, maximal_height, 0)])
        if False:
            tringles = np.array(tringles) / 50
            fig = plt.figure()
            ax = Axes3D(fig)
            for c in connectivity:
                verts = []
                for i in c:
                    verts.append(tuple(np.array(tringles[i]) - np.array([-0.5, -0.5, 0])))
                print(verts)
                d = Poly3DCollection([verts])
                d.set_color(colors.rgb2hex(np.random.rand(3)))
                ax.add_collection3d(d)
            plt.show()
            return
        boxes = building.create_world_file_from_planes(world_name, objects, width, height, maximal_height)
        world_info = building.create_world_info(world_name, width, height, built_up, maximal_width, minimal_width,
                                                maximal_width, minimal_width, maximal_height, minimal_height,
                                                maximal_orientation, minimal_orientation)
        print("Boxes count:", len(boxes))
        json_text = {"world": world_info, "objects": boxes, "mesh": {"triangles": tringles,
                                                                     "connectivity": connectivity}}
        building.create_json_file(world_name, json_text)
        building.create_map_file(world_name, map, pixel_width, pixel_height)
        return map

    def get_u(self, arcs):
        l = self.get_l(arcs)
        l_vertical = l[1]
        l_horizontal = l[0]
        i = self.get_i(arcs)[0]
        last_piece = box(l_horizontal.length - l_vertical.width / 2 - i.width / 2,
                         i.length / 2 - l_vertical.length / 2 + i.center_y, i.width, i.length, i.height, i.orientation)
        # l_vertical.height = np.average([l_vertical.height, l_horizontal.height, i.height])
        # l_horizontal.height = l_vertical.height
        # i.height = l_vertical.height
        return [l_vertical, l_horizontal, last_piece]

    def get_l(self, arcs):
        vertical = self.get_i(arcs)[0]
        h = self.get_i(arcs)[0]
        horizontal = box(h.length / 2 - vertical.width / 2, vertical.length / 2 + h.width / 2, h.width, h.length,
                         h.height, np.pi / 2.0)
        # vertical.height = np.average([vertical.height, horizontal.height])
        # horizontal.height = vertical.height
        return [horizontal, vertical]

    def get_i(self, arcs):
        side_one = arcs["min_length"] + np.random.rand() * (arcs["max_length"] - arcs["min_length"])
        side_two = arcs["min_width"] + np.random.rand() * (arcs["max_width"] - arcs["min_width"])
        height = arcs["min_height"] + np.random.rand() * (arcs["max_height"] - arcs["min_height"])
        orientation_difference = (arcs["max_orientation"] - arcs["min_orientation"]) / 2.0
        orientation_center = orientation_difference + arcs["min_orientation"]
        sign = np.random.choice([-1, 1], 1)[0]
        orientation = orientation_center + np.random.rand() * orientation_difference * sign
        # orientation = 1
        center_x = arcs["center_x"]
        center_y = arcs["center_y"]
        b = box(center_x, center_y, side_two, side_one, height, orientation)
        return [b]

    def get_arcs(self, max_height=0, min_height=0, max_length=0, min_length=0, max_width=0, min_width=0,
                 max_ori=0, min_ori=0, center_x=0, center_y=0):
        arcs = {"max_height": max_height,
                "min_height": min_height,
                "max_length": max_length,
                "min_length": min_length,
                "max_width": max_width,
                "min_width": min_width,
                "max_orientation": max_ori,
                "min_orientation": min_ori,
                "center_x": center_x,
                "center_y": center_y}
        return arcs

    def create_world_info(self, world_name="box_world", width=1000, height=1000, built_up=0.60,
                          maximal_width=20.0, minimal_width=0.2, maximal_length=20.0, minimal_length=0.2,
                          maximal_height=35.0, minimal_height=0.3, maximal_orientation=1.54, minimal_orientation=0):
        return {
            "world_name": world_name, "width": width, "height": height, "built_up": built_up,
            "maximal_width": maximal_width, "minimal_width": minimal_width, "maximal_height": maximal_height,
            "minimal_height": minimal_height, "maximal_orientation": maximal_orientation,
            "minimal_orientation": minimal_orientation, "maximal_length": maximal_length,
            "minimal_length": minimal_length
        }

    def random_object(self, max_height=0, min_height=0, max_length=0, min_length=0, max_width=0, min_width=0, max_ori=0,
                      min_ori=0, center_x=0, center_y=0):
        arcs = self.get_arcs(max_height, min_height, max_length, min_length, max_width, min_width, max_ori, min_ori,
                             center_x, center_y)
        object_function = np.random.choice([self.get_i, self.get_l, self.get_u])
        return self.get_i(arcs)

    @staticmethod
    def from_json_file_object_to_object(json_object):
        center_x = json_object["x_pose"]
        center_y = json_object["y_pose"]
        side_two = json_object["x_size"]
        side_one = json_object["y_size"]
        height = json_object["z_size"]
        orientation = json_object["y_orient"]
        b = box(center_x, center_y, side_two, side_one, height, orientation)
        return [b]

    @staticmethod
    def split_to_points(object, image_coordinates=False):
        points = []
        for part in object:
            ps = []
            if image_coordinates:
                part_length = part.length / 2 - 1
                part_width = part.width / 2 - 1
            else:
                part_length = part.length / 2
                part_width = part.width / 2
            # IMPORTANT because of exclusive coordinates added -1
            p1 = [part.center_x - part.width / 2, part.center_y - part.length / 2]
            p2 = [part.center_x - part.width / 2, part.center_y + part_length]
            p3 = [part.center_x + part_width, part.center_y + part_length]
            p4 = [part.center_x + part_width, part.center_y - part.length / 2]
            ps.append(p1)
            ps.append(p2)
            ps.append(p3)
            ps.append(p4)
            points.append(
                Building.shift_and_rotate_points((part.center_x, part.center_y), ps, (0, 0), part.orientation))
        return np.array(points)

    @staticmethod
    def shift_and_rotate_points(axis, points, shift, angle):
        changed_points = []
        s = np.sin(angle)
        c = np.cos(angle)
        for p in points:
            norm__p = [p[0] - axis[0], p[1] - axis[1]]
            rotated_p = [norm__p[0] * c - norm__p[1] * s, norm__p[0] * s + norm__p[1] * c]
            shifted_p = [rotated_p[0] + axis[0] + shift[0], rotated_p[1] + axis[1] + shift[1]]
            changed_points.append(shifted_p)
        return changed_points

    @staticmethod
    def calculate_center_of_object(object):
        x = 0
        y = 0
        for o in object:
            x += o.center_x
            y += o.center_y
        return [x / len(object), y / len(object)]

    def create_world_file(self, world_name, objects, scale, width, height):
        world_path = self.world_path + world_name
        index = 0
        boxes = []
        boxes_string = ""
        for object in objects:
            for o in object:
                h = o.height * 1 / 10
                z = h / 2
                red = np.random.rand()
                green = np.random.rand()
                blue = h / 255.0
                boxes_string += self.read_box("box_{}".format(str(index)),
                                              str(o.center_x * scale - width / 2),
                                              str(o.center_y * scale - height / 2),
                                              z, str(0.0), str(0.0), str(o.orientation), o.width * scale,
                                              o.length * scale,
                                              h,
                                              red,
                                              green, blue)
                index += 1
                b = {
                    "name": "box_{}".format(str(index)), "x_pose": o.center_x * scale, "y_pose": o.center_y * scale,
                    "z_pose": h / 2 + 0.2, "r_orient": 0.0, "p_orient": 0.0, "y_orient": o.orientation,
                    "x_size": o.width * scale, "y_size": o.length * scale, "z_size": h, "red": red, "green": green,
                    "blue": blue
                }
                boxes.append(b)
        skeleton = self.read_skeleton(max(width, height), boxes_string)
        with open(world_path, "w") as f:
            f.write(skeleton)
        return boxes

    def prepare_directories(self, world_name):
        world_path = self.world_path + world_name
        script_path = world_path + "/scripts"
        if not os.path.exists(script_path):
            os.makedirs(script_path)

    def triangulation_connectivity(self, objects, world_box):
        both_faces = False
        box_index = 0
        ps = []
        connectivity = []
        for o in objects:
            b_start = box_index * 8
            points = Building.split_to_points(o)[0]
            for j in [o[0].height, 0]:
                for i in range(4):
                    ps.append([points[i][0], points[i][1], j])
            # upper
            connectivity.append([b_start + 0, b_start + 1, b_start + 2])
            connectivity.append([b_start + 0, b_start + 2, b_start + 3])
            # upper opposite face
            if both_faces:
                connectivity.append([b_start + 2, b_start + 1, b_start + 0])
                connectivity.append([b_start + 3, b_start + 2, b_start + 0])
            # back
            connectivity.append([b_start + 7, b_start + 3, b_start + 2])
            connectivity.append([b_start + 6, b_start + 7, b_start + 2])
            # back opposite face
            if both_faces:
                connectivity.append([b_start + 2, b_start + 3, b_start + 7])
                connectivity.append([b_start + 2, b_start + 7, b_start + 6])
            # right
            connectivity.append([b_start + 6, b_start + 2, b_start + 1])
            connectivity.append([b_start + 5, b_start + 6, b_start + 1])
            # right opposite face
            if both_faces:
                connectivity.append([b_start + 1, b_start + 2, b_start + 6])
                connectivity.append([b_start + 1, b_start + 6, b_start + 5])
            # front
            connectivity.append([b_start + 4, b_start + 5, b_start + 1])
            connectivity.append([b_start + 4, b_start + 1, b_start + 0])
            # front opposite face
            if both_faces:
                connectivity.append([b_start + 1, b_start + 5, b_start + 4])
                connectivity.append([b_start + 0, b_start + 1, b_start + 4])
            # bottom
            connectivity.append([b_start + 6, b_start + 5, b_start + 4])
            connectivity.append([b_start + 6, b_start + 4, b_start + 7])
            # bottom opposite face
            if both_faces:
                connectivity.append([b_start + 4, b_start + 5, b_start + 6])
                connectivity.append([b_start + 7, b_start + 4, b_start + 6])
            # left
            connectivity.append([b_start + 4, b_start + 0, b_start + 3])
            connectivity.append([b_start + 4, b_start + 3, b_start + 7])
            # left opposite face
            if both_faces:
                connectivity.append([b_start + 3, b_start + 0, b_start + 4])
                connectivity.append([b_start + 7, b_start + 3, b_start + 4])
            box_index += 1
        # world box
        b_start = box_index * 8
        points = Building.split_to_points(world_box)[0]
        for j in [world_box[0].height, 0]:
            for i in range(4):
                ps.append([points[i][0], points[i][1], j])
        # back
        connectivity.append([b_start + 7, b_start + 3, b_start + 2])
        connectivity.append([b_start + 6, b_start + 7, b_start + 2])
        # back opposite face
        if both_faces:
            connectivity.append([b_start + 2, b_start + 3, b_start + 7])
            connectivity.append([b_start + 2, b_start + 7, b_start + 6])
        # right
        connectivity.append([b_start + 6, b_start + 2, b_start + 1])
        connectivity.append([b_start + 5, b_start + 6, b_start + 1])
        # right opposite face
        if both_faces:
            connectivity.append([b_start + 1, b_start + 2, b_start + 6])
            connectivity.append([b_start + 1, b_start + 6, b_start + 5])
        # front
        connectivity.append([b_start + 4, b_start + 5, b_start + 1])
        connectivity.append([b_start + 4, b_start + 1, b_start + 0])
        # front opposite face
        if both_faces:
            connectivity.append([b_start + 1, b_start + 5, b_start + 4])
            connectivity.append([b_start + 0, b_start + 1, b_start + 4])
        # bottom
        connectivity.append([b_start + 6, b_start + 5, b_start + 4])
        connectivity.append([b_start + 6, b_start + 4, b_start + 7])
        # bottom opposite face
        if both_faces:
            connectivity.append([b_start + 4, b_start + 5, b_start + 6])
            connectivity.append([b_start + 7, b_start + 4, b_start + 6])
        # left
        connectivity.append([b_start + 4, b_start + 0, b_start + 3])
        connectivity.append([b_start + 4, b_start + 3, b_start + 7])
        # left opposite face
        if both_faces:
            connectivity.append([b_start + 3, b_start + 0, b_start + 4])
            connectivity.append([b_start + 7, b_start + 3, b_start + 4])
        box_index += 1
        return ps, connectivity

    def create_world_file_from_planes(self, world_name, objects, width, height, max_height):
        world_path = self.world_path + world_name
        world_file_name = world_path + "/" + world_name + ".world"
        scripts_path = world_path + "/scripts"
        ogre_script_file_name = scripts_path + "/" + world_name + ".material"
        index = 0
        boxes = []
        boxes_string = ""
        ogre_string = ""
        for object in objects:
            for o in object:
                box, ogre, b = self.generate_box_from_boxes(o, index, scripts_path, width, height)
                boxes_string += box
                ogre_string += ogre
                index += 1
                boxes.append(b)
        box, ogre = self.generate_outside_box_from_boxes(max_height, height, width, scripts_path)
        boxes_string += box
        ogre_string += ogre
        skeleton = self.read_skeleton(max(width, height), boxes_string)
        with open(world_file_name, "w") as f:
            f.write(skeleton)
        with open(ogre_script_file_name, "w") as f:
            f.write(ogre_string)
        return boxes

    def create_json_file(self, world_name, boxes):
        world_path = self.world_path + world_name
        json_file_name = world_path + "/" + world_name + ".json"
        with open(json_file_name, 'w') as outfile:
            json.dump(boxes, outfile, indent=4)

    def create_map_file(self, world_name, map, width, height):
        world_path = self.world_path + world_name
        map_file_name = world_path + "/" + world_name + ".bmp"
        original_path = world_path + "/origin.bmp"
        cv2.imwrite(original_path, map)
        M = cv2.getRotationMatrix2D((width / 2.0 - 0.5, height / 2.0 - 0.5), -90, 1)
        map = cv2.warpAffine(map, M, (width, height))
        # map = cv2.flip(map, 0 )
        cv2.imwrite(map_file_name, map)

    @staticmethod
    def create_map_from_objects(objects, world_width, world_height, world_altitude, resolution):
        width = world_width * resolution
        height = world_height * resolution
        height_map = np.zeros((width, height), dtype=np.uint8)
        depth_map = np.zeros((width, height), dtype=np.uint8)
        for object in objects:
            object = Building.from_json_file_object_to_object(object)
            points = Building.split_to_points(object)
            points = np.array(points)
            points /= (1.0 / resolution)
            points = points.astype(np.int32)
            points += [width / 2, height / 2]
            axis = Building.calculate_center_of_object(object)
            for i in range(len(points)):
                cv2.fillConvexPoly(height_map, points[i], int((object[0].height / float(world_altitude)) * 255))
                cv2.fillConvexPoly(depth_map, points[i], 0)
                # print(new_points)
        height_map = cv2.flip(height_map, 0)  # vertical flip
        depth_map = cv2.flip(depth_map, 0)  # vertical flip
        map = [height_map, depth_map]
        return map

    def read_skeleton(self, plane_size, boxes):
        global loaded_skeleton
        if loaded_skeleton is None:
            loaded_skeleton = ""
            with open(self.skeleton_path, "r") as file:
                s = file.readlines()
                for line in s:
                    loaded_skeleton += line
        return loaded_skeleton.format(plane_size=plane_size, boxes=boxes)

    def read_box(self, name, x_pose, y_pose, z_pose, r_orient, p_orient, y_orient, x_size, y_size,
                 z_size, red,
                 green, blue):
        global loaded_box
        if loaded_box is None:
            loaded_box = ""
            with open(self.box_path, "r") as file:
                s = file.readlines()
                for line in s:
                    loaded_box += line
        return loaded_box.format(name=name, x_pose=x_pose, y_pose=y_pose, z_pose=z_pose, r_orient=r_orient,
                                 p_orient=p_orient,
                                 y_orient=y_orient,
                                 x_size=x_size, y_size=y_size, z_size=z_size, red=red, green=green, blue=blue)

    def read_face(self, name, face, x_pose, y_pose, z_pose, r_orient, p_orient, y_orient, width_size,
                  length_size, material_block, normal_x, normal_y, normal_z):
        global loaded_face
        if loaded_face is None:
            loaded_face = ""
            with open(self.plane_file_path, "r") as file:
                s = file.readlines()
                for line in s:
                    loaded_face += line
        return loaded_face.format(name=name, x_pose=x_pose, y_pose=y_pose, z_pose=z_pose, r_orient=r_orient,
                                  p_orient=p_orient, y_orient=y_orient, face=face, plane_width=width_size,
                                  plane_length=length_size, material_block=material_block, normal_x=normal_x,
                                  normal_y=normal_y, normal_z=normal_z)

    def read_face_box(self, name, face, x_pose, y_pose, z_pose, r_orient, p_orient, y_orient, x_size,
                      y_size, z_size, material_block):
        global loaded_face_box
        if loaded_face_box is None:
            loaded_face_box = ""
            with open(self.static_box_file_path, "r") as file:
                s = file.readlines()
                for line in s:
                    loaded_face_box += line
        return loaded_face_box.format(name=name, x_pose=x_pose, y_pose=y_pose, z_pose=z_pose, r_orient=r_orient,
                                      p_orient=p_orient, y_orient=y_orient, face=face, x_size=x_size, y_size=y_size,
                                      z_size=z_size, material_block=material_block)

    def read_script_block(self, name, face, scripts_path):
        global loaded_script
        if loaded_script is None:
            loaded_script = ""
            with open(self.script_file_path, "r") as file:
                s = file.readlines()
                for line in s:
                    loaded_script += line
        return loaded_script.format(name=name, scripts_path=scripts_path, face=face,
                                    textures_path=self.textures_path)

    def read_color(self, red, green, blue):
        global loaded_color
        if loaded_color is None:
            loaded_color = ""
            with open(self.color_file_path, "r") as file:
                s = file.readlines()
                for line in s:
                    loaded_color += line
        return loaded_color.format(red=red, green=green, blue=blue)

    def read_ogre_script(self, name, face, scale_x, scale_y, texture_name):
        global loaded_ogre_script
        if loaded_ogre_script is None:
            loaded_ogre_script = ""
            with open(self.ogre_script_file_path, "r") as file:
                s = file.readlines()
                for line in s:
                    loaded_ogre_script += line
        return loaded_ogre_script.format(name=name, texture_name=texture_name, face=face, scale_x=scale_x,
                                         scale_y=scale_y)

    def generate_box_from_boxes(self, box, index, scripts_path, wm, hm):
        box_height = 0.1
        h = box.height
        z = h / 2
        red = np.random.rand()
        green = np.random.rand()
        blue = h / 255.0
        i = np.random.randint(0, len(self.texture_names))
        texture_name = self.texture_names[i]
        texture_height = self.textures_heights[i]
        texture_width = self.textures_widths[i]
        color_string_upper = self.read_color(red, green, blue)
        script_string_minors = self.read_script_block("box_{}".format(str(index)), "minors", scripts_path)
        script_string_majors = self.read_script_block("box_{}".format(str(index)), "majors", scripts_path)
        scale_x = 1 / box.length / texture_width
        scale_y = 1 / box.height / texture_height
        ogre_script_string_minors = self.read_ogre_script("box_{}".format(str(index)), "minors",
                                                          scale_x, scale_y, texture_name)
        scale_x = 1 / box.width / texture_width
        ogre_script_string_majors = self.read_ogre_script("box_{}".format(str(index)), "majors",
                                                          scale_x, scale_y, texture_name)
        axis = (box.center_x, box.center_y)
        shift = (0, 0)
        orientation = box.orientation
        new_center = Building.shift_and_rotate_points(axis, [(box.center_x, box.center_y)], shift, orientation)[0]
        upper_plane_string = self.read_face_box("box_{}".format(str(index)), "upper",
                                                new_center[0], new_center[1], h + (box_height / 2.0), str(0.0),
                                                str(0.0), str(box.orientation),
                                                box.width, box.length, box_height, color_string_upper)
        new_center = \
            Building.shift_and_rotate_points(axis, [(box.center_x, box.center_y - box.length / 2 - box_height / 2.0)],
                                             shift, orientation)[0]
        front_plane_string = self.read_face_box("box_{}".format(str(index)), "front", new_center[0],
                                                new_center[1], z, 0.0, str(0.0), str(box.orientation + np.pi / 2),
                                                box_height, box.width, h, script_string_majors)
        new_center = \
            Building.shift_and_rotate_points(axis, [(box.center_x, box.center_y + box.length / 2 + box_height / 2.0)],
                                             shift, orientation)[0]
        back_plane_string = self.read_face_box("box_{}".format(str(index)), "back",
                                               new_center[0], new_center[1], z, 0.0, str(0.0),
                                               str(box.orientation + np.pi / 2), box_height, box.width, h,
                                               script_string_majors)
        new_center = \
            Building.shift_and_rotate_points(axis, [(box.center_x - box.width / 2 - box_height / 2.0, box.center_y)],
                                             shift, orientation)[0]
        left_plane_string = self.read_face_box("box_{}".format(str(index)), "left", new_center[0], new_center[1], z,
                                               0.0, 0.0, str(box.orientation + np.pi / 2), box.length, box_height, h,
                                               script_string_minors)
        new_center = \
            Building.shift_and_rotate_points(axis, [(box.center_x + box.width / 2 + box_height / 2.0, box.center_y)],
                                             shift, orientation)[0]
        right_plane_string = self.read_face_box("box_{}".format(str(index)), "right", new_center[0], new_center[1], z,
                                                0.0, 0.0, str(box.orientation + np.pi / 2), box.length,
                                                box_height, h, script_string_minors)
        b = {
            "name": "box_{}".format(str(index)), "x_pose": box.center_x, "y_pose": box.center_y,
            "z_pose": h / 2 + 0.2, "r_orient": 0.0, "p_orient": 0.0, "y_orient": box.orientation,
            "x_size": box.width, "y_size": box.length, "z_size": h, "red": red, "green": green,
            "blue": blue, "texture_name": texture_name
        }
        box_string = upper_plane_string + front_plane_string + back_plane_string + left_plane_string + right_plane_string
        ogre_script = ogre_script_string_minors + ogre_script_string_majors
        return box_string, ogre_script, b

    def generate_box(self, box, index, scripts_path, map_width, map_height):
        h = box.height
        z = h / 2
        red = np.random.rand()
        green = np.random.rand()
        blue = h / 255.0
        i = np.random.randint(0, len(self.texture_names))
        texture_name = self.texture_names[i]
        texture_height = self.textures_heights[i]
        texture_width = self.textures_widths[i]
        color_string_upper = self.read_color(red, green, blue)
        script_string_minors = self.read_script_block("box_{}".format(str(index)), "minors", scripts_path)
        script_string_majors = self.read_script_block("box_{}".format(str(index)), "majors", scripts_path)
        scale_x = 1 / box.length / texture_width
        scale_y = 1 / box.height / texture_height
        ogre_script_string_minors = self.read_ogre_script("box_{}".format(str(index)), "minors",
                                                          scale_x, scale_y, texture_name)
        scale_x = 1 / box.width / texture_width
        ogre_script_string_majors = self.read_ogre_script("box_{}".format(str(index)), "majors",
                                                          scale_x, scale_y, texture_name)
        axis = (box.center_x, box.center_y)
        shift = (0, 0)
        orientation = box.orientation
        new_center = Building.shift_and_rotate_points(axis, [(box.center_x, box.center_y)], shift, orientation)[0]
        upper_plane_string = self.read_face("box_{}".format(str(index)), "upper",
                                            new_center[0], new_center[1], h, str(0.0),
                                            str(0.0), str(box.orientation),
                                            box.width, box.length, color_string_upper, 0, 0, 1)
        new_center = \
            Building.shift_and_rotate_points(axis, [(box.center_x, box.center_y - box.length / 2)], shift, orientation)[
                0]
        front_plane_string = self.read_face("box_{}".format(str(index)), "front", new_center[0],
                                            new_center[1], z,
                                            np.pi / 2, str(0.0), str(box.orientation), box.width, h,
                                            script_string_majors, 1, 0, 0)
        new_center = \
            Building.shift_and_rotate_points(axis, [(box.center_x, box.center_y + box.length / 2)], shift, orientation)[
                0]
        back_plane_string = self.read_face("box_{}".format(str(index)), "back",
                                           new_center[0], new_center[1], z, -np.pi / 2, str(0.0),
                                           str(box.orientation), box.width, h, script_string_majors, -1, 0, 0)
        new_center = \
            Building.shift_and_rotate_points(axis, [(box.center_x - box.width / 2, box.center_y)], shift, orientation)[
                0]
        left_plane_string = self.read_face("box_{}".format(str(index)), "left", new_center[0], new_center[1], z,
                                           np.pi / 2, 0.0, str(box.orientation - np.pi / 2), box.length, h,
                                           script_string_minors, 0, -1, 0)
        new_center = \
            Building.shift_and_rotate_points(axis, [(box.center_x + box.width / 2, box.center_y)], shift, orientation)[
                0]
        right_plane_string = self.read_face("box_{}".format(str(index)), "right", new_center[0], new_center[1], z,
                                            -np.pi / 2, 0.0, str(box.orientation - np.pi / 2), box.length, h,
                                            script_string_minors, 0, 1, 0)
        b = {
            "name": "box_{}".format(str(index)), "x_pose": box.center_x, "y_pose": box.center_y,
            "z_pose": h / 2 + 0.2, "r_orient": 0.0, "p_orient": 0.0, "y_orient": box.orientation,
            "x_size": box.width, "y_size": box.length, "z_size": h, "red": red, "green": green,
            "blue": blue, "texture_name": texture_name
        }
        box_string = upper_plane_string + front_plane_string + back_plane_string + left_plane_string + right_plane_string
        ogre_script = ogre_script_string_minors + ogre_script_string_majors
        return box_string, ogre_script, b

    def generate_outside_box_from_boxes(self, height, length, width, scripts_path):
        box_height = 0.1
        i = np.random.randint(0, len(self.texture_names))
        texture_name = self.texture_names[i]
        texture_height = self.textures_heights[i]
        texture_width = self.textures_widths[i]
        script_string_minors = self.read_script_block("world_side", "minors", scripts_path)
        script_string_majors = self.read_script_block("world_side", "majors", scripts_path)
        scale_x = 1 / length / texture_width
        scale_y = 1 / height / texture_height
        ogre_script_string_minors = self.read_ogre_script("world_side", "minors", scale_x, scale_y, texture_name)
        scale_x = 1 / width / texture_width
        ogre_script_string_majors = self.read_ogre_script("world_side", "majors", scale_x, scale_y, texture_name)
        front_plane_string = self.read_face_box("world_side", "front", length / 2, 0 + box_height / 2.0, height / 2,
                                                0.0, 0.0, 0.0, box_height, width, height,
                                                script_string_majors)
        back_plane_string = self.read_face_box("world_side", "back", -length / 2, 0 - box_height / 2.0, height / 2,
                                               0.0, 0.0, 0.0, box_height, width, height,
                                               script_string_majors)
        left_plane_string = self.read_face_box("world_side", "left", 0, -width / 2 - box_height / 2.0, height / 2,
                                               0.0, str(0.0), str(0.0), length, box_height, height,
                                               script_string_minors)
        right_plane_string = self.read_face_box("world_side", "right", 0, width / 2 + box_height / 2.0, height / 2,
                                                0.0, str(0.0), str(0.0), length, box_height, height,
                                                script_string_minors)
        box_string = front_plane_string + back_plane_string + left_plane_string + right_plane_string
        ogre_script = ogre_script_string_minors + ogre_script_string_majors
        return box_string, ogre_script

    def generate_outside_box(self, height, length, width, scripts_path):
        i = np.random.randint(0, len(self.texture_names))
        texture_name = self.texture_names[i]
        texture_height = self.textures_heights[i]
        texture_width = self.textures_widths[i]
        script_string_minors = self.read_script_block("world_side", "minors", scripts_path)
        script_string_majors = self.read_script_block("world_side", "majors", scripts_path)
        scale_x = 1 / length / texture_width
        scale_y = 1 / height / texture_height
        ogre_script_string_minors = self.read_ogre_script("world_side", "minors", scale_x, scale_y, texture_name)
        scale_x = 1 / width / texture_width
        ogre_script_string_majors = self.read_ogre_script("world_side", "majors", scale_x, scale_y, texture_name)
        front_plane_string = self.read_face("world_side", "front", length / 2, 0, height / 2, np.pi / 2, 0.0,
                                            -np.pi / 2, width, height, script_string_majors, -1, 0, 0)
        back_plane_string = self.read_face("world_side", "back", -length / 2, 0, height / 2, -np.pi / 2, 0.0,
                                           -np.pi / 2, width, height, script_string_majors, 1, 0, 0)
        left_plane_string = self.read_face("world_side", "left", 0, -width / 2, height / 2, -np.pi / 2, str(0.0),
                                           str(0.0), length, height, script_string_minors, 0, 1, 0)
        right_plane_string = self.read_face("world_side", "right", 0, width / 2, height / 2, np.pi / 2, str(0.0),
                                            str(0.0), length, height, script_string_minors, 0, -1, 0)
        box_string = front_plane_string + back_plane_string + left_plane_string + right_plane_string
        ogre_script = ogre_script_string_minors + ogre_script_string_majors
        return box_string, ogre_script


if __name__ == "__main__":
    random_map = Building.generate_random_map()
    # grid_map = Building.grid_map()
