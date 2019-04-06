import cv2
import numpy as np
import json
from collections import namedtuple

box = namedtuple("box", ["center_x", "center_y", "width", "length", "height", "orientation"])
loaded_box = None
world_name = "box_world.world"
path = "/home/patik/Diplomka/dp_ws/src/pesat_resources/worlds/"
world_path = path + world_name


class Building(object):
    def __init__(self, max_height, min_height, max_lenght, min_length, max_width, min_width, max_ratio, min_ratio):
        self.max_height = max_height
        self.max_length = max_lenght
        self.min_length = min_length
        self.max_width = max_width
        self.min_width = min_width
        self.max_ratio = max_ratio
        self.min_ratio = min_ratio
        self.min_height = min_height

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
        ratio = 1  # arcs["min_ratio"] + np.random.rand() * arcs["max_ratio"]
        side_one = arcs["min_length"] + np.random.rand() * arcs["max_length"]
        side_two = (arcs["min_width"] + np.random.rand() * arcs["max_width"]) * 1 / ratio
        height = arcs["min_height"] + np.random.rand() * arcs["max_height"]
        orientation = 0
        center_x = 0
        center_y = 0
        b = box(center_x, center_y, side_two, side_one, height, orientation)
        return [b]

    def random_object(self):
        arcs = {"max_height": self.max_height,
                "min_height": self.min_height,
                "max_length": 1 + np.random.rand() * self.max_length,
                "min_length": 0 + np.random.rand() * self.min_length,
                "max_width": 1 + np.random.rand() * self.max_width,
                "min_width": 0 + np.random.rand() * self.min_width,
                "max_ratio": 1,  # 1 + np.random.rand() * self.max_ratio,
                "min_ratio": 1}  # 0 + np.random.rand() * self.min_ratio}
        object_function = np.random.choice([self.get_i, self.get_l, self.get_u])
        return self.get_i(arcs)

    def split_to_points(self, object):
        points = []
        for part in object:
            ps = []
            p1 = [part.center_x - part.width / 2, part.center_y - part.length / 2]
            p2 = [part.center_x - part.width / 2, part.center_y + part.length / 2]
            p3 = [part.center_x + part.width / 2, part.center_y + part.length / 2]
            p4 = [part.center_x + part.width / 2, part.center_y - part.length / 2]
            ps.append(p1)
            ps.append(p2)
            ps.append(p3)
            ps.append(p4)
            points.append(
                self.shift_and_rotate_points((part.center_x, part.center_y), ps, (0, 0), part.orientation))
        return np.array(points)

    def shift_and_rotate_points(self, axis, points, shift, angle):
        changed_points = []
        s = np.sin(angle)
        c = np.cos(angle)
        for p in points:
            norm__p = [p[0] - axis[0], p[1] - axis[1]]
            rotated_p = [norm__p[0] * c - norm__p[1] * s, norm__p[0] * s + norm__p[1] * c]
            shifted_p = [rotated_p[0] + axis[0] + shift[0], rotated_p[1] + axis[1] + shift[1]]
            changed_points.append(shifted_p)
        return changed_points

    def calculate_center_of_object(self, object):
        x = 0
        y = 0
        for o in object:
            x += o.center_x
            y += o.center_y
        return [x / len(object), y / len(object)]


def read_skeleton(skeleton_file_name, plane_size, boxes):
    string = ""
    with open(skeleton_file_name, "r") as file:
        s = file.readlines()
        for line in s:
            string += line
    return string.format(plane_size=plane_size, boxes=boxes)


def read_box(box_file_name, name, x_pose, y_pose, z_pose, r_orient, p_orient, y_orient, x_size, y_size, z_size, red,
             green, blue):
    global loaded_box
    if loaded_box is None:
        loaded_box = ""
        with open(box_file_name, "r") as file:
            s = file.readlines()
            for line in s:
                loaded_box += line
    return loaded_box.format(name=name, x_pose=x_pose, y_pose=y_pose, z_pose=z_pose, r_orient=r_orient,
                             p_orient=p_orient,
                             y_orient=y_orient,
                             x_size=x_size, y_size=y_size, z_size=z_size, red=red, green=green, blue=blue)


def generate_map(sc_factor=2, world_name="box_world.world",
                 path="/home/patik/Diplomka/dp_ws/src/pesat_resources/worlds/"):
    world_path = path + world_name
    width = int(1000 / sc_factor)
    height = int(1000 / sc_factor)
    ma_height = 255
    mi_height = 50
    ma_side_one = int(50 / sc_factor)
    mi_side_one = int(100 / sc_factor)
    ma_side_two = int(50 / sc_factor)
    mi_side_two = int(100 / sc_factor)
    map = np.zeros((width, height), dtype=np.uint8)
    image = np.zeros((width, height), dtype=np.uint8)
    objects = []
    objects_count = 0
    objects_required = 250 / sc_factor
    building = Building(ma_height, mi_height, ma_side_one, mi_side_one, ma_side_two, mi_side_two, 1, 1)
    while objects_count < objects_required:
        shadow = np.zeros((width, height), dtype=np.uint8)
        shadow_map = np.zeros((width, height), dtype=np.uint8)
        object = building.random_object()
        points = building.split_to_points(object)
        center_x = np.random.random_integers(0, width)
        center_y = np.random.random_integers(0, height)
        if np.random.rand() > 0.5:
            sign = -1
        else:
            sign = 1
        orientation = np.random.rand() * np.pi * sign
        axis = building.calculate_center_of_object(object)
        for i in range(len(points)):
            new_points = np.array(building.shift_and_rotate_points(axis, points[i], [center_x, center_y], orientation),
                                  dtype=np.int32)
            cv2.fillConvexPoly(shadow, new_points, 255)
            cv2.fillConvexPoly(shadow_map, new_points, object[i].height)
        rotated_shifted_object = []
        centers = []
        for o in object:
            centers.append([o.center_x, o.center_y])
        new_centers = np.array(building.shift_and_rotate_points(axis, centers, [center_x, center_y], orientation),
                               dtype=np.int32)
        out = False
        for i in range(len(centers)):
            if new_centers[i][0] < 0 or new_centers[i][0] > width or new_centers[i][1] < 0 or new_centers[i][
                1] > height:
                out = True
            rotated_shifted_object.append(
                box(new_centers[i][0], new_centers[i][1], object[i].width, object[i].length, object[i].height,
                    object[i].orientation + orientation))
        if out:
            continue
        if cv2.countNonZero(cv2.bitwise_and(image, shadow)) == 0:
            objects.append(rotated_shifted_object)
            map = cv2.bitwise_or(map, shadow_map)
            image = cv2.bitwise_or(image, shadow)
            objects_count += 1
    map = 255 - map
    scale = 1 / 1
    index = 0
    boxes = []
    box_path = "/home/patik/Diplomka/dp_ws/src/pesat_resouces/texts/box"
    skeleton_path = "/home/patik/Diplomka/dp_ws/src/pesat_resources/texts/skeleton"
    boxes_string = ""
    for object in objects:
        for o in object:
            h = o.height * 1 / 10
            z = h / 2
            red = np.random.rand()
            green = np.random.rand()
            blue = h / 255.0
            boxes_string += read_box(box_path, "box_{}".format(str(index)), str(o.center_x * scale - width / 2),
                                     str(o.center_y * scale - height / 2),
                                     z, str(0.0), str(0.0), str(o.orientation), o.width * scale, o.length * scale, h,
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
    skeleton = read_skeleton(skeleton_path, (max(width, height) + max(ma_side_two, ma_side_one) + 20) * 2, boxes_string)
    with open(world_path, "w") as f:
        f.write(skeleton)

    with open('/home/patik/Diplomka/dp_ws/src/pesat_resouces/maps/map_data.json', 'w') as outfile:
        json.dump(boxes, outfile, indent=4)

    M = cv2.getRotationMatrix2D((width / 2, height / 2), -90, 1)
    map = cv2.warpAffine(map, M, (width, height))
    # map = cv2.flip(map, 0 )
    cv2.imwrite("/home/patik/Diplomka/dp_ws/src/pesat_resouces/maps/map.bmp", map)
    return map


if __name__ == "__main__":
    generate_map()
