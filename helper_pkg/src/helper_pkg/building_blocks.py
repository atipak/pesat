import json
from collections import namedtuple
import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import signal, spatial
import multiprocessing as mp
import time
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree, ConvexHull
import datetime

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
        # self.texture_names = ["grey_bricks.png", "red_bricks.png"]
        self.texture_names = ["mossy.png", "mossy.png"]
        # bricks_count_in_column = [41, 16]
        bricks_count_in_column = [5, 5]
        # images_pixels_width = [1233, 1024]
        images_pixels_width = [1, 1]
        # images_pixels_heigth = [3600, 1024]
        images_pixels_heigth = [1, 1]
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
        self.subworld_path = src_path + "pesat_resources/subworlds/"
        self.textures_path = src_path + "pesat_resources/textures"
        self.maximal_iteration = 10000

    @classmethod
    def generate_grid_map(cls, vertical_boxes_count=6, horizontal_boxes_count=6, world_name="grid_box_world"):
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
        image = np.zeros((pixel_width, pixel_height), dtype=np.uint8)

        objects = []
        building = Building()
        iteration = 0
        while iteration < building.maximal_iteration:
            iteration += 1
            if iteration % 500 == 0:
                print("Current iteration: " + str(iteration))
                print("Current build up: " + str(cv2.countNonZero(map) / pixels_count))
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
            if abs(cv2.countNonZero(map) / pixels_count - built_up) < 0.01:
                break
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
        print("Build up: " + str(cv2.countNonZero(map - 255) / pixels_count))
        return map

    @classmethod
    def random_map(cls, width=10, height=10, built_up=0.6, maximal_width=20.0, minimal_width=2, maximal_height=20.0,
                   minimal_height=2, maximal_orientation=1.54, minimal_orientation=0, map_resolution=100, image=None,
                   target_size=1.5, random_factor=100, random_init=500):
        pixel_width = width * map_resolution
        pixel_height = height * map_resolution
        pixels_count = pixel_width * pixel_height
        map = np.zeros((pixel_width, pixel_height), dtype=np.uint8)
        objects = []
        building = Building()
        iteration = 0
        start = time.time()
        t = width
        c = 1
        g = 1
        l = 1.0
        while True:
            t = t / 2.0
            if t > minimal_width:
                g *= 4
                c += g
            else:
                break
            l += 1.0
        print("All parts: {}".format(c))
        # i, j, width, height
        wrong_ones_const = random_init
        wrong_ones_const_reduction = random_factor
        stack = [[0, 0, width, height, wrong_ones_const]]
        wrong_ones = 0
        whole_time = 0.0
        ones = 0
        full_constant = 1.0
        minimum_space = (target_size * 2 + minimal_width) * np.power(np.pi, 2) * full_constant * map_resolution
        while len(stack) > 0:
            if wrong_ones >= wrong_ones_const:
                wrong_ones = 0
                t = time.time() - start
                ones += 1
                whole_time += t
                print("Processed: {}%".format((1 - ((c - ones) / float(c))) * 100))
                print("Time", t)
                print("Whole time", whole_time)
                start = time.time()
                cp = stack.pop(0)
                w_2 = cp[2] / 2.0
                h_2 = cp[3] / 2.0
                if w_2 > minimal_width and h_2 > minimal_width:
                    w_2 = np.round(w_2, 2)
                    h_2 = np.round(h_2, 2)
                    a = [[cp[0], cp[1], w_2, h_2], [cp[0] + w_2, cp[1], w_2, h_2], [cp[0], cp[1] + h_2, w_2, h_2],
                         [cp[0] + w_2, cp[1] + h_2, w_2, h_2]]
                    for aa in a:
                        [i, j, end_i, end_j] = aa
                        p_start_i, p_start_j, p_width, p_height = int(i * map_resolution), int(j * map_resolution), int(
                            end_i * map_resolution), int(end_j * map_resolution)
                        nonzeros = np.count_nonzero(
                            image[p_start_j: p_start_j + p_height, p_start_i:p_start_i + p_width])
                        zeros = p_height * p_width - nonzeros
                        if zeros > minimum_space:
                            stack.append([i, j, end_i, end_j, cp[4] - wrong_ones_const_reduction])
                        else:
                            print("Full part {}, {}, {}, {} with {} %".format(i, j, end_i, end_j, (
                                    float(nonzeros) / (p_height * p_width)) * 100.0))
                if len(stack) == 0:
                    break
            current_part = stack[0]
            wrong_ones_const = current_part[4]
            center_x = float(np.random.uniform(0, current_part[2], 1)[0])
            center_y = float(np.random.uniform(0, current_part[3], 1)[0])
            m_width = np.clip(maximal_width, None, current_part[2])
            object = building.random_object(maximal_height, minimal_height, m_width, minimal_width,
                                            m_width, minimal_width, maximal_orientation, minimal_orientation,
                                            center_x, center_y)
            p_start_i, p_start_j, p_width, p_height = int(current_part[0] * map_resolution), int(current_part[
                                                                                                     1] * map_resolution), int(
                current_part[2] * map_resolution), int(current_part[3] * map_resolution)
            im = image[p_start_j: p_start_j + p_height, p_start_i:p_start_i + p_width]
            cv2.imwrite("img_im.png", im)
            v, rso, sh, int_sh_pts, int_pts, b = building.check_new_object(object, current_part[2], current_part[3],
                                                                           map_resolution, im, target_size, 0.9)
            if not v:
                wrong_ones += 1
                continue
            centers = (np.array([current_part[0], current_part[1]]) * map_resolution).astype(np.uint32)
            check_map = np.copy(map)
            for i in range(len(int_pts)):
                cv2.fillConvexPoly(check_map, int_pts[i] + centers, 255)
            check_build_up = cv2.countNonZero(check_map) / pixels_count
            if check_build_up > built_up:
                wrong_ones += 1
                continue
            b = [box(rso[0].center_x + current_part[0] + current_part[2] / 2.0 - width / 2.0,
                     rso[0].center_y + current_part[1] + current_part[3] / 2.0 - height / 2.0, rso[0].width,
                     rso[0].length, rso[0].height, rso[0].orientation)]
            objects.append(b)
            # print(rotated_shifted_object)
            for i in range(len(int_sh_pts)):
                cv2.fillConvexPoly(map, int_pts[i] + centers, 255)
                cv2.fillConvexPoly(image, int_sh_pts[i] + centers, 255)
            if abs(check_build_up - built_up) < 0.01:
                break
        check_build_up = cv2.countNonZero(map) / pixels_count
        print("build up: {}".format(check_build_up))
        map = 255 - map
        # print("Build up: " + str(cv2.countNonZero(map - 255) / pixels_count))
        return map, image, objects

    def find_center(self, free_centers, map_resolution):
        non_zeros = np.nonzero(free_centers)
        l = len(non_zeros[0])
        print(l)
        if l > 0:
            i = np.random.randint(0, l, 1)[0]
            center = np.array(non_zeros[1][i], non_zeros[0][i]).astype(np.float32) / map_resolution
            center += np.random.normal(0.0, 1.0 / map_resolution, 2)
            return center
        return None

    def add_new_box_to_free_centers_map(self, new_box, free_centers_map, min_width, map_resolution):
        b = [box(new_box[0].center_x, new_box[0].center_y, new_box[0].width + min_width,
                 new_box[0].length + min_width, new_box[0].height, new_box[0].orientation)]
        points = Building.split_to_points(b)
        int_points = (points * map_resolution).astype('int64')
        for i in range(len(points)):
            cv2.fillConvexPoly(free_centers_map, int_points[i], 0)

    def find_nearest_distance(self, image, w_2):
        indices = np.nonzero(image)
        if len(indices[0]) == 0:
            return image.shape[0] / 2
        i = np.abs(np.array(indices - np.array([[w_2], [w_2]])).sum(axis=0)).argmin()
        y, x = indices[0][i], indices[1][i]
        d = np.floor(np.sqrt(np.square(x - w_2) + np.square(y - w_2)))
        return np.floor(np.cos(np.pi / 4) * d)

    def maxSubSquare(self, M):
        R = len(M)  # no. of rows in M[][]
        C = len(M[0])  # no. of columns in M[][]

        S = np.zeros((R, C), dtype=np.uint32)
        # here we have set the first row and column of S[][]

        # Construct other entries
        for i in range(1, R):
            for j in range(1, C):
                if M[i, j] == 0:
                    S[i, j] = min(S[i, j - 1], S[i - 1, j],
                                  S[i - 1, j - 1]) + 1
                else:
                    S[i, j] = 0

        # Find the maximum entry and
        # indices of maximum entry in S[][]
        ind = np.unravel_index(np.argmax(S, axis=None), S.shape)
        return ind, S[ind]

    @classmethod
    def grid_map(cls, width=10, height=10, build_up=1.0, maximal_width=20.0, maximal_height=20.0,
                 maximal_orientation=1.54, map_resolution=100, target_size=1.5, image=None):
        pixel_width = width * map_resolution
        pixel_height = height * map_resolution
        pixels_count = pixel_width * pixel_height
        map = np.zeros((pixel_width, pixel_height), dtype=np.uint8)
        objects = []
        building = Building()
        size = (maximal_width / 2) / np.cos(maximal_orientation)
        start_position = size + target_size
        end_position = start_position
        shift = 2 * size + target_size
        rest = width - start_position - end_position
        boxes_count = int(rest / shift)
        boxes_count = np.clip(boxes_count, 0, np.floor(np.sqrt(build_up) * width / maximal_width)).astype(np.int32)
        restt = rest - boxes_count * shift
        e = restt / (boxes_count + 1)
        start_position += e
        end_position += e
        shift += e
        for vertical in range(boxes_count + 1):
            for horizontal in range(boxes_count + 1):
                object = building.random_object(maximal_height, maximal_height, maximal_width, maximal_width,
                                                maximal_width, maximal_width, maximal_orientation, maximal_orientation,
                                                start_position + horizontal * shift, start_position + vertical * shift)
                value, rotated_shifted_object, shadow, int_shadow_points, int_points, b = building.check_new_object(
                    object,
                    width,
                    height,
                    map_resolution,
                    image,
                    target_size,
                    0)
                if not value:
                    continue
                objects.append(rotated_shifted_object)
                # print(rotated_shifted_object)
                for i in range(len(int_shadow_points)):
                    cv2.fillConvexPoly(map, int_points[i], 255)
                    cv2.fillConvexPoly(image, int_shadow_points[i], 255)
        map = 255 - map
        # print("Build up: " + str(cv2.countNonZero(map - 255) / pixels_count))
        return map, image, objects

    def check_new_object(self, object, width, height, map_resolution, image, target_size, r):
        if np.random.rand() < r:
            b = [box(object[0].center_x, object[0].center_y, object[0].width + 1.5 * target_size,
                     object[0].length + 1.5 * target_size, object[0].height, object[0].orientation)]
        else:
            b = object
        shadow = np.copy(image)
        points = Building.split_to_points(object)
        shadow_points = Building.split_to_points(b)
        axis = Building.calculate_center_of_object(object)
        # all points inside map
        out = False
        for ps in shadow_points:
            for p in ps:
                if p[0] < 0 or p[0] > width or p[1] < 0 or p[1] > height:
                    out = True
        if out:
            return False, None, None, [], [], b
        int_shadow_points = (shadow_points / (1.0 / map_resolution)).astype('int64')
        int_points = (points / (1.0 / map_resolution)).astype('int64')
        a = np.array(int_shadow_points[0])
        xs = a[:, 0]
        ys = a[:, 1]
        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        original_count = np.count_nonzero(shadow[min_y:max_y + 1, min_x:max_x + 1])
        for i in range(len(shadow_points)):
            cv2.fillConvexPoly(shadow, int_shadow_points[i], 0)
        cv2.imwrite("orifinal_image.png", image)
        cv2.imwrite("shadow.png", shadow)
        changed_count = np.count_nonzero(shadow[min_y:max_y + 1, min_x:max_x + 1])
        if changed_count < original_count:
            return False, None, None, [], [], b
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
        return True, rotated_shifted_object, shadow, int_shadow_points, int_points, b

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
                          maximal_height=35.0, minimal_height=0.3, maximal_orientation=1.54, minimal_orientation=0,
                          dynamical_obstacles=False, percentage=0.0):
        return {
            "world_name": world_name, "width": width, "height": height, "built_up": built_up,
            "maximal_width": maximal_width, "minimal_width": minimal_width, "maximal_height": maximal_height,
            "minimal_height": minimal_height, "maximal_orientation": maximal_orientation,
            "minimal_orientation": minimal_orientation, "maximal_length": maximal_length,
            "minimal_length": minimal_length, "dynamical_obstacles": dynamical_obstacles, "percentage": percentage
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

    def create_map_file(self, world_name, map, width, height, prefix=""):
        world_path = self.world_path + world_name
        map_file_name = world_path + "/" + prefix + world_name + ".bmp"
        original_path = world_path + "/" + prefix + "origin.bmp"
        cv2.imwrite(original_path, map)
        M = cv2.getRotationMatrix2D((width / 2.0 - 0.5, height / 2.0 - 0.5), -90, 1)
        map = cv2.warpAffine(map, M, (width, height))
        # map = cv2.flip(map, 0 )
        cv2.imwrite(map_file_name, map)

    def draw_map_into_file(self, world_name, objects, drone_position, target_position, width, height, prefix=""):
        world_path = self.world_path + world_name
        map_file_name = world_path + "/" + prefix + world_name + ".png"
        self.draw_map(objects, drone_position, target_position, width, height)
        plt.savefig(map_file_name)
        plt.close()

    def draw_map(self, objects, drone_position, target_position, width, height):
        fig, ax = plt.subplots()
        ax.set_ylim(-width / 2.0, width / 2.0)
        ax.set_xlim(-height / 2.0, height / 2.0)
        for o in objects:
            points = self.split_to_points(o)
            xs, ys = [], []
            for p in points[0]:
                xs.append(p[0])
                ys.append(p[1])
            ax.fill(xs, ys, color=(0.0, 0.0, 0.0, 0.3))
        if target_position is not None:
            ax.scatter(target_position[0], target_position[1], s=20, marker='o', c="k")
        if drone_position is not None:
            ax.scatter(drone_position[0], drone_position[1], s=20, marker='^', c="k")
        return ax

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
            points += [int(width / 2), int(height / 2)]
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
        r_number = 0.4
        red = r_number
        green = r_number
        blue = r_number
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
        front_plane_string = self.read_face_box("world_side", "front", length / 2 + box_height / 2.0, 0, height / 2,
                                                0.0, 0.0, 0.0, box_height, width, height,
                                                script_string_majors)
        back_plane_string = self.read_face_box("world_side", "back", -length / 2 - box_height / 2.0, 0, height / 2,
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


def generate_map(size, build_up, b_height, placing, orientation, dynamical_obstacles, index, random_factor,
                 random_init):
    start = time.time()
    world_name = "world-{}-{}-{}-{}-{}-{}-{}".format(size, str(build_up).replace(".", "_"),
                                                     str(b_height).replace(".", "_"), placing,
                                                     str(orientation).replace(".", "_"), dynamical_obstacles, index)
    width = size
    height = size
    built_up = build_up
    maximal_width = min(size / 2.0, 20.0)
    minimal_width = min(size / 10.0, 2.0)

    if b_height == -1:
        maximal_height = min(size / 2.0, 20.0)
        minimal_height = min(size / 10.0, 2.0)
    else:
        maximal_height = b_height
        minimal_height = b_height
    if orientation == -1:
        maximal_orientation = np.pi
        minimal_orientation = 0
    else:
        maximal_orientation = orientation
        minimal_orientation = orientation

    ### target drone placing
    drone_start_position, target_start_position = set_drone_target_on_map(size, placing)

    drone_size = 1.0
    target_size = 1.5
    map_resolution = 100
    pixel_width = width * map_resolution
    pixel_height = height * map_resolution
    pixels_count = pixel_width * pixel_height
    building = Building()
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

    ### obstacles
    all_objects = []
    percent = 0
    if dynamical_obstacles == "y":
        percent = (np.random.randint(2, 8, 1)[0] / 100.0)
        build_up -= percent
    # static obstacle
    if orientation == -1:
        statical_map, image, statical_objects = Building.random_map(width, height, built_up, maximal_width,
                                                                    minimal_width, maximal_height,
                                                                    minimal_height, maximal_orientation,
                                                                    minimal_orientation,
                                                                    map_resolution, image, target_size, random_factor,
                                                                    random_init)
        all_objects.extend(statical_objects)
    else:
        maximal_width = np.random.uniform(1.5, np.clip(np.log10(width) * 10, 0, width / 3), 1)[0]

        statical_map, image, statical_objects = Building.grid_map(width, height, build_up, maximal_width,
                                                                  maximal_height, maximal_orientation, map_resolution,
                                                                  target_size, image)
        all_objects.extend(statical_objects)
    # dynamical obstacles
    if percent != 0:
        dynamical_map, image, dynamical_objects = Building.random_map(width, height, percent, maximal_width,
                                                                      minimal_width,
                                                                      maximal_height, minimal_height,
                                                                      maximal_orientation,
                                                                      minimal_orientation, map_resolution, image,
                                                                      target_size, random_factor, random_init)
        all_objects.extend(dynamical_objects)
    building.prepare_directories(world_name)
    tringles, connectivity = building.triangulation_connectivity(all_objects,
                                                                 [box(0, 0, width, height, maximal_height, 0)])
    boxes = building.create_world_file_from_planes(world_name, all_objects, width, height, maximal_height)
    world_info = building.create_world_info(world_name, width, height, built_up, maximal_width, minimal_width,
                                            maximal_width, minimal_width, maximal_height, minimal_height,
                                            maximal_orientation, minimal_orientation)
    # print("Boxes count:", len(boxes))
    statical_map_boxes = boxes[:len(statical_objects)]
    dynamical_map_boxes = boxes[len(statical_objects):]
    json_text = {"world": world_info, "objects": {"dynamical": dynamical_map_boxes,
                                                  "statical": statical_map_boxes}, "mesh": {"triangles": tringles,
                                                                                            "connectivity": connectivity}}
    building.create_json_file(world_name, json_text)
    building.draw_map_into_file(world_name, statical_objects, drone_start_position, target_start_position, width,
                                height, prefix="statical_image_")
    # building.create_map_file(world_name, statical_map, pixel_width, pixel_height, prefix="statical_")
    im = statical_map
    if percent != 0:
        # building.create_map_file(world_name, dynamical_map, pixel_width, pixel_height, prefix="dynamical_")
        building.draw_map_into_file(world_name, dynamical_objects, drone_start_position, target_start_position, width,
                                    height, prefix="dynamical_image_")
        im = np.bitwise_and(statical_map, dynamical_map).astype(np.uint8)
    # building.create_map_file(world_name, im, pixel_width, pixel_height)
    building.draw_map_into_file(world_name, all_objects, drone_start_position, target_start_position, width,
                                height, prefix="image_")
    # print("Percentage: {}".format(percent))
    bb = cv2.countNonZero(im - 255) / float(pixels_count)
    # print("Build up: " + str(bb))
    return [bb, len(boxes), percent, time.time() - start]


def set_drone_target_on_map(size, placing):
    max_iterations = 20
    while True:
        drone_x = np.random.uniform(-size / 2.0 + 1, size / 2.0 - 1.5, 1)[0]
        drone_y = np.random.uniform(-size / 2.0 + 1.5, size / 2.0 - 1.5, 1)[0]
        drone_start_position = [drone_x, drone_y]
        target_start_position = None
        if placing == "n":
            # 5
            max_x = np.clip(drone_x + 30, drone_x + 3, size / 2.0 - 1.5)
            max_y = np.clip(drone_y + 1.5, drone_y + 0, size / 2.0 - 1.5)
            min_y = np.clip(drone_y - 1.5, -size / 2.0 + 1.5, drone_y - 1)
            target_x = np.random.uniform(drone_x + 3, max_x, 1)[0]
            target_y = np.random.uniform(min_y, max_y, 1)[0]
            target_start_position = [target_x, target_y]
        else:
            iterations = 0
            while True:
                iterations += 1
                if iterations > max_iterations:
                    break
                i = np.random.randint(1, 5, 1)[0]
                if 3 <= i <= 4:
                    max_y = np.clip(drone_y + 15, drone_y + 1, size / 2.0 - 1.2)
                    min_y = np.clip(drone_y - 15, -size / 2.0 + 1.2, drone_y - 1)
                elif i == 1:
                    min_y = -size / 2.0 + 1.2
                    max_y = np.clip(drone_y - 15 - 1.2, -size / 2.0 + 1.2, None)
                else:
                    max_y = size / 2.0 - 1.2
                    min_y = np.clip(drone_y + 15 + 1.2, None, size / 2.0 - 1.2)
                if 1 <= i <= 2:
                    min_x = -size / 2.0 + 1.2
                    max_x = size / 2.0 - 1.2
                elif i == 3:
                    min_x = -size / 2.0 + 1.2
                    max_x = np.clip(drone_x - 1, -size / 2.0 + 1.2, None)
                else:
                    min_x = np.clip(drone_x + 30, None, size / 2.0 - 1.2)
                    max_x = size / 2.0 - 1.2
                if max_x <= min_x or max_y <= min_y:
                    continue
                else:
                    target_x = np.random.uniform(min_x, max_x, 1)[0]
                    target_y = np.random.uniform(min_y, max_y, 1)[0]
                    target_start_position = [target_x, target_y]
                    break
        if target_start_position is not None:
            break
    return drone_start_position, target_start_position


def generate_random_submap(size, b_height, index, random_factor, random_init):
    start = time.time()
    world_name = "subworld-{}".format(index)
    width = size
    height = size
    maximal_width = min(size / 2.0, 20.0)
    minimal_width = min(size / 10.0, 2.0)

    if b_height == -1:
        maximal_height = min(size / 2.0, 20.0)
        minimal_height = min(size / 10.0, 2.0)
    else:
        maximal_height = b_height
        minimal_height = b_height
    maximal_orientation = np.pi
    minimal_orientation = 0

    target_size = 1.5
    map_resolution = 100
    pixel_width = width * map_resolution
    pixel_height = height * map_resolution
    pixels_count = pixel_width * pixel_height
    building = Building()
    image = np.zeros((pixel_width, pixel_height), dtype=np.uint8)

    ### obstacles
    all_objects = []
    # static obstacle
    statical_map, image, statical_objects = Building.random_map(width, height, 1.0, maximal_width,
                                                                minimal_width, maximal_height,
                                                                minimal_height, maximal_orientation,
                                                                minimal_orientation,
                                                                map_resolution, image, target_size, random_factor,
                                                                random_init)

    bb = cv2.countNonZero(statical_map - 255) / float(pixels_count)
    json_text = {"world_name": world_name, "width": size, "build_up": bb, "height": b_height,
                 "objects": statical_objects}
    json_file_name = building.subworld_path + world_name + ".json"
    with open(json_file_name, 'w') as outfile:
        json.dump(json_text, outfile, indent=4)
    map_file_name = building.subworld_path + world_name + ".png"
    building.draw_map(statical_objects, None, None, width, height)
    plt.savefig(map_file_name)
    plt.close()
    return world_name, [bb, size, b_height, time.time() - start]


def decrease_build_up(objects, size, target_build_up):
    size_squared = size * size
    target_size = target_build_up * size_squared
    a = [o[0][2] * o[0][3] for o in objects]
    objects_sorted = np.array(objects[np.argsort(a)])[::-1]
    a_sorted = np.sort(a)[::-1]
    sum_a = 0.0
    selected = []
    for i in range(len(a_sorted)):
        if sum_a + a_sorted[i] < target_size:
            sum_a += a_sorted[i]
            selected.append(objects_sorted[i])
    return selected


def move_and_resize_objects(objects, k, l, size, original_size, height):
    moved_objects = []
    for o in objects:
        o_copy = np.copy(o)
        o_copy[0][0] += original_size / 2.0 + k * original_size - size / 2.0
        o_copy[0][1] += original_size / 2.0 + l * original_size - size / 2.0
        if height == -1:
            o_copy[0][4] = np.random.uniform(min(size / 10.0, 2.0), min(size / 2.0, 20.0), 1)[0]
        else:
            o_copy[0][4] = height
        moved_objects.append(o_copy)
    return moved_objects


def dynamic_avoidance_map():
    width, height, maximal_height, built_up, maximal_width, minimal_width = 100, 100, 20, 0.0, 20, 0
    maximal_height, minimal_height, maximal_orientation, minimal_orientation = 20, 0, np.pi, 0
    # obstacle_one = [box(4, -4, 8, 2, 10, 0.0)]
    # obstacle_two = [box(4, 4, 8, 2, 10, 0.0)]
    obstacle_three = [box(11, 0, 4.24, 4.24, 5, 0)]
    obstacle_four = [box(18.5, 0, 0.5, 20, 13, 0.0)]
    obstacles = [obstacle_three, obstacle_four]
    world_name = "base_dynamic_world"
    building = Building()
    building.prepare_directories(world_name)
    tringles, connectivity = building.triangulation_connectivity(obstacles,
                                                                 [box(0, 0, width, height, maximal_height, 0)])
    boxes = building.create_world_file_from_planes(world_name, obstacles, width, height, maximal_height)
    world_info = building.create_world_info(world_name, width, height, built_up, maximal_width, minimal_width,
                                            maximal_width, minimal_width, maximal_height, minimal_height,
                                            maximal_orientation, minimal_orientation)
    statical_map_boxes = boxes[:]
    dynamical_map_boxes = []
    json_text = {"world": world_info, "objects": {"dynamical": dynamical_map_boxes,
                                                  "statical": statical_map_boxes}, "mesh": {"triangles": tringles,
                                                                                            "connectivity": connectivity}}
    building.create_json_file(world_name, json_text)
    building.draw_map_into_file(world_name, obstacles, [0.0, 0.0], None, width, height, prefix="statical_image_")


def base_maps():
    sizes = [10, 100]
    builts = [0.1, 0.4, 0.7]
    typs = [0, np.pi / 2]
    for size in sizes:
        for built in builts:
            for typ in typs:
                world_name = "box_world_" + str(size) + "_" + str(size) + "_" + str(built) + "_cl_" + str(typ)
                width = size
                height = size
                built_up = built
                maximal_width = 20.0
                minimal_width = 2
                maximal_height = 20.0
                minimal_height = 2
                maximal_orientation = typ
                minimal_orientation = 0
                map_resolution = 100
                drone_size = 1.0
                drone_start_position = [0, 0]
                target_size = 1.5
                target_start_position = [5, 0]
                random_map = Building.generate_random_map(world_name, width, height, built_up, maximal_width,
                                                          minimal_width, maximal_height, minimal_height,
                                                          maximal_orientation, minimal_orientation, map_resolution,
                                                          drone_size, drone_start_position, target_size,
                                                          target_start_position)


def multi_generate_map(ar):
    [w, b, h, p, o, d, index] = ar
    [bb, boxes, percent] = generate_map(w, b, h, p, o, d, index)
    print(
        "Map width: {}, build up: {}, height: {}, position: {}, orientation: {}, dynamical_obstacles: {}, index: {}, result build up: {}, result boxes count: {}, result percentage: {}".format(
            w, b, h, p, o, d, index, bb, boxes, percent))
    return [bb, boxes, percent]


def experiment_maps():
    # width, build up, height, position, orientation, dynamical obstacles
    width = [10, 100, 1000]
    build_up = [0.1, 0.45, 0.75]
    height = [1.5, 20, -1]
    position = ["n", "f"]
    orientation = [-1, -1, 0, 0.45]
    dynamical_obstacles = ["y", "n"]
    count = 2
    index = 0
    ar = []
    for _ in range(count):
        for w in width:
            for b in build_up:
                for h in height:
                    for p in position:
                        for o in orientation:
                            for d in dynamical_obstacles:
                                index += 1
                                ar.append([w, b, h, p, o, d, index])
    p = mp.Pool(4)
    # a = p.map(multi_generate_map, ar)
    # generate_map(10, 0.4, 1.5, "n", 0, "n", index)
    rs = []
    for i in [[1000, -50]]:  # [[1000,-50], [1000,50], [800,100]]:
        r = generate_map(50, 0.75, 2.5555, "f", -1, "y", index, i[1], i[0])
        rs.append(r)
    print(rs)
    print("OK")


def ex_submap(index):
    b_height = np.random.choice([-1, 1.5, 20])
    name, data = generate_random_submap(50, b_height, index, -200, 1200)
    return [name, data]


def generate_experiments_submaps():
    b = Building()
    if not os.path.exists(b.subworld_path):
        os.makedirs(b.subworld_path)
    d = {}
    p = mp.Pool(4)
    a = p.map(ex_submap, range(500))
    for aa in a:
        d[aa[0]] = aa[1]
    json_file_name = b.subworld_path + "general.json"
    with open(json_file_name, 'w') as outfile:
        json.dump(d, outfile, indent=4)


def generate_random_maps_from_submaps():
    width = [100, 1000]
    build_up = [0.1, 0.25, 0.4]
    position = ["n", "f"]
    dynamical_obstacles = ["y", "n"]
    count = 4
    index = 0
    arr = []
    for wi in width:
        for bu in build_up:
            for po in position:
                for do in dynamical_obstacles:
                    for co in range(count):
                        index += 1
                        arr.append([wi, bu, -1, po, do, index])
    p = mp.Pool(4)
    a = p.map(grmfs, arr)


def grmfs(a):
    [size, build_up, height, position, dynamical_obstacles, index] = a
    generate_random_map_from_submaps(size, build_up, height, position, dynamical_obstacles, index)
    return 0


def generate_random_map_from_submaps(size, build_up, height, position, dynamical_obstacles, index):
    start = time.time()
    statical_objects, dynamical_objects, all_objects, drone_position, target_position, final_built_up = generate_map_from_submaps(
        size, build_up, height, position, dynamical_obstacles)
    print("Time {}".format(time.time() - start))
    start = time.time()
    world_name = "sworld-{}-{}-{}-{}-{}-{}-{}".format(size, str(np.round(final_built_up, 4)).replace(".", "_"),
                                                      str(height).replace(".", "_"), position,
                                                      str(-1).replace(".", "_"), dynamical_obstacles, index)
    maximal_width = min(size / 2.0, 20.0)
    minimal_width = min(size / 10.0, 2.0)
    maximal_orientation = np.pi
    minimal_orientation = 0
    if height == -1:
        maximal_height = min(size / 2.0, 20.0)
        minimal_height = min(size / 10.0, 2.0)
    else:
        maximal_height = height
        minimal_height = height
    create_world(world_name, statical_objects, dynamical_objects, all_objects, drone_position, target_position,
                 size, final_built_up, maximal_width, minimal_width, maximal_height, minimal_height,
                 maximal_orientation, minimal_orientation)
    print("Time {}".format(time.time() - start))
    return world_name, drone_position, target_position


def generate_map_by_standard_method(size, build_up, height, position, dynamical_obstacles):
    b = Building()
    maximal_width = min(size / 2.0, 20.0)
    minimal_width = min(size / 10.0, 2.0)
    if height == -1:
        maximal_height = 20
        minimal_height = 2
    else:
        maximal_height = height
        minimal_height = height
    maximal_orientation = np.pi
    minimal_orientation = 0
    ### target drone placing
    drone_start_position, target_start_position = set_drone_target_on_map(size, position)

    drone_size = 1.0
    target_size = 1.5
    map_resolution = 100
    pixel_width = size * map_resolution
    pixel_height = size * map_resolution
    pixels_count = pixel_width * pixel_height
    building = Building()
    image = np.zeros((pixel_width, pixel_height), dtype=np.uint8)
    # insert places for target and drone
    drone_position = [
        box(drone_start_position[0] + size / 2, drone_start_position[1] + size / 2, drone_size, drone_size,
            drone_size, 0)]
    target_position = [
        box(target_start_position[0] + size / 2, target_start_position[1] + size / 2, target_size, target_size,
            target_size, 0)]
    drone_target_line = [
        box((target_start_position[0] + size + drone_start_position[0]) / 2,
            (target_start_position[1] + size + drone_start_position[1]) / 2,
            (target_start_position[1] + size + drone_start_position[1]), target_size,
            target_size, 0)]
    drone_position_points = (Building.split_to_points(drone_position) / (1.0 / map_resolution)).astype('int64')
    target_position_points = (Building.split_to_points(target_position) / (1.0 / map_resolution)).astype('int64')
    drone_target_line_points = (Building.split_to_points(drone_target_line) / (1.0 / map_resolution)).astype('int64')
    cv2.fillConvexPoly(image, drone_position_points[0], 255)
    cv2.fillConvexPoly(image, target_position_points[0], 255)
    cv2.fillConvexPoly(image, drone_target_line_points[0], 255)
    cv2.imwrite("td.png", image)

    statical_map, image, statical_objects = Building.random_map(size, size, build_up, maximal_width, minimal_width,
                                                                maximal_height, minimal_height, maximal_orientation,
                                                                minimal_orientation, map_resolution, image, target_size,
                                                                -200, 1200)
    final_build_up = cv2.countNonZero(statical_map - 255) / float(pixels_count)
    drone_position = [drone_position[0].center_x - size / 2, drone_position[0].center_y - size / 2, 2]
    target_position = [target_position[0].center_x - size / 2, target_position[0].center_y - size / 2, 1]
    return statical_objects, [], statical_objects, drone_position, target_position, final_build_up


def generate_random_map_by_origin_method(size, build_up, height, position, dynamical_obstacles, index):
    start = time.time()
    statical_objects, dynamical_objects, all_objects, drone_position, target_position, final_built_up = generate_map_by_standard_method(
        size, build_up, height, position, dynamical_obstacles)
    print("Time {}".format(time.time() - start))
    start = time.time()
    world_name = "sworld-{}-{}-{}-{}-{}-{}-{}".format(size, str(np.round(final_built_up, 4)).replace(".", "_"),
                                                      str(height).replace(".", "_"), position,
                                                      str(-1).replace(".", "_"), dynamical_obstacles, index)
    maximal_width = min(size / 2.0, 20.0)
    minimal_width = min(size / 10.0, 2.0)
    maximal_orientation = np.pi
    minimal_orientation = 0
    if height == -1:
        maximal_height = min(size / 2.0, 20.0)
        minimal_height = min(size / 10.0, 2.0)
    else:
        maximal_height = height
        minimal_height = height
    create_world(world_name, statical_objects, dynamical_objects, all_objects, drone_position, target_position,
                 size, final_built_up, maximal_width, minimal_width, maximal_height, minimal_height,
                 maximal_orientation, minimal_orientation)
    print("Time {}".format(time.time() - start))
    return world_name, drone_position, target_position


def objects_reduction(objects, size):
    map_resolution = 100
    pixel_width = size * map_resolution
    pixel_height = size * map_resolution
    map = np.zeros((pixel_width, pixel_height), dtype=np.uint8)
    building = Building()
    areas = objects[:, 0, 2] * objects[:, 0, 3]
    mask = areas > 9
    big_objects = objects[mask]
    for bo in big_objects:
        bob = [box(bo[0, 0] + size / 2.0, bo[0, 1] + size / 2.0, bo[0, 2], bo[0, 3], bo[0, 4], bo[0, 5])]
        points = Building.split_to_points(bob)
        int_points = (points * map_resolution).astype('int64')
        for i in range(len(int_points)):
            cv2.fillConvexPoly(map, int_points[i], 255)
    small_objects = objects[~mask]
    centers = []
    used = np.zeros(len(small_objects), dtype=np.bool)
    for so in small_objects:
        centers.append([so[0, 0], so[0, 1]])
    kd_centers = spatial.cKDTree(centers)
    new_objects = []
    for i in range(len(used)):
        if not used[i]:
            dd, jj = kd_centers.query(centers[i], 6)
            found = False
            for j in range(len(jj)):
                if dd[j] > 1:
                    tj = jj[j]
                    if not used[tj]:
                        x1, y1, w1, l1, v1 = small_objects[tj, 0, 0], small_objects[tj, 0, 1], small_objects[tj, 0, 2], \
                                             small_objects[tj, 0, 3], small_objects[tj, 0, 4]
                        x2, y2, w2, l2, v2 = small_objects[i, 0, 0], small_objects[i, 0, 1], small_objects[i, 0, 2], \
                                             small_objects[i, 0, 3], small_objects[i, 0, 4]
                        final_height = (v1 + v2) / 2.0
                        center = [(x1 + x2) / 2.0 + size / 2.0, (y1 + y2) / 2.0 + size / 2.0]
                        width = max(np.sqrt(np.square(w1) + np.square(l1)), np.sqrt(np.square(w2) + np.square(l2)))
                        length = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
                        orientation = np.arctan2(y1 - y2, x1 - x2)
                        if orientation < 0:
                            orientation += 2 * np.pi
                        if orientation > np.pi:
                            orientation = np.arctan2(y2 - y1, x2 - x1)
                            if orientation < 0:
                                orientation += 2 * np.pi
                            if orientation > np.pi:
                                continue
                        b = [box(center[0], center[1], width, length, final_height, orientation)]
                        v, rso, sh, int_sh_pts, int_pts, b = building.check_new_object(b, size, size, map_resolution,
                                                                                       map, 1.5, 1)
                        if not v:
                            continue
                        new_objects.append([[center[0], center[1], width, length, final_height, orientation]])
                        found = True
                        used[tj] = True
                        used[i] = True
                        for i in range(len(int_pts)):
                            cv2.fillConvexPoly(map, int_pts[i], 255)
                        break
            if not found:
                b = [box(small_objects[i, 0, 0], small_objects[i, 0, 1], small_objects[i, 0, 2], small_objects[i, 0, 3],
                         small_objects[i, 0, 4], small_objects[i, 0, 5])]
                v, rso, sh, int_sh_pts, int_pts, b = building.check_new_object(b, size, size, map_resolution,
                                                                               map, 1.5, 1)
                for i in range(len(int_pts)):
                    cv2.fillConvexPoly(map, int_pts[i], 255)
                new_objects.append(np.copy(small_objects[i]))
                used[i] = True
    oo = []
    oo.extend(big_objects)
    oo.extend(new_objects)
    return np.array(oo)


def generate_map_from_submaps(size, build_up, height, position, dynamical_obstacles):
    b = Building()
    if size % 50 != 0:
        raise Exception("Wrong size.")
    json_file_name = b.subworld_path + "general.json"
    with open(json_file_name, 'r') as inputfile:
        data = json.load(inputfile)
    keys, values = [], []
    for key, value in data.items():
        keys.append(key)
        values.append(value)
    values = np.array(values)
    # print(np.max(values[:, 0]), np.min(values[:, 0]), np.mean(values[:, 0]))
    # _ = plt.hist(values[:, 0], bins='auto')
    # plt.title("Histogram with 'auto' bins")
    # plt.show()
    keys = np.array(keys)
    mask = values[:, 0] >= build_up
    filtered_values = values[mask]
    filtered_keys = keys[mask]
    ratio = int(size / 50)
    count = int(ratio * ratio)
    a = np.random.choice(np.arange(0, len(filtered_values)), count)
    right_objects = []
    all_objects = []
    reduced_objects = 0
    for k in range(ratio):
        for l in range(ratio):
            aa = a[k * ratio + l]
            name = filtered_keys[aa]
            json_file_name = b.subworld_path + name + ".json"
            with open(json_file_name, 'r') as inputfile:
                map_data = json.load(inputfile)
            objects = np.array(map_data["objects"])
            c_o = len(objects)
            # objects = objects_reduction(objects, 50)
            reduced_objects += c_o - len(objects)
            selected_objects = decrease_build_up(objects, 50.0, build_up)
            selected_objects = move_and_resize_objects(selected_objects, k, l, size, 50.0, height)
            moved_objects = move_and_resize_objects(objects, k, l, size, 50.0, height)
            right_objects.extend(selected_objects)
            all_objects.extend(moved_objects)
    right_objects = np.array(right_objects)
    all_objects = np.array(all_objects)
    print("After reduction: {}".format(reduced_objects))
    drone_position, target_position, _ = free_position_for_drone_and_target(all_objects, position)
    if drone_position is None:
        print("Unfortunately there weren't find places for drone and its target.")
    else:
        t = right_objects[:, 0, [0, 1]] == np.array(drone_position[:2])
        s = np.sum(t, axis=1)
        indices_drone = np.nonzero(s == 2)[0]
        t = right_objects[:, 0, [0, 1]] == np.array(target_position[:2])
        s = np.sum(t, axis=1)
        indices_target = np.nonzero(s == 2)[0]
        mm = np.ones(len(right_objects), dtype=np.bool)
        mm[indices_drone] = False
        mm[indices_target] = False
        right_objects = right_objects[mm]
    if dynamical_obstacles == "y":
        a = np.array([o[0][2] * o[0][3] for o in right_objects])
        mask = a > 9
        statical_objects = list(right_objects[mask])
        dynamical_objects = []
        smaller_objects = right_objects[~mask]
        if len(smaller_objects) > 0:
            r = len(smaller_objects) * 0.1
            r = np.clip(r, None, 0.03 * len(right_objects))
            dynamical_indices = np.random.choice(np.arange(0, len(smaller_objects)), int(r), False)
            statical_mask = np.ones(len(smaller_objects), dtype=np.bool)
            statical_mask[dynamical_indices] = False
            dynamical_objects = list(smaller_objects[~statical_mask])
            statical_objects.extend(list(smaller_objects[statical_mask]))
    else:
        dynamical_objects = []
        statical_objects = list(right_objects)
    s = np.sum(right_objects[:, 0, 2] * right_objects[:, 0, 3])
    bp = s / (size * size)
    return statical_objects, dynamical_objects, right_objects, drone_position, target_position, bp


def create_world(world_name, s_objects, d_objects, right_objects, drone_position, target_position,
                 size, built_up, maximal_width, minimal_width, maximal_height, minimal_height, maximal_orientation,
                 minimal_orientation):
    b = Building()
    b.prepare_directories(world_name)
    width, height = size, size
    all_objects = []
    statical_objects = []
    dynamical_objects = []
    for ro in right_objects:
        all_objects.append([box(ro[0][0], ro[0][1], ro[0][2], ro[0][3], ro[0][4], ro[0][5])])
    for ro in s_objects:
        statical_objects.append([box(ro[0][0], ro[0][1], ro[0][2], ro[0][3], ro[0][4], ro[0][5])])
    for ro in d_objects:
        dynamical_objects.append([box(ro[0][0], ro[0][1], ro[0][2], ro[0][3], ro[0][4], ro[0][5])])
    start = time.time()
    tringles, connectivity = b.triangulation_connectivity(all_objects, [box(0, 0, width, height, maximal_height, 0)])
    print("Triangulation {}".format(time.time() - start))
    start = time.time()
    max_world_height = max(maximal_height, 20)
    boxes = b.create_world_file_from_planes(world_name, all_objects, width, height, max_world_height)
    print("Boxes {}".format(time.time() - start))
    start = time.time()
    world_info = b.create_world_info(world_name, width, height, built_up, maximal_width, minimal_width,
                                     maximal_width, minimal_width, max_world_height, minimal_height,
                                     maximal_orientation, minimal_orientation)
    statical_map_boxes = boxes[:len(statical_objects)]
    dynamical_map_boxes = boxes[len(statical_objects):]
    json_text = {"world": world_info, "objects": {"dynamical": dynamical_map_boxes,
                                                  "statical": statical_map_boxes}, "mesh": {"triangles": tringles,
                                                                                            "connectivity": connectivity}}
    b.create_json_file(world_name, json_text)
    print("Json {}".format(time.time() - start))
    start = time.time()
    b.draw_map_into_file(world_name, statical_objects, drone_position, target_position, width, height,
                         prefix="statical_image_")
    print("Statical map {}".format(time.time() - start))
    start = time.time()
    if len(dynamical_objects) != 0:
        b.draw_map_into_file(world_name, dynamical_objects, drone_position, target_position, width, height,
                             prefix="dynamical_image_")
    print("Dynamical map {}".format(time.time() - start))
    start = time.time()
    b.draw_map_into_file(world_name, all_objects, drone_position, target_position, width, height, prefix="image_")
    print("Full map {}".format(time.time() - start))


def convolution_method(boxes, target_size, drone_size, map_resolution):
    for b in boxes:
        points = Building.split_to_points(b)
        int_points = (points * map_resolution).astype('int64')
        for i in range(len(points)):
            cv2.fillConvexPoly(map, int_points[i], 255)
    kernel_drone = np.ones((int(drone_size * map_resolution), int(drone_size * map_resolution)))
    kernel_target = np.ones((int(target_size * map_resolution), int(target_size * map_resolution)))
    print("Convolution")
    drone_conv = signal.convolve2d(map, kernel_drone, boundary='symm', mode='same')
    target_conv = signal.convolve2d(map, kernel_target, boundary='symm', mode='same')
    free_drone_mask = drone_conv == 0
    free_target_mask = target_conv == 0
    drone_posibilities = np.nonzero(free_drone_mask)
    target_posibilities = np.nonzero(free_target_mask)
    if len(drone_posibilities[0]) == 0 or len(target_posibilities[0]) == 0:
        return None, None
    max_x = 30 * map_resolution
    min_y = -15 * map_resolution
    max_y = 15 * map_resolution
    perm_drone = np.random.permutation(len(drone_posibilities[0]))
    print("Perm drone: {}".format(perm_drone))


def free_position_for_drone_and_target(objects, mode):
    boolean_mode_near = mode == "n"
    hfov = 1.7
    image_height = 480
    image_width = 856
    camera_range = 50
    focal_length = image_width / (2.0 * np.tan(hfov / 2.0))
    vfov = 2 * np.arctan2(image_height / 2, focal_length)
    x_coors = []
    y_coors = []
    corners_height = []
    boolean_mask = []
    objects_centers = []
    for obj in objects:
        o = obj[0]
        object_class = [box(*o)]
        corners = Building.split_to_points(object_class)
        height = o[4]
        corners_height.append([corners, height])
        objects_centers.append([o[0], o[1]])
        if o[2] * o[3] < 9:
            boolean_mask.append(True)
            x_coors.append(o[0])
            y_coors.append(o[1])
        else:
            boolean_mask.append(False)
    x_coors = np.array(x_coors)
    y_coors = np.array(y_coors)
    corners_height = np.array(corners_height)
    tree = cKDTree(np.array(objects_centers))
    boolean_mask = np.array(boolean_mask, dtype=np.bool)
    if len(x_coors) == 0:
        return None, None, None
    perm_drone = np.random.permutation(len(x_coors))
    boolean_mask_nonzeros = np.nonzero(boolean_mask)[0]
    max_x = 30
    min_x = 5
    min_y = -15
    max_y = 15
    for i in perm_drone:
        y_mask = np.logical_and(y_coors > y_coors[i] + min_y, y_coors < y_coors[i] + max_y)
        x_mask = np.logical_and(x_coors > x_coors[i] + min_x, x_coors < x_coors[i] + max_x)
        mask = np.logical_and(y_mask, x_mask)
        if not boolean_mode_near:
            mask = ~mask
        target_y = y_coors[mask]
        if len(target_y) == 0:
            continue
        target_x = x_coors[mask]
        target_permutation = np.random.permutation(len(target_y))
        for target_index in target_permutation:
            # target_index = np.random.randint(0, len(target_y), 1)[0]
            nonzeros = np.nonzero(mask)[0]
            small_objects_target_index = nonzeros[target_index]
            for height in range(4, 12):
                drone_target_distance = np.sqrt(
                    np.square(x_coors[i] - target_x[target_index]) + np.square(
                        y_coors[i] - target_y[target_index]) + np.square(height - 1))
                drone_configuration = [x_coors[i], y_coors[i], height, 0.0,
                                       0.0, 0]
                objects_drone_index = boolean_mask_nonzeros[i]
                objects_target_index = boolean_mask_nonzeros[small_objects_target_index]
                m = np.ones(len(objects), dtype=np.bool)
                m[objects_drone_index] = False
                m[objects_target_index] = False
                vis = is_target_visible(drone_configuration, vfov, hfov, camera_range,
                                        [target_x[target_index], target_y[target_index], 0.2], tree, corners_height[m])
                if (vis and boolean_mode_near) or (not vis and not boolean_mode_near):
                    return [x_coors[i], y_coors[i], height], [target_x[target_index], target_y[target_index], 1], \
                           objects[m]
    return None, None, None


def in_hull(points, point):
    hull = ConvexHull(points, incremental=True)
    plt.plot([point[0]], [point[1]], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    area = hull.area
    hull.add_points(np.array([point]))
    return abs(area - hull.area) < 0.1


def is_target_visible(position, vfov, hfov, max_range, target_position, tree, objects):
    height = 0.2
    h_angles = [position[5] - hfov / 2.0, position[5] + hfov / 2.0]
    v_angles = np.clip([position[4] - vfov / 2.0, position[4] + vfov / 2.0],
                       (0, 0), (np.pi / 2, np.pi / 2))
    angles = [[v_angles[0], h_angles[0]],  # bottom left
              [v_angles[1], h_angles[0]],  # upper left
              [v_angles[1], h_angles[1]],  # upper right
              [v_angles[0], h_angles[1]]]  # bottom right
    r = R.from_euler('yz', angles)
    start_point = np.array([position[0], position[1], position[2]])
    vectors = r.apply(np.array([1, 0, 0]))
    t = height + (-position[2] / vectors[:, 2])
    t = np.nan_to_num(t)
    t[t < 0] = max_range
    t = np.clip(t, None, max_range)
    visible_points = np.empty((4, 2))
    visible_points[:, 0] = t * vectors[:, 0] + position[0]
    visible_points[:, 1] = t * vectors[:, 1] + position[1]
    if not in_hull(visible_points, [target_position[0], target_position[1]]):
        plt.clf()
        return False
    # find corners
    ii = np.array(tree.query_ball_point([start_point[:2]], max_range)[0]).astype(np.int32)
    ii = ii[ii < len(objects)]
    if len(ii) == 0:
        return True
    else:
        mask = np.zeros(len(objects), np.bool)
        mask[ii] = True
        selected_objects = objects[mask]
        for selected_object in selected_objects:
            # height = selected_object[1]
            cs = np.zeros((4, 3))
            cs[:, :2] = selected_object[0]
            cs[:, 2] = selected_object[1]
            vectors = cs - start_point
            t = height + (-position[2] / vectors[:, 2])
            t = np.nan_to_num(t)
            t[t < 0] = max_range
            t = np.clip(t, None, max_range)
            if (vectors[:, 2] == 0).any():
                color = 'r'
            else:
                color = 'b'
            xyz = np.empty((8, 2))
            xyz[:4, 0], xyz[:4, 1] = cs[:, 0], cs[:, 1]
            xyz[4:, 0], xyz[4:, 1] = t * vectors[:, 0] + position[0], t * vectors[:, 1] + position[1]
            plt.fill(cs[:, 0], cs[:, 1], color, alpha=0.3)
            if in_hull(xyz, [target_position[0], target_position[1]]):
                plt.clf()
                return False
        plt.plot([position[0]], [position[1]], 'x')
        plt.show()
        return True


def generate_drone_launch_file(file_path, world_name="basic", drone_place=None, extra_localization="false",
                               slam="false"):
    if drone_place is None:
        drone_place = [0.0, 0.0, 0.1]
    text = """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<launch>
    <arg name="world_name" default="{}" />
    <arg name="drone_x" default="{}" />
    <arg name="drone_y" default="{}" />
    <arg name="drone_z" default="{}" />
    <arg name="extra_localization" default="{}" />
    <arg name="slam" default="{}" />
    


    <!-- Gazebo -->
    <include file="$(find velocity_controller)/launch/bebop_cmd_vel.launch">
        <arg name="extra_localization" value="$(arg extra_localization)" />
        <arg name="world_name" value="$(arg world_name)"/>
        <arg name="world_path" value="$(find pesat_resources)/worlds"/>
        <arg name="x" default="$(arg drone_x)" />
        <arg name="y" default="$(arg drone_y)" />
        <arg name="z" default="$(arg drone_z)" />
    </include>

    <!-- map server -->
    <group ns="bebop2">
        <!-- Moveit -->
        <include file="$(find central_system)/launch/drone_moveit.launch"/>
        <include file="$(find central_system)/launch/map_planning_server.launch"/>
    </group>


    <!-- Robot localization package -->
    <include file="$(find central_system)/launch/localization.launch" if="$(arg extra_localization)"/>

    <!-- SLAM -->
    <include file="$(find central_system)/launch/orb_slam2_gazebo_mono.launch" if="$(arg slam)">
        <arg name="namespace" value="bebop2" />
    </include>

</launch>
    """.format(world_name, drone_place[0], drone_place[1], drone_place[2], extra_localization, slam)
    with open(file_path, "w") as file:
        file.write(text)


def generate_target_launch_file(file_path, target_position=None, target="true", targetloc="true", cctv="true"):
    if target_position is None:
        target_position = [5.0, 0.0, 1.5]
    if len(target_position) == 2:
        target_position.append(1.5)
    text = """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<launch>
    <!-- target section-->
	<arg name="target" default="{}" />
	<arg name="targetloc" default="{}" />
	<arg name="cctv" default="{}" />
	<arg name="target_x" default="{}"/>
    <arg name="target_y" default="{}"/>
    <arg name="target_z" default="{}"/>

    <group ns="target">
        <!-- Target localization -->
        <include file="$(find target_localization)/launch/target_loc.launch" if="$(arg targetloc)">
            <!--<arg name="image_topic" value="/bebop2/camera_base/image_raw"/>-->
        </include>

        <!-- Target ball -->
        <group if="$(arg target)">
            <include file="$(find target_ball)/launch/target_moveit.launch" />
            <include file="$(find target_ball)/launch/target.launch">
                <arg name="x" default="$(arg target_x)"/>
                <arg name="y" default="$(arg target_y)"/>
                <arg name="z" default="$(arg target_z)"/>
            </include>
        </group>

    </group>
    <group ns="environment">
        <!-- CCTV cameras for filling database of drone, support for searching -->
        <include file="$(find cctv_system)/launch/cctv_system.launch" if="$(arg cctv)">
        </include>
    </group>
</launch>
    """.format(target, targetloc, cctv, target_position[0], target_position[1], target_position[2])
    with open(file_path, "w") as file:
        file.write(text)


def generate_environment_configuration_file(file_path, obstacles_file_name, world_folder_path, drone_position,
                                            target_position):
    text = """environment_configuration:
  map:
    frame: map
    obstacles_file: {}.json
    section_file: {}/section_file.json
    size: 10
  world:
    path: $(find pesat_resources)/worlds
    name: box_world
  properties:
    extra_localization: false
  watchdog:
    cameras_file: .
    camera_shot_topic: "/cctv/shot"
    camera_registration_service: "/cctv/registration"
    camera_update_topic: "/cctv/update"
    camera_notification_topic: "/cctv/notifications"
    camera_switch_topic: "/cctv/switch\"
  init_positions:
    target:
      x: {}
      y: {}
    drone:
      x: {}
      y: {}
      z: {}
    """.format(obstacles_file_name, world_folder_path, target_position[0], target_position[1], drone_position[0], drone_position[1], drone_position[2])
    with open(file_path, "w") as file:
        file.write(text)


def create_maps_searching():
    types = [[100, 0.1, -1, 2], [100, 0.3, 0, 2], [100, 0.1, 20, 2], [100, 0.3, 20, 2]]
    create_section_file = True
    for t in types:
        for i in range(t[3]):
            create_new_map(False, True, t[0], t[1], t[2], "f", "n")


def create_new_map(substitute_paths, create_section_file, size, built, height, position, dynamical_obstacle):
    t1 = datetime.datetime.now()
    name, drone_position, target_position = generate_random_map_from_submaps(size, built, height, position,
                                                                             dynamical_obstacle,
                                                                             "{}_{}_{}".format(t1.hour, t1.minute,
                                                                                               t1.second))
    print("Map with name {} was created.".format(name))
    world_path = "/home/patik/Diplomka/dp_ws/src/pesat_resources/worlds/"
    world_folder_path = world_path + name
    if substitute_paths:
        path = "/home/patik/Diplomka/dp_ws/src/pesat_core/launch/"
        ending = "drone.launch"
        abs_path = path + ending
        doubled_name = name + "/" + name
        generate_drone_launch_file(abs_path, doubled_name, drone_position)
        ending = "target.launch"
        abs_path = path + ending
        generate_target_launch_file(abs_path, target_position)
        obstacles_file_name = world_path + doubled_name
        path = "/home/patik/Diplomka/dp_ws/src/pesat_resources/config/"
        environment_configuration_file_name = "environment_configuration.yaml"
        abs_path = path + environment_configuration_file_name
        generate_environment_configuration_file(abs_path, obstacles_file_name, world_folder_path, drone_position,
                                                target_position)
    if create_section_file:
        from algorithms.src.algorithms import section_map_creator
        rvalue = section_map_creator.create_section_from_file(world_folder_path)
        if rvalue:
            print("Section file was created without errors.")
        else:
            print("During creating section file error occured. Check console for more information.")
    return name


if __name__ == "__main__":
    # generate_experiments_submaps()
    create_new_map(True, True, 100, 0.2, -1, "f", "n")
    # generate_random_map_by_origin_method(10, 0.1, 1.5, "f", "y", "32_15_10")
    # generate_random_maps_from_submaps()
    # grid_map = Building.grid_map()
    # dynamic_avoidance_map()
    pass
