import numpy as np
from collections import namedtuple
from geometry_msgs.msg import Point

class Math():
    Point = namedtuple("Point", ["x", "y"])
    Sector = namedtuple("Sector", ["center", "end", "start", "radius_squared"])
    Annulus = namedtuple("Annulus", ["center", "end", "start", "nearer_radius_squared", "farther_radius_squared"])

    @staticmethod
    def create_sector(position, angle, direction, radius):
        center = Math.Point(position[0], position[1])
        dir = Math.rotate_2d_vector(angle, [direction[0], direction[1]])
        start = Math.Point(dir[0], dir[1])
        dir = Math.rotate_2d_vector(-angle, [direction[0], direction[1]])
        end = Math.Point(dir[0], dir[1])
        radius_squared = radius * radius
        sector = Math.Sector(center, start, end, radius_squared)
        return sector

    @staticmethod
    def create_annulus(position, angle, direction, interval):
        center = Math.Point(position[0], position[1])
        dir = Math.rotate_2d_vector(angle, [direction[0], direction[1]])
        start = Math.Point(dir[0], dir[1])
        dir = Math.rotate_2d_vector(-angle, [direction[0], direction[1]])
        end = Math.Point(dir[0], dir[1])
        nearer_radius_squared = interval[0] * interval[0]
        farther_radius_squared = interval[1] * interval[1]
        annulus = Math.Annulus(center, start, end, nearer_radius_squared, farther_radius_squared)
        return annulus

    @staticmethod
    def rounding(value):
        try:
            value_sign = np.sign(value)
            abs_value = np.abs(value)
            diff = abs_value - int(abs_value)
            if diff < 0.25:
                r = 0.0
            elif diff < 0.75:
                r = 0.5
            else:
                r = 1
            return (int(abs_value) + r) * value_sign
        except:
            return abs_value * value_sign

    @staticmethod
    def vectorized_rounding(value):
        value_sign = np.sign(value)
        abs_value = np.abs(value)
        diff = abs_value - abs_value.astype(np.int32)
        r = np.array(len(diff))
        r[0.75 > r > 0.25] = 0.5
        r[1 > r > 0.75] = 1.0
        return abs_value.astype(np.int32) + r * value_sign

    @staticmethod
    def points_add(p1, p2):
        point = Point()
        point.x = p1.x + p2.x
        point.y = p1.y + p2.y
        point.z = p1.z + p2.z
        return point

    @staticmethod
    def points_mul(p1, p2):
        point = Point()
        point.x = p1.x * p2.x
        point.y = p1.y * p2.y
        point.z = p1.z * p2.z
        return point

    @staticmethod
    def point_constant_mul(p1, const):
        point = Point()
        point.x = p1.x * const
        point.y = p1.y * const
        point.z = p1.z * const
        return point

    @staticmethod
    def floor_rounding(value):
        rounded_value = Math.rounding(value)
        if rounded_value > value:
            return rounded_value - 0.5
        return rounded_value

    @staticmethod
    def ceiling_rounding(value):
        rounded_value = Math.rounding(value)
        if rounded_value < value:
            return rounded_value + 0.5
        return rounded_value

    @staticmethod
    def is_within_radius(p, radius_squared):
        return p.x * p.x + p.y * p.y <= radius_squared

    @staticmethod
    def is_within_interval(p, start, end):
        return start <= p.x * p.x + p.y * p.y <= end

    @staticmethod
    def is_inside_sector(p, sector):
        rel_p = Math.Point(p.x - sector.center.x, p.y - sector.center.y)
        return not Math.are_clockwise(sector.start, rel_p) and Math.are_clockwise(sector.end, rel_p) and \
               Math.is_within_radius(rel_p, sector.radius_squared)

    @staticmethod
    def is_inside_annulus(p, annulus):
        rel_p = Math.Point(p.x - annulus.center.x, p.y - annulus.center.y)
        return not Math.are_clockwise(annulus.start, rel_p) and Math.are_clockwise(annulus.end, rel_p) and \
               Math.is_within_interval(rel_p, annulus.nearer_radius_squared, annulus.farther_radius_squared)

    @staticmethod
    def cartesian_coor(angle, radius):
        return Math.Point(np.cos(angle) * radius, np.sin(angle) * radius)

    @staticmethod
    def correct_halfspace(point, rotated, axis, center, halfspace):
        centered_point = Math.Point(point.x - center.x, point.y - center.y)
        coordinates = Math.cartesian_coor(rotated, axis.x)
        if Math.are_clockwise(centered_point, coordinates) and halfspace:
            return True
        elif not Math.are_clockwise(centered_point, coordinates) and not halfspace:
            return True
        else:
            return False

    @staticmethod
    def are_clockwise(p1, p2):
        return -p1.x * p2.y + p1.y * p2.x > 0

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm < 0.0001:
            return v
        return v / norm

    @staticmethod
    def normalize_vectorized(v):
        norm = np.linalg.norm(v, axis=1)
        norm[norm < 0.0001] = 1
        return v / norm.reshape((-1, 1))

    @staticmethod
    def inside_ellipse(point, center, axis, rotation):
        cosa = np.cos(rotation)
        sina = np.sin(rotation)
        a_top = np.square(cosa * (point.x - center.x) + sina * (point.y - center.y))
        b_top = np.square(sina * (point.x - center.x) - cosa * (point.y - center.y))
        a_bottom = np.square(axis.x)
        b_bottom = np.square(axis.y)
        ellipse = (a_top / a_bottom) + (b_top / b_bottom)
        if ellipse <= 1:
            return True
        else:
            return False

    @staticmethod
    def two_lowest_indices(array):
        biggest_value = float("inf")
        second_biggest_value = float("inf")
        biggest_index = -1
        second_biggest_index = -1
        for i in range(len(array)):
            a = array[i]
            if a < biggest_index:
                second_biggest_value = biggest_value
                second_biggest_index = biggest_index
                biggest_value = a
                biggest_index = i
            if a < second_biggest_value:
                second_biggest_value = a
                second_biggest_index = i
        return np.array([biggest_index, second_biggest_index])

    @staticmethod
    def euclidian_distance(math_point_start, math_point_end):
        return np.sqrt(
            np.square(math_point_end.x - math_point_start.x) + np.square(math_point_start.y - math_point_end.y))

    @staticmethod
    def array_euclidian_distance(start, end):
        return np.sqrt(np.square(end[0] - start[0]) + np.square(start[1] - end[1]))

    @staticmethod
    def calculate_yaw_from_points(x_map, y_map, x_map_next, y_map_next):
        direction_x = x_map_next - x_map
        direction_y = y_map_next - y_map
        if abs(direction_x) < 1e-6 and abs(direction_y) < 1e-6:
            return 0
        # direction_length = np.sqrt(np.power(direction_x, 2) + np.power(direction_y, 2))
        # print("direction length", direction_length)
        # if abs(direction_length) > 1:
        #    return np.arccos(1 / direction_length)
        # else:
        #    return 0
        return np.arctan2(direction_y, direction_x)

    @staticmethod
    def calculate_yaw_from_points_vectors(x_map, y_map, x_map_next, y_map_next):
        directions = np.array([x_map_next - x_map, y_map_next - y_map]).reshape((-1, 2))
        sumed_directions = directions.sum(axis=1)
        directions[sumed_directions < 2 * 1e-6] = [0, 0]
        return np.arctan2(directions[:, 1], directions[:, 0])

    @staticmethod
    def rotate_2d_vector(theta, vector):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        vec = np.array(vector)
        vec = np.matmul(R, vec)
        return vec

    @staticmethod
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    @staticmethod
    def rotateX(vector, angle):
        return [vector[0],
                vector[1] * np.cos(angle) - vector[2] * np.sin(angle),
                vector[1] * np.sin(angle) + vector[2] * np.cos(angle)]

    @staticmethod
    def rotateZ(vector, angle):
        return [vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
                vector[0] * np.sin(angle) + vector[1] * np.cos(angle),
                vector[2]]

    @staticmethod
    def rotateY(vector, angle):
        return [vector[0] * np.cos(angle) + vector[2] * np.sin(angle),
                vector[1],
                -vector[0] * np.sin(angle) + vector[2] * np.cos(angle)]

    @staticmethod
    def is_near_enough(position_one, position_two, distance_limit):
        point_one = Math.Point(position_one[0], position_one[1])
        point_two = Math.Point(position_two[0], position_two[1])
        dist = Math.euclidian_distance(point_one, point_two)
        return dist <= distance_limit

    @staticmethod
    def uniform_circle(length, size):
        length = np.sqrt(np.random.uniform(0, length, size))
        angle = np.pi * np.random.uniform(0, 2, size)

        x = length * np.cos(angle)
        y = length * np.sin(angle)
        array = np.zeros((size, 2))
        array[:, 0] = x
        array[:, 1] = y
        return array