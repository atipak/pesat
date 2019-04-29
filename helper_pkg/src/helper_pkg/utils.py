import numpy as np
from collections import namedtuple, deque
import itertools

class DataStructures():
    class SliceableDeque(deque):
        def __getitem__(self, index):
            if isinstance(index, slice):
                return type(self)(itertools.islice(self, index.start,
                                                   index.stop, index.step))
            return deque.__getitem__(self, index)

class Math():
    Point = namedtuple("Point", ["x", "y"])

    @staticmethod
    def rounding(value):
        diff = value - int(value)
        if diff < 0.25:
            r = 0.0
        elif diff < 0.75:
            r = 0.5
        else:
            r = 1
        return int(value) + r

    @staticmethod
    def is_within_radius(p, radius_squared):
        return p.x * p.x + p.y * p.y <= radius_squared

    @staticmethod
    def is_inside_sector(p, camera):
        rel_p = Math.Point(p.x - camera.center.x, p.y - camera.center.y)
        return not Math.are_clockwise(camera.start, rel_p) and Math.are_clockwise(camera.end, rel_p) and \
               Math.is_within_radius(rel_p, camera.radius_squared)

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


class Map():
    def __init__(self, array):
        self._map = array
        self._width = self._map.shape[0]
        self._height = self._map.shape[1]

    @property
    def map(self):
        return self._map

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def is_free(self, x, y, min_value=255):
        if not 0 <= x <= self.width or not 0 <= y <= self.height:
            return False
        return self._map[x, y] >= min_value

    def random_place_target_drone(self, min_distance, max_distance):
        while True:
            x = np.random.random_integers(0, self._width)
            y = np.random.random_integers(0, self._height)
            if self.is_free(x, y):
                i, j = self.free_place_in_field(x, y, min_distance, max_distance)
                if i is not None:
                    return x, y, i, j

    def free_place_in_field(self, x, y, min_distance, max_distance):
        min_x = x - max_distance
        if min_x < 0:
            min_x = 0
        max_x = x + max_distance
        if max_x > self._width:
            max_x = self._width
        min_y = y - max_distance
        if min_y < 0:
            min_y = 0
        max_y = y + max_distance
        if max_y > self._height:
            max_y = self._height

        min_xs = x - min_distance
        if min_xs < 0:
            min_xs = 0
        max_xs = x + min_distance
        if max_xs > self._width:
            max_xs = self._width
        min_ys = y - min_distance
        if min_ys < 0:
            min_ys = 0
        max_ys = y + min_distance
        if max_ys > self._height:
            max_ys = self._height
        range_x = range(min_x, min_xs) + range(max_xs, max_x)
        range_y = range(min_y, min_ys) + range(max_ys, max_y)
        for i in range_x:
            for j in range_y:
                if self.is_free(i, j):
                    return i, j
        return None, None

    def crop_map(self, center, width, height):
        center = [center[0], center[1]]
        if width > self.width:
            width = self.width
        if height > self.height:
            height = self.height
        half_width = width / 2
        half_height = height / 2
        if center[0] - half_width < 0:
            center[0] = half_width
        if center[0] + half_width > self.width:
            center[0] = self.width - half_width
        if center[1] - half_height < 0:
            center[1] = half_height
        if center[1] + half_height > self.height:
            center[1] = self.height - half_height
        return self.map[center[0] - half_width:center[0] + half_width, center[1] - half_height:center[1] + half_height]
