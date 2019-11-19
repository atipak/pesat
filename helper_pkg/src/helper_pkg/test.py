from scipy.spatial.transform import Rotation as R
import numpy as np
from geometry_msgs.msg import Pose

pitch = np.array([np.pi/8])
yaw = np.array([np.pi/2])
pitch_yaw_def = np.array([pitch[0], yaw[0]])
yaw_pitch_def = np.array([yaw[0], pitch[0]])
ry = R.from_euler('y', pitch)
rz = R.from_euler('z', yaw)
rY = R.from_euler('Y', pitch)
rZ = R.from_euler('Z', yaw)
rZY = R.from_euler('ZY', yaw_pitch_def)
rYZ = R.from_euler('YZ', pitch_yaw_def)
rzy = R.from_euler('zy', yaw_pitch_def)
ryz = R.from_euler('yz', pitch_yaw_def)
only_z = rz.apply(np.array([1,0,0]))
only_y = ry.apply(np.array([1,0,0]))
y_z = rz.apply(only_y)
z_y = ry.apply(only_z)
only_yaw = rZ.apply(np.array([1,0,0]))
only_pitch = rY.apply(np.array([1,0,0]))
pitch_yaw = rZ.apply(only_pitch)
yaw_pitch = rY.apply(only_yaw)
rzy_res = rzy.apply(np.array([1,0,0]))
ryz_res = ryz.apply(np.array([1,0,0]))
rYZ_res = rYZ.apply(np.array([1,0,0]))
rZY_res = rZY.apply(np.array([1,0,0]))
print(only_z, only_y, y_z, z_y)
print(only_yaw, only_pitch, pitch_yaw, yaw_pitch)
print(rYZ_res, rzy_res)
print(ryz_res, rZY_res)

import time
from collections import deque
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint

def vectorized_rounding(value):
    value_sign = np.sign(value)
    abs_value = np.abs(value)
    diff = abs_value - abs_value.astype(np.int32)
    r = np.array(len(diff))
    r[0.75 > r > 0.25] = 0.5
    r[1 > r > 0.75] = 1.0
    return abs_value.astype(np.int32) + r * value_sign

def append_to_collection( x, y, collection):
    if x not in collection:
        collection[x] = {}
    if y not in collection[x]:
        collection[x][y] = True

def rectangle_ray_tracing_3d(position, vfov, hfov, max_range):
    # start_time = time.time()
    height = 0.2
    h_angles = [position.orientation.z - hfov / 2.0, position.orientation.z + hfov / 2.0]
    v_angles = np.clip([position.orientation.y - vfov / 2.0, position.orientation.y + vfov / 2.0],
                       (0, 0), (np.pi / 2, np.pi / 2))
    angles = [[v_angles[0], h_angles[0]],  # bottom left
              [v_angles[1], h_angles[0]],  # upper left
              [v_angles[1], h_angles[1]],  # upper right
              [v_angles[0], h_angles[1]]]  # bottom right
    r = R.from_euler('yz', angles)
    start_point = np.array([position.position.x, position.position.y, position.position.z])
    # print("start_position", start_point)
    vectors = r.apply(np.array([1, 0, 0]))
    t = np.abs(height + (-position.position.z / vectors[:, 2]))
    t[t > max_range] = max_range
    xyz = np.empty((4, 2))
    xyz[:, 0] = t * vectors[:, 0] + position.position.x
    xyz[:, 1] = t * vectors[:, 1] + position.position.y
    # print("t", t)
    # print("vectors", vectors)
    # print("angles", angles)
    # print("xyz", xyz)
    polygon = ShapelyPolygon(xyz)
    centroid = vectorized_rounding(polygon.centroid) + [0.25, 0.25]
    queue = deque()
    queue.append(centroid)
    coordinates = {}
    visited = {}
    # print("bb", time.time() - start_time)
    # start_time = time.time()
    ray = np.empty((2, 3), dtype=np.float32)
    ray[0, :] = start_point
    # a = []
    # b = []
    # d = []
    c = 0
    t = np.zeros((2000,2000))
    for i in range(1, 10000):
        item = np.array([2.0 + i, 2.0])
        point = ShapelyPoint(item)
        if polygon.contains(point):
            c +=1
        if item[0] + 0.5 not in visited or item[1] not in visited[item[0] + 0.5]:
            queue.append(np.array([item[0] + 0.5, item[1]]))
        if item[0] - 0.5 not in visited or item[1] not in visited[item[0] - 0.5]:
            queue.append(np.array([item[0] - 0.5, item[1]]))
        if item[0] in visited and item[1] + 0.5 not in visited[item[0]]:
            queue.append(np.array([item[0], item[1] + 0.5]))
        if item[0] in visited and item[1] - 0.5 not in visited[item[0]]:
            queue.append(np.array([item[0], item[1] - 0.5]))
    return coordinates

p = Pose()
p.position.z = 3
item = np.array([2.0,2.0])
start = time.time()
for _ in range(1,10000):
    point = ShapelyPoint(item)
#tt = rectangle_ray_tracing_3d(p, np.pi / 4, np.pi / 1.5, 50)
print("t", time.time() - start)
