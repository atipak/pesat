import numpy as np
from collections import deque
import itertools
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import PointField, PointCloud2
from pesat_utils.pesat_math import Math

class DataStructures():
    class SliceableDeque(deque):
        def __getitem__(self, index):
            if isinstance(index, slice):
                return type(self)(itertools.islice(self, index.start,
                                                   index.stop, index.step))
            return deque.__getitem__(self, index)

    @staticmethod
    def point_cloud_name(size=4):
        names = ["x", "y", "z", "roll", "pitch", "yaw", "predecessor", "weight"]
        fields = [PointField(names[i], i * size, PointField.FLOAT32, 1) for i in range(len(names))]
        return fields

    @staticmethod
    def array_to_pointcloud2(data, stamp=None, frame_id=None):
        size = 4
        msg = PointCloud2()
        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        msg.fields = DataStructures.point_cloud_name()
        msg.height = 1  # data.shape[0]
        msg.width = data.shape[0]
        msg.is_bigendian = False
        msg.point_step = size * data.shape[1]
        msg.row_step = size * data.shape[0] * data.shape[1]
        msg.is_dense = int(np.isfinite(data).all())
        msg.data = np.asarray(data, np.float32).tostring()
        return msg

    @staticmethod
    def pointcloud2_to_array(cloud_msg):
        size = 4
        if cloud_msg is None:
            return None
        cloud_arr = np.fromstring(cloud_msg.data, np.float32)
        return np.reshape(cloud_arr, (cloud_msg.width, int(cloud_msg.point_step / size)))

    @staticmethod
    def pointcloud2_to_pose_stamped(cloud_msg):
        array = DataStructures.pointcloud2_to_array(cloud_msg)
        pose = DataStructures.array_to_pose(array)
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header = cloud_msg.header
        return pose_stamped

    @staticmethod
    def array_to_image(data, width, height, step, center=(0, 0)):
        buckets = {}
        rounded_data = np.zeros((len(data), 2))
        rounded_center = (Math.rounding(center[0]) / step, Math.rounding(center[1]) / step)
        rounded_data[:, 0] = np.clip(np.round(data[:, 0] / step) + rounded_center[0] - width / 2, 0, width - 1).astype(
            np.int32)
        rounded_data[:, 1] = np.clip(np.round(data[:, 1] / step) + rounded_center[1] - height / 2, 0,
                                     height - 1).astype(np.int32)
        image = np.zeros((width, height))
        values, counts = np.unique(rounded_data, return_counts=True, axis=0)
        image[values[:, 0].astype(np.int32), values[:, 1].astype(np.int32)] = counts / len(data)
        return image

    @staticmethod
    def array_to_pose(data):
        if len(data) == 0:
            return None
        buckets = {}
        max_coor = (Math.rounding(data[0][0]), Math.rounding(data[0][1]), Math.rounding(data[0][2]))
        max_value = 1
        for d in data:
            rounded_x = Math.rounding(d[0])
            rounded_y = Math.rounding(d[1])
            rounded_z = Math.rounding(d[2])
            if rounded_x not in buckets:
                buckets[rounded_x] = {}
            if rounded_y not in buckets[rounded_x]:
                buckets[rounded_x][rounded_y] = {}
            if rounded_z not in buckets[rounded_x][rounded_y]:
                buckets[rounded_x][rounded_y][rounded_z] = []
            buckets[rounded_x][rounded_y][rounded_z].append(d)
            if len(buckets[rounded_x][rounded_y][rounded_z]) > max_value:
                max_coor = (rounded_x, rounded_y, rounded_z)
                max_value = len(buckets[rounded_x][rounded_y][rounded_z])
        avg_result = np.average(buckets[max_coor[0]][max_coor[1]][max_coor[2]], axis=0)
        pose = Pose()
        pose.position.x = avg_result[0]
        pose.position.y = avg_result[1]
        pose.position.z = avg_result[2]
        pose.orientation.x = avg_result[3]
        pose.orientation.y = avg_result[4]
        pose.orientation.z = avg_result[5]
        return pose

    @staticmethod
    def image_to_array(image, samples_count, step, noise_position_std, noise_orientation_std, center=(0, 0)):
        new_samples = np.zeros((samples_count, len(DataStructures.point_cloud_name())))
        coordinates = np.reshape(np.array(np.meshgrid(np.arange(0, image.shape[0]), np.arange(0, image.shape[1]))).T,
                                 (image.shape[0] * image.shape[1], 2)) * step + np.array(center) - (
                              np.array(image.shape) / 2) * step
        samples = np.reshape(image * samples_count, (image.shape[0] * image.shape[1])).astype(np.int32)
        temp_count = np.sum(samples)
        new_samples[:temp_count, :2] = np.repeat(coordinates, samples, axis=0)
        new_samples[temp_count:, :2] = coordinates[np.random.choice(len(coordinates), samples_count - temp_count)]
        new_samples[:, :3] += np.random.normal(0, noise_position_std, (samples_count, 3))
        new_samples[:, 3:6] += np.random.normal(0, noise_orientation_std, (samples_count, 3))
        new_samples[:, 7] = 1
        return np.array(new_samples)

    @staticmethod
    def pose_to_array(pose, samples_count, noise_position_std, noise_orientation_std):
        position_noise = np.random.normal(0, noise_position_std, (samples_count, 3))
        orientation_noise = np.random.normal(0, noise_orientation_std, (samples_count, 3))
        new_samples = []
        for i in range(samples_count):
            new_samples.append([pose.position.x + position_noise[i][0],
                                pose.position.y + position_noise[i][1],
                                pose.position.z + position_noise[i][2],
                                pose.orientation.x + orientation_noise[i][0],
                                pose.orientation.y + orientation_noise[i][1],
                                pose.orientation.z + orientation_noise[i][2],
                                0, 1])

        return np.array(new_samples)

    @staticmethod
    def image_to_particles(image):
        particles = []
        # [left up corner x, left up corner y, 0, width, height, weight, 0]
        _, _, index = DataStructures.recursive_decomposition(image, 0, 0, image.shape[0] - 1, image.shape[1] - 1,
                                                             particles, 0)
        return np.array(particles)

    @staticmethod
    def recursive_decomposition(image, start_x, start_y, end_x, end_y, particles, index):
        if start_x > end_x or start_y > end_y:
            return 0, False, 0
        if (start_x, start_y) == (end_x, end_y):
            if image[start_x, start_y] > 0.00000001:
                return image[start_x, start_y], True, 0
            else:
                return 0, True, 0
        else:
            x_diff = end_x - start_x
            y_diff = end_y - start_y
            width_half = int(x_diff / 2)
            height_half = int(y_diff / 2)
            i = 1
            if height_half > 0:
                i += 1
            if width_half > 0:
                i += 1
            if width_half > 0 and height_half > 0:
                i += 1
            val = np.zeros(i)
            i = 0
            # left up
            val[i], same1, index1 = DataStructures.recursive_decomposition(image, start_x, start_y,
                                                                           start_x + width_half,
                                                                           start_y + height_half, particles, index)
            # left bottom
            index += index1
            if height_half > 0:
                # print("original", start_x, start_y, end_x, end_y)
                # print("added", width_half, height_half)
                # print("new", start_x, start_x + width_half, start_y + height_half + 1, end_y)
                i += 1
                val[i], same2, index1 = DataStructures.recursive_decomposition(image, start_x,
                                                                               start_y + height_half + 1,
                                                                               start_x + width_half, end_y, particles,
                                                                               index)
            else:
                rec2, same2, index1 = 0, False, 0
            # right up
            index += index1
            if width_half > 0:
                i += 1
                val[i], same3, index1 = DataStructures.recursive_decomposition(image, start_x + width_half + 1, start_y,
                                                                               end_x, start_y + height_half, particles,
                                                                               index)
            else:
                rec3, same3, index1 = 0, False, 0
            # right bottom
            index += index1
            if width_half > 0 and height_half > 0:
                i += 1
                val[i], same4, index1 = DataStructures.recursive_decomposition(image, start_x + width_half + 1,
                                                                               start_y + height_half + 1, end_x,
                                                                               end_y, particles, index)
            else:
                rec4, same4, index1 = 0, False, 0
            index += index1
            var = np.var(val)
            if not same1 or not same2 or not same3 or not same4 or var > 0.00005:
                i = 0
                index1 = 0
                if same1:
                    if val[i] != 0:
                        index1 += 1
                        particles.append([start_x, start_y, 0, width_half, height_half, val[i], 0])
                if same2:
                    i += 1
                    if val[i] != 0:
                        index1 += 1
                        particles.append(
                            [start_x, start_y + height_half + 1, 0, width_half, y_diff - height_half, val[i], 0])
                if same3:
                    i += 1
                    if val[i] != 0:
                        index1 += 1
                        particles.append(
                            [start_x + width_half + 1, start_y, 0, x_diff - width_half, height_half, val[i], 0])
                if same4:
                    i += 1
                    if val[i] != 0:
                        index1 += 1
                        particles.append(
                            [start_x + width_half + 1, start_y + height_half + 1, 0, x_diff - width_half,
                             y_diff - height_half, val[i], 0])
                return val[0], False, index + index1
            return val[0], True, index

    @staticmethod
    def copy_pose(pose):
        p = Pose()
        p.position.x = pose.position.x
        p.position.y = pose.position.y
        p.position.z = pose.position.z
        p.orientation.x = pose.orientation.x
        p.orientation.y = pose.orientation.y
        p.orientation.z = pose.orientation.z
        p.orientation.w = pose.orientation.w
        return p