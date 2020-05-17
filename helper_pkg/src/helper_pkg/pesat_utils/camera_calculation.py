import numpy as np
from scipy.stats import multivariate_normal
from pesat_utils.data_structures import DataStructures
from pesat_utils.pesat_math import Math


class CameraCalculation():

    @staticmethod
    def generate_samples_object_in_fov(mean, location_accuracy, coordinates, resolution):
        conv = [[location_accuracy * resolution, 0],
                [0, location_accuracy * resolution]]
        samples = np.random.multivariate_normal(mean, conv, 100)
        samples_in_fov = []
        for sample in samples:
            if sample[0] in coordinates and sample[1] in coordinates[sample[0]]:
                samples_in_fov.append(sample)
        return samples_in_fov

    @staticmethod
    def generate_particles_object_in_fov(mean, location_accuracy, coordinates, resolution):
        conv = [[location_accuracy * resolution, 0],
                [0, location_accuracy * resolution]]
        mvn = multivariate_normal(mean, conv)
        weights = mvn.pdf(coordinates)
        mask = weights > 0.001
        right_coordinates = coordinates[mask, :]
        right_weights = weights[mask]
        weights_sum = np.sum(right_weights)
        particles = np.zeros((np.count_nonzero(mask), 3))
        particles[:, :2] = right_coordinates
        particles[:, 2] = right_weights / weights_sum
        print(len(right_coordinates), weights_sum)
        return particles

    @staticmethod
    def generate_particles_object_out_of_fov(coordinates):
        weight = len(coordinates)
        particles = np.zeros((len(coordinates), 3))
        particles[:, :2] = coordinates
        particles[:, 2] = 1.0 / weight
        return particles

    @staticmethod
    def particles_to_image(array, width, height):
        image = np.zeros((width, height))
        image[array[:, 1].astype(np.int32), array[:, 0].astype(np.int32)] = array[:, 2]
        s = np.sum(array[:, 2])
        if s > 0:
            image /= np.sum(array[:, 2])
        return image

    @staticmethod
    def generate_samples_object_out_of_fov(coordinates, samples_count_per_pixel=3):
        samples = []
        for x_coor in coordinates:
            for y_coor in coordinates[x_coor]:
                for _ in range(samples_count_per_pixel):
                    samples.append([x_coor, y_coor])
        return np.array(samples)

    @staticmethod
    def generate_position_samples(samples, velocity, var_velocity):
        position_samples = np.zeros((len(samples), len(DataStructures.point_cloud_name())))
        for i in range(len(samples)):
            position_samples[i, 0] = samples[i, 0]
            position_samples[i, 1] = samples[i, 1]
            yaw = Math.calculate_yaw_from_points(0, 0, velocity[0], velocity[1])
            position_samples[i, 5] = yaw + np.random.uniform(-var_velocity, var_velocity, 1)[0]
            position_samples[i, 7] = 1
        return position_samples

    @staticmethod
    def get_visible_coordinates(world_map, position, interval, hfov, camera_step):
        end_angle = position.orientation.z + hfov / 2.0
        start_angle = position.orientation.z - hfov / 2.0
        step_count = (interval[1] - interval[0]) / camera_step + 1
        coordinates = CameraCalculation.get_coordinates_of_fov(world_map, position, end_angle, start_angle, step_count,
                                                               interval, np.deg2rad(0.5))
        return coordinates, CameraCalculation.coordinates_length(coordinates)

    @staticmethod
    def get_coordinates_of_fov(world_map, start_position, start_angle, end_angle, step_count, interval, step):
        coordinates = {}
        pixel_size = 1 / world_map.resolution
        while start_angle > end_angle:
            v = Math.cartesian_coor(start_angle, interval[1])
            norm_v = Math.normalize(np.array([v.x, v.y]))
            for i in range(step_count):
                stepped_v = (norm_v * i + interval[0])
                map_point = world_map.map_point_from_real_coordinates(start_position.position.x + stepped_v[0],
                                                                      start_position.position.y + stepped_v[1], 0)
                if world_map.is_free(map_point, "target"):
                    # stepped_v = (stepped_v / pixel_size).astype(int)
                    stepped_v = [map_point.x, map_point.y]
                    if stepped_v[0] not in coordinates:
                        coordinates[stepped_v[0]] = {}
                    if stepped_v[1] not in coordinates[stepped_v[0]]:
                        coordinates[stepped_v[0]][stepped_v[1]] = True
                else:
                    break
            start_angle -= step
        return coordinates

    @staticmethod
    def coordinates_length(coordinates):
        l = 0
        for key in coordinates:
            l += len(coordinates[key])
        return l

    @staticmethod
    def write_coordinates(coordinates, target_array):
        for x_coor in coordinates:
            for y_coor in coordinates[x_coor]:
                # map_point = world_map.map_point_from_real_coordinates(x_coor, y_coor, 0)
                target_array[y_coor, x_coor] = 1
        return target_array

    @staticmethod
    def combine_coordinates(coordinates_collection):
        coordinates = coordinates_collection[0]
        for i in range(1, len(coordinates_collection)):
            for coor_x in coordinates_collection[i]:
                for coor_y in coordinates_collection[i][coor_x]:
                    if coor_x not in coordinates:
                        coordinates[coor_x] = {}
                    if coor_y not in coordinates[coor_x]:
                        coordinates[coor_x][coor_y] = True
        return coordinates