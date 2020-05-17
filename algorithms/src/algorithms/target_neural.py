#!/usr/bin/env python
import rospy
import tensorflow as tf
from helper_pkg.utils import Constants
import helper_pkg.PredictionManagement as pm



class PredictionNetwork(pm.PredictionAlgorithm):
    DRONE_POSITIONS = 8
    TARGET_POSITIONS = 8
    INPUT_TYPE = Constants.InputOutputParameterType.map
    OUTPUT_TYPE = Constants.InputOutputParameterType.map
    MAP_PRESENCE = Constants.MapPresenceParameter.yes
    SIZE = 640

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        pose = Pose()
        return pose

    def state_variables(self,data, drone_positions, target_positions, map, **kwargs):
        return {}
