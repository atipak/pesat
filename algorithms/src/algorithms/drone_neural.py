#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Pose
from helper_pkg.utils import Constants
import helper_pkg.PredictionManagement as pm


class DeepPrediction(pm.PredictionAlgorithm):
    DRONE_POSITIONS = 7
    TARGET_POSITIONS = 8
    INPUT_TYPE = Constants.InputOutputParameterType.map
    OUTPUT_TYPE = Constants.InputOutputParameterType.map
    MAP_PRESENCE = Constants.MapPresenceParameter.yes

    def __init__(self, threads, seed=42):
        super(DeepPrediction, self).__init__()
        self._prediction_algorithm_name = "Deep prediction"

    def pose_from_parameters(self, drone_positions, target_positions, map, **kwargs):
        p = Pose()
        return p

    def state_variables(self,data, drone_positions, target_positions, map, **kwargs):
        return {}
