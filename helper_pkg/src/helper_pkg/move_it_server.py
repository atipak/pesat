#!/usr/bin/env python
from abc import abstractmethod

import rospy
import json
import os
import sys
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
import moveit_commander
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity


class MoveitServer(object):
    def __init__(self, configuration):
        super(MoveitServer, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        environment_configuration = rospy.get_param("environment_configuration")
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.jump_threshold = configuration["move_it"]["jump_threshold"]
        self.eef_step = configuration["move_it"]["eef_step"]
        self._workspace_size = 20
        self._group_name = configuration["move_it"]["group_name"]
        while True:
            try:
                self.move_group = moveit_commander.MoveGroupCommander(self._group_name)
                break
            except RuntimeError as re:
                rospy.loginfo("RUNTIME ERROR: " + str(re))
                rospy.sleep(2)
        self.obstacles = None
        self.planning_scene_diff_publisher = rospy.Publisher(configuration["move_it"]['planning_scene_topic'],
                                                             PlanningScene,
                                                             queue_size=1)
        self.obstacles_file = environment_configuration["map"]["obstacles_file"]
        self.load_obstacles()
        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self.baselink_frame = configuration["properties"]["base_link"]
        self.map_frame = environment_configuration["map"]["frame"]
        self.sv_srv = rospy.ServiceProxy('check_state_validity', GetStateValidity)
        self.sv_srv.wait_for_service()
        self.rs = self.robot.get_current_state()

    def is_admissible(self, point):
        gsvr = GetStateValidityRequest()
        c_state = self.robot.get_current_state()
        c_state.multi_dof_joint_state.transforms[0].translation.x = point.x
        c_state.multi_dof_joint_state.transforms[0].translation.y = point.y
        c_state.multi_dof_joint_state.transforms[0].translation.z = point.z
        gsvr.robot_state = c_state
        gsvr.group_name = self._group_name
        try:
            result = self.sv_srv.call(gsvr)
            if result.valid:
                return True
            else:
                return False
        except Exception as exc:
            print("Moveit server:" + str(exc))
            return False

    def load_obstacles(self):
        if self.obstacles is None:
            if os.path.isfile(self.obstacles_file):
                with open(self.obstacles_file, "r") as file:
                    self.obstacles = json.load(file)["objects"]
                attached_object = AttachedCollisionObject()
                attached_object.object.header.frame_id = "map"
                attached_object.object.id = "obstacles"
                for obstacle in self.obstacles:
                    pose = Pose()
                    pose.position.x = obstacle["x_pose"]
                    pose.position.y = obstacle["y_pose"]
                    pose.position.z = obstacle["z_pose"]
                    qt = tf.transformations.quaternion_from_euler(obstacle["r_orient"], obstacle["p_orient"],
                                                                  obstacle["y_orient"])
                    pose.orientation.x = qt[0]
                    pose.orientation.y = qt[1]
                    pose.orientation.z = qt[2]
                    pose.orientation.w = qt[3]
                    primitive = SolidPrimitive()
                    primitive.type = primitive.BOX
                    primitive.dimensions = [obstacle["x_size"], obstacle["y_size"], obstacle["z_size"]]
                    attached_object.object.primitives.append(primitive)
                    attached_object.object.primitive_poses.append(pose)
                planning_scene = PlanningScene()
                planning_scene.world.collision_objects.append(attached_object.object)
                planning_scene.is_diff = True
                self.planning_scene_diff_publisher.publish(planning_scene)

    def get_plan_from_poses(self, start_position, end_position):
        if self.compare_poses(start_position, end_position):
            return [end_position]
        pose_plan = []
        try:
            c_state = self.robot.get_current_state()
            _ = self._tfBuffer.lookup_transform(self.map_frame, self.baselink_frame, rospy.Time())
            if start_position is not None:
                c_state.multi_dof_joint_state.transforms[0].translation.x = start_position.position.x
                c_state.multi_dof_joint_state.transforms[0].translation.y = start_position.position.y
                c_state.multi_dof_joint_state.transforms[0].translation.z = start_position.position.z
                q = tf.transformations.quaternion_from_euler(0.0, 0.0, start_position.orientation.z)
                c_state.multi_dof_joint_state.transforms[0].rotation.x = q[0]
                c_state.multi_dof_joint_state.transforms[0].rotation.y = q[1]
                c_state.multi_dof_joint_state.transforms[0].rotation.z = q[2]
                c_state.multi_dof_joint_state.transforms[0].rotation.w = q[3]
            self.move_group.set_workspace(
                [-self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.x,
                 -self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.y,
                 -self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.z,
                 self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.x,
                 self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.y,
                 self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.x +
                 c_state.multi_dof_joint_state.transforms[0].translation.z])
            target_joints = self.get_target_joints(end_position)
            self.move_group.set_joint_value_target(target_joints)
            self.move_group.set_start_state(c_state)
            plan = self.move_group.plan()
            for point in plan.multi_dof_joint_trajectory.points:
                for transform in point.transforms:
                    pose = Pose()
                    pose.position.x = transform.translation.x
                    pose.position.y = transform.translation.y
                    pose.position.z = transform.translation.z
                    pose.orientation.x = transform.rotation.x
                    pose.orientation.y = transform.rotation.y
                    pose.orientation.z = transform.rotation.z
                    pose.orientation.w = transform.rotation.w
                    pose_plan.append(pose)
            self.move_group.clear_pose_targets()
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as excs:
            rospy.loginfo(str(excs))
        return pose_plan

    @abstractmethod
    def get_target_joints(self, end_position):
        pass

    def compare_poses(self, pose1, pose2):
        return np.allclose(
            [pose1.position.x, pose1.position.y, pose1.position.z, pose1.orientation.x, pose1.orientation.y,
             pose1.orientation.z],
            [pose2.position.x, pose2.position.y, pose2.position.z, pose2.orientation.x, pose2.orientation.y,
             pose2.orientation.z])
