#!/usr/bin/env python
from abc import abstractmethod

import rospy
import json
import os
import sys
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import AttachedCollisionObject, PlanningScene, PositionConstraint, Constraints, JointConstraint
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

    def load_obstacles1(self):
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
                    print("loaded obstacle with pose:", pose)
                    primitive = SolidPrimitive()
                    primitive.type = primitive.BOX
                    primitive.dimensions = [obstacle["x_size"], obstacle["y_size"], obstacle["z_size"]]
                    attached_object.object.primitives.append(primitive)
                    attached_object.object.primitive_poses.append(pose)
                planning_scene = PlanningScene()
                planning_scene.world.collision_objects.append(attached_object.object)
                planning_scene.is_diff = True
                self.planning_scene_diff_publisher.publish(planning_scene)

    def load_obstacles(self):
        if self.obstacles is None:
            if os.path.isfile(self.obstacles_file):
                with open(self.obstacles_file, "r") as file:
                    f = json.load(file)
                    self.obstacles = f["objects"]["statical"]
                    self.world = f["world"]
                    self.create_outside_world(self.world["width"], self.world["height"], self.world["maximal_height"])
                index = 0
                rospy.sleep(2)
                for obstacle in self.obstacles:
                    if self.add_box(obstacle, index):
                        print("Obstacle with index " + str(index) + " was added.")
                    else:
                        print("Obstacle with index " + str(index) + " wasn't added.")
                    index += 1

    def create_outside_world(self, width, length, height):
        #  name, face, x_pose, y_pose, z_pose, r_orient, p_orient, y_orient, x_size,
        #                       y_size, z_size, material_block
        box_height = 1.0
        # front
        self.obstacles.append({"x_pose": length / 2 + box_height / 2.0, "y_pose": 0, "z_pose": height / 2,
                               "r_orient": 0.0, "p_orient": 0.0, "y_orient": 0.0, "x_size": box_height, "y_size": width,
                               "z_size": height})
        # back
        self.obstacles.append({"x_pose": -length / 2 - box_height / 2.0, "y_pose": 0, "z_pose": height / 2,
                               "r_orient": 0.0, "p_orient": 0.0, "y_orient": 0.0, "x_size": box_height, "y_size": width,
                               "z_size": height})
        # left
        self.obstacles.append({"x_pose": 0, "y_pose": -width / 2 - box_height / 2.0, "z_pose": height / 2,
                               "r_orient": 0.0, "p_orient": 0.0, "y_orient": 0.0, "x_size": length,
                               "y_size": box_height, "z_size": height})
        # right
        self.obstacles.append({"x_pose": 0, "y_pose": width / 2 + box_height / 2.0, "z_pose": height / 2,
                               "r_orient": 0.0, "p_orient": 0.0, "y_orient": 0.0, "x_size": length,
                               "y_size": box_height, "z_size": height})
        # bottom
        self.obstacles.append({"x_pose": 0, "y_pose": 0, "z_pose": -box_height / 2,
                               "r_orient": 0.0, "p_orient": 0.0, "y_orient": 0.0, "x_size": length, "y_size": width,
                               "z_size": box_height})
        # up
        self.obstacles.append({"x_pose": 0, "y_pose": 0, "z_pose": height + box_height / 2,
                               "r_orient": 0.0, "p_orient": 0.0, "y_orient": 0.0, "x_size": length, "y_size": width,
                               "z_size": box_height})

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
            diff_x = c_state.multi_dof_joint_state.transforms[0].translation.x - end_position.position.x
            diff_y = c_state.multi_dof_joint_state.transforms[0].translation.y - end_position.position.y
            diff_z = c_state.multi_dof_joint_state.transforms[0].translation.z - end_position.position.z
            self._workspace_size = abs(diff_x) + abs(diff_y) + abs(diff_z) + 1
            print(self._workspace_size)
            self.move_group.set_workspace(
                [-self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.x,
                 -self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.y,
                 -self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.z,
                 self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.x,
                 self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.y,
                 self._workspace_size + c_state.multi_dof_joint_state.transforms[0].translation.z])
            target_joints = self.get_target_joints(end_position)
            self.move_group.set_joint_value_target(target_joints)
            self.move_group.set_start_state(c_state)
            explicit_quat = [c_state.multi_dof_joint_state.transforms[0].rotation.x,
                             c_state.multi_dof_joint_state.transforms[0].rotation.y,
                             c_state.multi_dof_joint_state.transforms[0].rotation.z,
                             c_state.multi_dof_joint_state.transforms[0].rotation.w]
            (_, _, yaw) = tf.transformations.euler_from_quaternion(explicit_quat)
            rospy.loginfo("Start configuration x: {}, y: {}, z: {}, yaw: {}".format(
                c_state.multi_dof_joint_state.transforms[0].translation.x,
                c_state.multi_dof_joint_state.transforms[0].translation.y,
                c_state.multi_dof_joint_state.transforms[0].translation.z,
                yaw))
            rospy.loginfo("End configuration x: {}, y: {}, z: {}, yaw: {}".format(end_position.position.x,
                                                                                  end_position.position.y,
                                                                                  end_position.position.z,
                                                                                  end_position.orientation.z))
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

    def add_box(self, obstacle, i, timeout=4):
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
        box_pose = PoseStamped()
        box_pose.header.frame_id = self.robot.get_planning_frame()
        box_pose.pose = pose
        box_name = "obstacle_" + str(i)
        expand_constant = 0.5
        self.scene.add_box(box_name, box_pose,
                           size=(obstacle["x_size"] + expand_constant, obstacle["y_size"] + expand_constant,
                                 obstacle["z_size"] + expand_constant))
        return self.wait_for_state_update(box_name, box_is_known=True, timeout=timeout)

    def wait_for_state_update(self, box_name, box_is_known=False, box_is_attached=False, timeout=4):
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = self.scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = box_name in self.scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False

    def open_world_box(self):
        up_side_name = "obstacle_" + str(len(self.obstacles) - 1)
        bottom_side_name = "obstacle_" + str(len(self.obstacles) - 2)
        self.scene.remove_world_object(up_side_name)
        self.scene.remove_world_object(bottom_side_name)

    def add_constraints(self):
        cs = Constraints()
        jc = JointConstraint()
        c = PositionConstraint()
        c.link_name = self.baselink_frame
        c.header.frame_id = self.map_frame

        self.move_group.set_path_constraints()
