#!/usr/bin/env python
import rospy
import tf
from pesat_msgs.srv import PointFloat, PosePlan
from helper_pkg.move_it_server import MoveitServer

class DroneMoveit(MoveitServer):
    def __init__(self):
        rospy.init_node('map_planning_server', anonymous=False)
        drone_configuration = rospy.get_param("drone_configuration")
        super(DroneMoveit, self).__init__(drone_configuration)
        self._srv_point_height = rospy.Service(drone_configuration["map_server"]["point_height_service"], PointFloat,
                                self.callback_get_height)
        self._srv_admissibility = rospy.Service(drone_configuration["map_server"]["admissibility_service"], PointFloat,
                                self.callback_is_admissible)
        self._pose_planning = rospy.Service(drone_configuration["map_server"]["pose_planning_service"], PosePlan,
                                self.callback_get_plan)

    def callback_get_height(self, data):
        return 0

    def callback_get_plan(self, data):
        start_pose = data.start
        end_pose = data.end
        plan = self.get_plan_from_poses(start_pose, end_pose)
        return [plan]

    def callback_is_admissible(self, data):
        is_admissible = self.is_admissible(data.point)
        if is_admissible:
            return [1]
        else:
            return [0]

    def get_target_joints(self, end_position):
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, end_position.orientation.z)
        target_joints = [end_position.position.x, end_position.position.y, end_position.position.z, q[0], q[1], q[2],
                         q[3]]
        return target_joints

if __name__ == '__main__':
    map_server = DroneMoveit()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        rate.sleep()

