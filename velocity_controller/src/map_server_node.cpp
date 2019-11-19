#include <ros/ros.h>
#include <pesat_msgs/PointFloat.h>

bool map_height(pesat_msgs::PointFloat::Request  &req,
         pesat_msgs::PointFloat::Response &res)
{
  res.value = 0;
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "map_server");
  ros::NodeHandle n;

  ros::ServiceServer service = n.advertiseService("point_height", map_height);
  ROS_INFO("Map server ready.");
  ros::spin();

  return 0;
}
