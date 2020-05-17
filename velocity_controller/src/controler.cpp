#include <fstream>
#include <iostream>
#include <math.h>

#include <Eigen/Core>
#include <mav_msgs/conversions.h>
#include <mav_msgs/default_topics.h>
#include <ros/ros.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <pesat_msgs/PointFloat.h>
#include <pesat_msgs/ControllerState.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TwistWithCovariance.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/Float64.h>
//#include <string>

//#include <eigen_conversions/eigen_msg.h>

// Use the structure definitions from the rotors_joy_interface 

#define DEG2RAD(x) ((x) / 180.0 * M_PI)

void send_state_message();
bool init_transforms();
float clip(float n, float lower, float upper);
float get_minimum_in_point(float x, float y, ros::ServiceClient mapClient);
float get_maximum_in_point(float x, float y);
bool renew_velocities();
bool renew_positions();
bool renew_sim_frames();
bool set_sim_position_baselink_transform(float x, float y, float z, float roll, float pitch, float yaw);
bool set_final_velocity_base_link_transform(float x, float y, float z, float roll, float pitch, float yaw);
bool reset_baselink_position();
bool reset_velocities_frames();
bool update_sim_base_link_position(float x, float y, float z, float yaw);
bool real_to_sim();
bool apply_velocity_to_drone(ros::ServiceClient mapClient);
bool move_drone(float x, float y, float z, float yaw);
bool apply_velocity_to_camera();
void set_takeoff_params(ros::ServiceClient mapClient);
void set_land_params(ros::ServiceClient mapClient);
void twist_callback(const geometry_msgs::TwistWithCovarianceConstPtr& msg);
void take_off_callback(const std_msgs::EmptyConstPtr& msg);
void land_callback(const std_msgs::EmptyConstPtr& msg);
void emergency_callback(const std_msgs::EmptyConstPtr& msg);
void check_twist_time();


tf::StampedTransform stampedTransformBaseLink;  
tf::StampedTransform stampedTransform;
// simulation baselink -> velocity frame
tf::Transform final_base_link_velocity_transform;
// world -> simulation baselink
tf::Transform base_link_position_transform;
tf::Vector3 origin;
tf::Quaternion rotation;
tf::Vector3 orientation_camera_velocity;
tf::Vector3 orientation_camera;

tf::TransformListener* listener;
tf::TransformBroadcaster* br;
ros::Publisher trajectory_pub, control_state_pub, horizontal_camera_controller_pub, vertical_camera_controller_pub;
ros::Subscriber vel_sub, take_off_sub;


geometry_msgs::TwistWithCovariance twist_msg;
geometry_msgs::TwistWithCovariance nullified_twist_msg;
bool simulated_frames_set_up = false;
bool mutex = false;
bool parameters_set = false;
bool is_landing, is_takingoff, is_emergency = false;
const int stop_state_const = 0, takeoff_state_const = 1, land_state_const = 2;
const int emergency_state_const = 3, fly_state_const = 4;
float limit_value = 0.1;
int axis_direction_yaw, axis_direction_pitch, axis_direction_roll;
double max_horizontal_speed, max_vertical_speed, max_yawrate, max_camerarate, max_camera_angle, max_altitude;
float dt = 0.0;
double yaw = 0.0;
const double pi_half = M_PI / 2;
double last_message_time;
double twist_message_timeout = 0.5;

int main(int argc, char** argv) {

  ros::init(argc, argv, "velocity_control");
  ros::NodeHandle nh;

  ROS_INFO("Started velocity_control. Drone can be controlled by cmd_vel command (twist msg).");

  nh.param("drone_configuration/control/axis_direction_yaw"   , axis_direction_yaw, 1);
  nh.param("drone_configuration/control/axis_direction_pitch"   , axis_direction_pitch, 1);
  nh.param("drone_configuration/control/axis_direction_roll"   , axis_direction_roll, 1);
  nh.param("drone_configuration/control/video_max_horizontal_speed", max_horizontal_speed, 1.0);
  nh.param("drone_configuration/control/video_max_vertical_speed", max_vertical_speed, 1.0);
  nh.param("drone_configuration/control/video_max_rotation_speed", max_yawrate, DEG2RAD(45));
  nh.param("drone_configuration/control/video_max_camera_rotation_speed", max_camerarate, DEG2RAD(45));
  nh.param("drone_configuration/control/max_camera_rotation", max_camera_angle, DEG2RAD(45));
  nh.param("drone_configuration/control/max_altitude", max_altitude, 30.0);

  // Continuously publish waypoints.
  trajectory_pub = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>(
                    mav_msgs::default_topics::COMMAND_TRAJECTORY, 10);

  control_state_pub = nh.advertise<pesat_msgs::ControllerState>("controllerState", 10);
  horizontal_camera_controller_pub = nh.advertise<std_msgs::Float64>("/bebop2/camera_base_controller/command", 10);
  vertical_camera_controller_pub = nh.advertise<std_msgs::Float64>("/bebop2/rotator_controller/command", 10);
  // Subscribe to bebop velocity commands messages
  vel_sub = nh.subscribe("/bebop2/cmd_vel", 10, &twist_callback);
  take_off_sub = nh.subscribe("/bebop2/takeoff", 10, &take_off_callback);
  ros::ServiceClient mapClient = nh.serviceClient<pesat_msgs::PointFloat>("/map_server/bebop2/point_height_service", true);
  listener = new (tf::TransformListener);
  br = new(tf::TransformBroadcaster);
  ros::Rate rate(10.0);
  dt = rate.expectedCycleTime().toSec();
  yaw = 0.0;
  ROS_INFO("Velocity command driving.");
  while (nh.ok()){
    if (init_transforms()) {
        break;
    }
	rate.sleep();
  }
  ROS_INFO("Simulation frames was set up.");
  while (nh.ok()){
    ros::spinOnce();
    renew_sim_frames();
    if (!mutex) {
        // velocity upgrade and send new target point
        apply_velocity_to_drone(mapClient);
        apply_velocity_to_camera();
    }
    else {
        reset_velocities_frames();
        // check which action is demanded
        if (is_takingoff) {
            if (!parameters_set) {
                set_takeoff_params(mapClient);
            }
            else {
                    renew_sim_frames();
                    if (real_to_sim()) {
                        mutex = false;
                        is_takingoff = false;
                        parameters_set = false;
                    }
            }
        }
        else if (is_landing) {
            if (!parameters_set) {
                set_land_params(mapClient);
            }
            else {
                    renew_sim_frames();
                    if (real_to_sim()) {
                        mutex = false;
                        is_landing = false;
                        parameters_set = false;
                    }
            }
        }
        else if (is_emergency) {
            reset_velocities_frames();
            mutex = false;
            is_emergency = false;
        }
    }
    check_twist_time();
    send_state_message();
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}

// helps functions
float clip(float n, float lower, float upper) {
  return std::max(lower, std::min(n, upper));
}

void check_twist_time() {
    if (std::abs(ros::Time::now().toSec() - last_message_time) > twist_message_timeout) {
        twist_msg = nullified_twist_msg;
        reset_velocities_frames();
    }
}

// ROS topic/service communication
void send_state_message() {
	pesat_msgs::ControllerState state_msg;
	// velocity command set up
	if (is_landing) {
	    state_msg.state = land_state_const;
	}
	else if (is_takingoff) {
	    state_msg.state = takeoff_state_const;
	}
	else if (is_emergency) {
	    state_msg.state = emergency_state_const;
	}
	else {
        if (final_base_link_velocity_transform.getOrigin().x() != 0 || final_base_link_velocity_transform.getOrigin().y() != 0 ||
        final_base_link_velocity_transform.getOrigin().z() != 0 || final_base_link_velocity_transform.getRotation().getW() != 1 ||
        orientation_camera_velocity.getX() != 0 || orientation_camera_velocity.getY() != 0) {
            state_msg.twist.linear.x = twist_msg.twist.linear.x;
            state_msg.twist.linear.y = twist_msg.twist.linear.y;
            state_msg.twist.linear.z = twist_msg.twist.linear.z;
            state_msg.twist.angular.x = twist_msg.twist.angular.x;
            state_msg.twist.angular.y = twist_msg.twist.angular.y;
            state_msg.twist.angular.z = twist_msg.twist.angular.z;
            state_msg.state = 1;
        }
        else {
            // zero velocity
            state_msg.state = 0;
        }
	}
	control_state_pub.publish(state_msg);
}

float get_minimum_in_point(float x, float y, ros::ServiceClient mapClient) {
	  pesat_msgs::PointFloat::Request  req;
      pesat_msgs::PointFloat::Response res;
      geometry_msgs::Point point;
      point.x = x;
      point.y = y;
      req.point = point;
      try {
          bool success = mapClient.call(req, res);
          float min_z;
          if(success) {
            min_z = res.value;
          } else {
            min_z = 0;
          }
          return min_z;
      }
      catch (...) {
        return 0;
      }

}

float get_maximum_in_point(float x, float y) {
    return max_altitude;
}



// TF package communications
// init transforms
bool init_transforms() {
    if (reset_baselink_position()) {
        return true;
    }
    else {
        return false;
    }
}

// sends simulation velocity frames into tf package
bool renew_velocities() {
  	br->sendTransform(tf::StampedTransform(final_base_link_velocity_transform, ros::Time::now(), "bebop2/simulation/position/base_link", "bebop2/simulation/velocity/base_link"));
}
// sends simulation position frames into tf package
bool renew_positions() {
  	br->sendTransform(tf::StampedTransform(base_link_position_transform, ros::Time::now(), "world", "bebop2/simulation/position/base_link"));
}
// sends simulation velocity and position frames into tf package
bool renew_sim_frames() {
	renew_positions();
	renew_velocities();
}

// setting local transformations
bool set_sim_position_baselink_transform(float x, float y, float z, float roll, float pitch, float yaw) {
	tf::Vector3 origin = tf::Vector3(x, y, z);
	tf::Quaternion rotation = tf::createQuaternionFromYaw(yaw);
  	base_link_position_transform.setOrigin( origin );
  	base_link_position_transform.setRotation( rotation );
}

bool set_final_velocity_base_link_transform(float x, float y, float z, float roll, float pitch, float yaw) {
	tf::Vector3 origin = tf::Vector3(x, y, z);
	tf::Quaternion rotation = tf::createQuaternionFromYaw(yaw);
  	final_base_link_velocity_transform.setOrigin( origin );
  	final_base_link_velocity_transform.setRotation( rotation );
}

bool reset_baselink_position() {
	if (listener->canTransform ("/world", "/bebop2/base_link", ros::Time(0))) {
		     // simulated base_link 
		     listener->lookupTransform("/world", "/bebop2/base_link", ros::Time(0), stampedTransformBaseLink);
		     double roll, pitch, yaw;
             stampedTransformBaseLink.getBasis().getRPY(roll, pitch, yaw);
		     update_sim_base_link_position(stampedTransformBaseLink.getOrigin().x(), stampedTransformBaseLink.getOrigin().y(), stampedTransformBaseLink.getOrigin().z(), tf::getYaw(stampedTransformBaseLink.getRotation()));
		     if (!reset_velocities_frames()) {return false;}
		     return true;
	}
	return false;
}

bool reset_velocities_frames() {
	set_final_velocity_base_link_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	orientation_camera_velocity = tf::Vector3(0, 0, 0);
	return true;
}

bool update_sim_base_link_position(float x, float y, float z, float yaw) {
	set_sim_position_baselink_transform(x, y, z, 0.0, 0.0, yaw);
	return true;
}

// moving
// drone
bool real_to_sim() {
 	if (listener->canTransform ("/world", "/bebop2/base_link", ros::Time(0))) {
	      listener->lookupTransform("/world", "/bebop2/base_link", ros::Time(0), stampedTransformBaseLink);
		  move_drone(base_link_position_transform.getOrigin().x(), base_link_position_transform.getOrigin().y(), base_link_position_transform.getOrigin().z(), tf::getYaw(base_link_position_transform.getRotation()));
		  apply_velocity_to_camera();
		  float x = stampedTransformBaseLink.getOrigin().x();
		  float y = stampedTransformBaseLink.getOrigin().y();
		  float z = stampedTransformBaseLink.getOrigin().z();
		  double yaw = tf::getYaw(stampedTransformBaseLink.getRotation());
		  if(abs(base_link_position_transform.getOrigin().x() - x) < limit_value && abs(base_link_position_transform.getOrigin().y() - y) < limit_value && abs(base_link_position_transform.getOrigin().z() - z) < limit_value && abs(tf::getYaw(base_link_position_transform.getRotation()) - yaw) < limit_value) {
			//ROS_INFO("Goal reached");
			return true;
		  }
		  return false;
	}
	else {
		ROS_INFO("Can't transform world -> base_link or world -> sim/base_link.");
		return false;
	}
}

bool apply_velocity_to_drone(ros::ServiceClient mapClient) {
	// desired position + yaw
    try{
        if (listener->canTransform ("/world", "/bebop2/simulation/velocity/base_link", ros::Time(0))) {
          listener->lookupTransform("/world", "/bebop2/simulation/velocity/base_link", ros::Time(0), stampedTransform);
          float min_z = get_minimum_in_point(stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y(), mapClient);
          float max_z = get_maximum_in_point(stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y());
          min_z = stampedTransform.getOrigin().z() > min_z ? stampedTransform.getOrigin().z() : min_z;
          min_z = min_z < max_z ? min_z : max_z;
          double desired_yaw = tf::getYaw(stampedTransform.getRotation());
          if (stampedTransform.getOrigin().x() != 0 || stampedTransform.getOrigin().y() != 0 || min_z != 0 || desired_yaw != 0) {
              move_drone(stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y(), min_z, desired_yaw);
              // update simulated base link
              update_sim_base_link_position(stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y(), min_z, desired_yaw);
          }
          return true;
          // ROS_INFO("Transform msg: time: %f, x: %f, y: %f, z: %f, yaw: %f", stampedTransform.stamp_, stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y(), stampedTransform.getOrigin().z(), tf::getYaw(stampedTransform.getRotation()));
          // ROS_INFO("Base link msg: time: %f, x: %f, y: %f, z: %f, yaw: %f", stampedTransformBaseLink.stamp_, stampedTransformBaseLink.getOrigin().x(), stampedTransformBaseLink.getOrigin().y(), stampedTransformBaseLink.getOrigin().z(), tf::getYaw(stampedTransformBaseLink.getRotation()));
        }
        else {
          reset_baselink_position();
          return false;
        }
    }
    catch (tf::TransformException &ex) {
      ROS_ERROR("%s",ex.what());
      ros::Duration(1.0).sleep();
      return false;
    }
}

bool move_drone(float x, float y, float z, float yaw) {
	  // tf::Vector3 -> Eigen::Vector3d, yaw
	  Eigen::Vector3d desired_position(x, y, z);
	  static trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
	  trajectory_msg.header.stamp = ros::Time::now();
	  trajectory_msg.header.seq++;

	  mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(desired_position,
	      yaw, &trajectory_msg);

	  trajectory_msg.points[0].time_from_start = ros::Duration(1.0);
	  trajectory_pub.publish(trajectory_msg);
	  return true;
}

// camera
bool apply_velocity_to_camera() {
    float clipped_rho = clip(orientation_camera.getX() + orientation_camera_velocity.getX(), -max_camera_angle, max_camera_angle);
    float clipped_pi = clip(orientation_camera.getY() + orientation_camera_velocity.getY(), -max_camera_angle, max_camera_angle);
    orientation_camera.setX(clipped_rho);
    orientation_camera.setY(clipped_pi);
    std_msgs::Float64 vertical;
    vertical.data = clipped_rho;
    std_msgs::Float64 horizontal;
    horizontal.data = clipped_pi;
    //ROS_INFO("Camera msg: clipped_rho: %f, clipped_pi: %f, x - orientation camera: %f, y - orientation camera: %f, x - vel: %f, y - vel: %f", clipped_rho, clipped_pi, orientation_camera.getX(), orientation_camera.getY(), orientation_camera_velocity.getX(), orientation_camera_velocity.getY());
    vertical_camera_controller_pub.publish(vertical);
    horizontal_camera_controller_pub.publish(horizontal);
}

// params setting
void set_takeoff_params(ros::ServiceClient mapClient) {
	reset_velocities_frames();
	// get source position (real base_link) and set final position (simulated base_link)
	if (listener->canTransform ("/world", "/bebop2/base_link", ros::Time(0))) {
		     // simulated base_link
		     listener->lookupTransform("/world", "/bebop2/base_link", ros::Time(0), stampedTransformBaseLink);
		     float min_z = get_minimum_in_point(stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y(), mapClient);
		     // already in air
		     if (stampedTransformBaseLink.getOrigin().z() > min_z + 0.3) {
			mutex = false;
			is_takingoff = false;
			return;
		     }
		     // still on the ground
		     update_sim_base_link_position(stampedTransformBaseLink.getOrigin().x(), stampedTransformBaseLink.getOrigin().y(), min_z + 1, tf::getYaw(stampedTransformBaseLink.getRotation()));
		     parameters_set = true;
	}
}

void set_land_params(ros::ServiceClient mapClient) {
	reset_velocities_frames();
	// get source position (real base_link) and set final position (simulated base_link)
	if (listener->canTransform ("/world", "/bebop2/base_link", ros::Time(0))) {
		     // simulated base_link
		     listener->lookupTransform("/world", "/bebop2/base_link", ros::Time(0), stampedTransformBaseLink);
		     float min_z = get_minimum_in_point(stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y(), mapClient);
		     // still on the ground
		     update_sim_base_link_position(stampedTransformBaseLink.getOrigin().x(), stampedTransformBaseLink.getOrigin().y(), min_z, tf::getYaw(stampedTransformBaseLink.getRotation()));
		     parameters_set = true;
	}
}

// callbacks
void twist_callback(const geometry_msgs::TwistWithCovarianceConstPtr& msg){
    if (mutex) {return;}
    twist_msg = *msg;
    last_message_time = ros::Time::now().toSec();
  	// creation/change of the vel frame
  	double clipped_x = clip((*msg).twist.linear.x, -1, 1);
  	double clipped_y = clip((*msg).twist.linear.y, -1, 1);
  	double clipped_z = clip((*msg).twist.linear.z, -1, 1);
  	double clipped_rho = clip((*msg).twist.angular.x, -1, 1);
  	double clipped_pi = clip((*msg).twist.angular.y, -1, 1);
  	double clipped_psi = clip((*msg).twist.angular.z, -1, 1);
  	tf::Vector3 temp_origin_drone = tf::Vector3(clipped_x, clipped_y, 0);
  	double temp_horizontal = clipped_rho * axis_direction_roll;
  	double temp_vertical = clipped_pi * axis_direction_pitch;
  	double temp_yaw = clipped_psi * axis_direction_yaw;
  	tf::Vector3 temp_orientation_camera = tf::Vector3(temp_horizontal, temp_vertical, 0);

  	// scaling by time
  	temp_origin_drone = temp_origin_drone * max_horizontal_speed * dt;
  	temp_origin_drone.setZ(max_vertical_speed * clipped_z * dt);
  	temp_yaw = temp_yaw * max_yawrate * dt;
  	orientation_camera_velocity = temp_orientation_camera * max_camerarate * dt;
  	// setting
  	set_final_velocity_base_link_transform(temp_origin_drone.x(), temp_origin_drone.y(), temp_origin_drone.z(), 0.0, 0.0, temp_yaw);
}

void take_off_callback(const std_msgs::EmptyConstPtr& msg) {
	// reset velocity
	reset_velocities_frames();
	mutex = true;
	is_takingoff = true;
	is_landing = false;
	is_emergency = false;
}

void land_callback(const std_msgs::EmptyConstPtr& msg) {
    reset_velocities_frames();
    mutex = true;
    is_takingoff = false;
	is_landing = true;
	is_emergency = false;
}

void emergency_callback(const std_msgs::EmptyConstPtr& msg) {
	reset_velocities_frames();
	mutex = true;
	is_takingoff = false;
	is_landing = false;
	is_emergency = true;

}
