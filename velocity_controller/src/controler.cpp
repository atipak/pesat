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

#include <pesat_msgs/PointHeight.h>
#include <pesat_msgs/ControllerState.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TwistWithCovariance.h>
#include <geometry_msgs/Twist.h>

//#include <eigen_conversions/eigen_msg.h>

// Use the structure definitions from the rotors_joy_interface 

#define DEG2RAD(x) ((x) / 180.0 * M_PI)

tf::TransformListener listener;
tf::TransformBroadcaster br;
ros::Publisher trajectory_pub, control_state_pub;
ros::Subscriber vel_sub, take_off_sub;
geometry_msgs::TwistWithCovariance twist_msg;
ros::ServiceClient mapClient;


bool twist_msg_ready = false;
bool simulated_frames_set_up = false;
bool mutex = false;
bool parameters_set = false;
bool is_landing, is_takingoff, is_emergency = false;
const int stop_state_const = 0, takeoff_state_const = 1, land_state_const = 2;
const int emergency_state_const = 3, fly_state_const = 4;
float limit_value = 0.1;


int axis_direction_yaw, axis_direction_pitch, axis_direction_roll;
 
double max_vel, max_yawrate, max_camerarate, max_camera_angle;

void send_state_message();
float clip(float n, float lower, float upper);
float get_minimum_in_point(float x, float y);
bool renew_velocities();
bool renew_positions();
bool renew_sim_frames();
bool set_sim_position_baselink_transform(float x, float y, float z, float roll, float pitch, float yaw);
bool set_sim_position_camera_transform(float x, float y, float z, float roll, float pitch, float yaw);
bool set_final_velocity_base_link_transform(float x, float y, float z, float roll, float pitch, float yaw);
bool set_final_velocity_camera_transform(float x, float y, float z, float roll, float pitch, float yaw);
bool reset_baselink_position();
bool reset_camera_position();
bool reset_velocities_frames();
bool update_sim_base_link_position(float x, float y, float z, float yaw);
bool real_to_sim();
bool apply_velocity_to_drone();
bool move_drone(float x, float y, float z, float yaw);
bool apply_velocity_to_camera();
void set_takeoff_params();
void set_land_params();
void twist_callback(const geometry_msgs::TwistWithCovarianceConstPtr& msg);
void take_off_callback(const std_msgs::EmptyConstPtr& msg);
void land_callback(const std_msgs::EmptyConstPtr& msg);
void emergency_callback(const std_msgs::EmptyConstPtr& msg);


tf::StampedTransform stampedTransformBaseLink;  
tf::StampedTransform stampedTransform;
// simulation baselink -> velocity frame
tf::Transform final_base_link_velocity_transform;
tf::Transform final_camera_velocity_transform;
// world -> simulation baselink
tf::Transform base_link_position_transform;
tf::Transform camera_position_transform;
// simulation baselink -> camera -> velocity
tf::Vector3 origin;
tf::Quaternion rotation;
ros::Rate rate(10.0);
double yaw = 0.0;

int main(int argc, char** argv) {

  ros::init(argc, argv, "velocity_control");
  ros::NodeHandle nh;

  // Continuously publish waypoints.
  trajectory_pub = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>(
                    mav_msgs::default_topics::COMMAND_TRAJECTORY, 10);

  control_state_pub = nh.advertise<pesat_msgs::ControllerState>("controllerState", 10);

  // Subscribe to bebop velocity commands messages
  vel_sub = nh.subscribe("/bebop2/cmd_vel", 10, &twist_callback);
  take_off_sub = nh.subscribe("/bebop2/takeoff", 10, &take_off_callback);

  ROS_INFO("Started velocity_control. Drone can be controlled by cmd_vel command (twist msg).");


  nh.param("axis_direction_yaw_"   , axis_direction_yaw, 1);
  nh.param("axis_direction_pitch_"   , axis_direction_pitch, 1);
  nh.param("axis_direction_roll_"   , axis_direction_roll, 1);
 
  nh.param("max_vel", max_vel, 1.0);
  nh.param("max_yawrate", max_yawrate, DEG2RAD(45));
  nh.param("max_camerarate", max_camerarate, DEG2RAD(45));

  nh.param("max_camera_angle", max_camera_angle, DEG2RAD(30));
  mapClient = nh.serviceClient<pesat_msgs::PointHeight>("point_height");
  //rate = Rate(10.0);
  yaw = 0.0;
  ROS_INFO("Velocity command driving.");
  while (nh.ok() && !reset_baselink_position()){
	rate.sleep();
  }
  ROS_INFO("Simulation frames was set up.");
  while (nh.ok()){
    ros::spinOnce();
    renew_sim_frames();
    if (!mutex) {
        // velocity upgrade and send new target point
        apply_velocity_to_drone();
        apply_velocity_to_camera();
    }
    else {
        reset_velocities_frames();
        // check which action is demanded
        if (is_takingoff) {
            if (!parameters_set) {
                set_takeoff_params();
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
                set_land_params();
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
        if (twist_msg.twist.linear.x != 0 || twist_msg.twist.linear.y != 0 || twist_msg.twist.linear.z != 0 ||
            twist_msg.twist.angular.x != 0 || twist_msg.twist.angular.y != 0 || twist_msg.twist.angular.z != 0) {
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

float get_minimum_in_point(float x, float y) {
	  pesat_msgs::PointHeight::Request  req;
      pesat_msgs::PointHeight::Response res;
	  req.x = x;
	  req.y = y;
	  bool success = mapClient.call(req, res);
	  float min_z;
	  if(success) {
	    min_z = res.z;
	  } else {
	    min_z = 0;
	  }
	  return min_z;
}

// TF package communications
// sends simulation velocity frames into tf package
bool renew_velocities() {
  	br.sendTransform(tf::StampedTransform(final_base_link_velocity_transform, ros::Time::now(), "bebop2/simulation/position/base_link", "bebop2/simulation/velocity/base_link"));
  	br.sendTransform(tf::StampedTransform(final_camera_velocity_transform, ros::Time::now(), "bebop2/simulation/position/camera_optical_link", "bebop2/simulation/velocity/camera_optical_link"));
}
// sends simulation position frames into tf package
bool renew_positions() {
  	br.sendTransform(tf::StampedTransform(base_link_position_transform, ros::Time::now(), "world", "bebop2/simulation/position/base_link"));
  	br.sendTransform(tf::StampedTransform(camera_position_transform, ros::Time::now(), "bebop2/simulation/position/base_link", "bebop2/simulation/position/camera_optical_link"));
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

bool set_sim_position_camera_transform(float x, float y, float z, float roll, float pitch, float yaw) {
	tf::Vector3 origin = tf::Vector3(x, y, z);
	tf::Quaternion rotation = tf::createQuaternionFromYaw(yaw);
  	camera_position_transform.setOrigin( origin );
  	camera_position_transform.setRotation( rotation );
}

bool set_final_velocity_base_link_transform(float x, float y, float z, float roll, float pitch, float yaw) {
	tf::Vector3 origin = tf::Vector3(x, y, z);
	tf::Quaternion rotation = tf::createQuaternionFromYaw(yaw);
  	final_base_link_velocity_transform.setOrigin( origin );
  	final_base_link_velocity_transform.setRotation( rotation );
}

bool set_final_velocity_camera_transform(float x, float y, float z, float roll, float pitch, float yaw) {
	tf::Vector3 origin = tf::Vector3(x, y, z);
	tf::Quaternion rotation = tf::createQuaternionFromRPY (roll, pitch, yaw);
  	final_camera_velocity_transform.setOrigin( origin );
  	final_camera_velocity_transform.setRotation( rotation );
}

bool reset_baselink_position() {
    static tf::TransformListener listener;
	static tf::TransformBroadcaster br; 
	if (listener.canTransform ("/world", "/bebop2/base_link", ros::Time(0))) {
		     // simulated base_link 
		     listener.lookupTransform("/world", "/bebop2/base_link", ros::Time(0), stampedTransformBaseLink);
		     update_sim_base_link_position(stampedTransformBaseLink.getOrigin().x(), stampedTransformBaseLink.getOrigin().y(), stampedTransformBaseLink.getOrigin().z(), tf::getYaw(stampedTransformBaseLink.getRotation()));
		     if (!reset_velocities_frames()) {return false;}
		     return true;
	}
	return false;
}

bool reset_camera_position() {
	if (listener.canTransform ("/bebop2/base_link", "/bebop2/camera_optical_link", ros::Time(0))) {
		     // simulated base_link
		     listener.lookupTransform("/bebop2/base_link", "/bebop2/camera_optical_link", ros::Time(0), stampedTransformBaseLink);
		     double roll, pitch, yaw;
             stampedTransformBaseLink.getBasis().getRPY(roll, pitch, yaw);
		     set_sim_position_camera_transform(stampedTransformBaseLink.getOrigin().x(),  stampedTransformBaseLink.getOrigin().y(),
		     stampedTransformBaseLink.getOrigin().z(), roll, pitch, yaw);
	         renew_positions();
		     if (!reset_velocities_frames()) {return false;}
		     return true;
	}
	return false;
}

bool reset_velocities_frames() {
	set_final_velocity_base_link_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	set_final_velocity_camera_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	renew_velocities();
	return true;
}

bool update_sim_base_link_position(float x, float y, float z, float yaw) {
	set_sim_position_baselink_transform(x, y, z, 0.0, 0.0, yaw);
	renew_positions();
	return true;
}

// moving
// drone
bool real_to_sim() {
 	if (listener.canTransform ("/world", "/bebop2/base_link", ros::Time(0))) {
	      listener.lookupTransform("/world", "/bebop2/base_link", ros::Time(0), stampedTransformBaseLink);
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

bool apply_velocity_to_drone() {
	// desired position + yaw
    try{
        if (listener.canTransform ("/world", "/bebop2/simulation/velocity/base_link", ros::Time(0))) {
          listener.lookupTransform("/world", "/bebop2/simulation/velocity/base_link", ros::Time(0), stampedTransform);
          float min_z = get_minimum_in_point(stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y());
          min_z = stampedTransform.getOrigin().z() > min_z ? stampedTransform.getOrigin().z() : min_z;
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
    if (listener.canTransform ("bebop2/simulation/position/base_link", "bebop2/simulation/velocity/camera_optical_link", ros::Time(0))) {
	      listener.lookupTransform("bebop2/simulation/position/base_link", "bebop2/simulation/velocity/camera_optical_link", ros::Time(0), stampedTransformBaseLink);
          double roll, pitch, yaw;
          stampedTransformBaseLink.getBasis().getRPY(roll, pitch, yaw);
		  float x = stampedTransformBaseLink.getOrigin().x();
		  float y = stampedTransformBaseLink.getOrigin().y();
		  float z = stampedTransformBaseLink.getOrigin().z();
		  double clipped_yaw = clip(yaw, -max_camera_angle, max_camera_angle);
		  double clipped_roll = clip(roll, -max_camera_angle, max_camera_angle);
		  tf::Transform camera_transform;
		  tf::Vector3 camera_origin = tf::Vector3(x, y, z);
	      tf::Quaternion camera_rotation = tf::createQuaternionFromRPY(clipped_roll, pitch,clipped_yaw);
  	      camera_transform.setOrigin( camera_origin );
  	      camera_transform.setRotation( camera_rotation );
          br.sendTransform(tf::StampedTransform(camera_transform, ros::Time::now(), "bebop2/simulation/position/base_link", "bebop2/simulation/position/camera_optical_link"));
          br.sendTransform(tf::StampedTransform(camera_transform, ros::Time::now(), "bebop2/base_link", "bebop2/camera_optical_link"));
		  return true;
	}
	else {
		ROS_INFO("Can't transform world -> base_link or world -> sim/base_link.");
		return false;
	}
}


// params setting
void set_takeoff_params() {
	reset_velocities_frames();
	// get source position (real base_link) and set final position (simulated base_link)
	if (listener.canTransform ("/world", "/bebop2/base_link", ros::Time(0))) {
		     // simulated base_link
		     listener.lookupTransform("/world", "/bebop2/base_link", ros::Time(0), stampedTransformBaseLink);
		     float min_z = get_minimum_in_point(stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y());
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

void set_land_params() {
	reset_velocities_frames();
	// get source position (real base_link) and set final position (simulated base_link)
	if (listener.canTransform ("/world", "/bebop2/base_link", ros::Time(0))) {
		     // simulated base_link
		     listener.lookupTransform("/world", "/bebop2/base_link", ros::Time(0), stampedTransformBaseLink);
		     float min_z = get_minimum_in_point(stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y());
		     // still on the ground
		     update_sim_base_link_position(stampedTransformBaseLink.getOrigin().x(), stampedTransformBaseLink.getOrigin().y(), min_z, tf::getYaw(stampedTransformBaseLink.getRotation()));
		     parameters_set = true;
	}
}

// callbacks
void twist_callback(const geometry_msgs::TwistWithCovarianceConstPtr& msg){
    if (mutex) {return;}
  	// creation/change of the vel frame
  	double clipped_x = clip((*msg).twist.linear.x, -1, 1);
  	double clipped_y = clip((*msg).twist.linear.y, -1, 1);
  	double clipped_z = clip((*msg).twist.linear.z, -1, 1);
  	double clipped_rho = clip((*msg).twist.angular.x, -1, 1);
  	double clipped_pi = clip((*msg).twist.angular.y, -1, 1);
  	double clipped_psi = clip((*msg).twist.angular.z, -1, 1);
  	tf::Vector3 temp_origin = tf::Vector3((*msg).twist.linear.x, (*msg).twist.linear.y, (*msg).twist.linear.z);
  	double temp_horizontal = (*msg).twist.angular.x * axis_direction_roll;
  	double temp_vertical = (*msg).twist.angular.y * axis_direction_pitch;
  	double temp_yaw = (*msg).twist.angular.z * axis_direction_yaw;

  	// scaling by time
  	temp_origin = temp_origin * max_vel * rate.expectedCycleTime().toSec();
  	temp_yaw = temp_yaw * max_yawrate * rate.expectedCycleTime().toSec();
  	temp_horizontal = temp_horizontal * max_camerarate * rate.expectedCycleTime().toSec();
  	temp_vertical = temp_vertical * max_camerarate * rate.expectedCycleTime().toSec();
  	// setting
  	set_final_velocity_base_link_transform(temp_origin.x(), temp_origin.y(), temp_origin.z(), temp_yaw, 0.0, 0.0);
  	set_final_velocity_camera_transform(0.0, 0.0, 0.0, temp_horizontal, temp_vertical, 0.0);
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
