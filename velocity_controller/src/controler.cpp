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

ros::Publisher trajectory_pub, control_state_pub;
ros::Subscriber vel_sub, take_off_sub;
geometry_msgs::TwistWithCovariance twist_msg;


bool twist_msg_ready = false;
bool simulated_frames_set_up = false;
bool mutex = false;
bool parameters_set = false;
bool land, takeoff, emergency = false;


int axis_direction_yaw;
 
double max_vel,
       max_yawrate;

double distance_traveled;

int state;


void twist_callback(const geometry_msgs::TwistWithCovarianceConstPtr& msg);
void take_off_callback(const std_msgs::EmptyConstPtr& msg);
void land_callback(const std_msgs::EmptyConstPtr& msg);
void emergency_callback(const std_msgs::EmptyConstPtr& msg);
bool update_sim_base_link(float x, float y, float z, float yaw);
bool move_drone(float x, float y, float z, float yaw);
bool real_to_sim();
bool reset_velocity_frame();
bool reset_baselink_frame();
bool shift_drone(ros::ServiceClient mapClient);
bool velocity_update();
void set_takeoff_params(ros::ServiceClient mapClient);
bool renew_velocity();
bool renew_sim_baselink();
bool renew_sim_frames();
bool set_sim_baselink_transform(float x, float y, float z, float yaw, float angular_x, float angular_y);
bool set_velocity_transform(float x, float y, float z, float yaw, float angular_x, float angular_y);
float get_minimum_in_point(float x, float y, ros::ServiceClient mapClient);
void send_state_message();


tf::StampedTransform stampedTransformBaseLink;  
tf::StampedTransform stampedTransform;
// simulation baselink -> velocity frame
tf::Transform velocity_transform;
// world -> simulation baselink
tf::Transform sim_baselink_transform;
tf::Vector3 origin;
tf::Quaternion rotation;
ros::Time prev_time;
double dt;
double yaw = 0.0;

int main(int argc, char** argv) {

  ros::init(argc, argv, "velocity_control");
  ros::NodeHandle nh;
  static tf::TransformListener listener;
  static tf::TransformBroadcaster br; 

  // Continuously publish waypoints.
  trajectory_pub = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>(
                    mav_msgs::default_topics::COMMAND_TRAJECTORY, 10);

  control_state_pub = nh.advertise<pesat_msgs::ControllerState>("controllerState", 10);

  // Subscribe to bebop velocity commands messages
  vel_sub = nh.subscribe("/bebop2/cmd_vel", 10, &twist_callback);
  take_off_sub = nh.subscribe("/bebop2/takeoff", 10, &take_off_callback);

  ROS_INFO("Started velocity_control. Drone can be controled by cmd_vel command (twist msg).");


  nh.param("axis_direction_yaw_"   , axis_direction_yaw, 1);
 
  nh.param("max_vel", max_vel, 1.0);
  nh.param("max_yawrate", max_yawrate, DEG2RAD(45));

  distance_traveled = 0.0;
  ros::ServiceClient mapClient;
  mapClient = nh.serviceClient<pesat_msgs::PointHeight>("point_height");
  ros::Rate rate(10.0);
  prev_time = ros::Time::now();
  dt = (ros::Time::now() - prev_time).toSec();
  yaw = 0.0;
  ROS_INFO("Velocity command driving.");
  while (nh.ok() && !reset_baselink_frame()){
	rate.sleep();
  }
  ROS_INFO("Simulation frames was set up.");
  while (nh.ok()){
    ros::spinOnce();
    renew_sim_frames();
    if (!mutex) {
	// velocity upgrade and send new target point
	if (twist_msg_ready) {
		velocity_update();
	}
	else {
		renew_sim_frames();
	}
	shift_drone(mapClient);
	state = 1;
    }
    else {
	reset_velocity_frame();
	// check which action is demanded
	if (takeoff) {
		state = 2;
		if (!parameters_set) {
			set_takeoff_params(mapClient);
		}
		else {
				renew_sim_frames();
				if (real_to_sim()) {
					mutex = false;
					takeoff = false;
					parameters_set = false;
				}
		}
	}
	else if (land) {
		state = 3;
	}
	else {
		state = 4;
	}
    }
    send_state_message();
    prev_time = ros::Time::now();
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}

void send_state_message() {
	pesat_msgs::ControllerState state_msg;
	// velocity command set up
	if (state == 1) {
		if (twist_msg.twist.linear.x != 0 || twist_msg.twist.linear.y != 0 || twist_msg.twist.linear.z != 0 ||
		    twist_msg.twist.angular.x != 0 || twist_msg.twist.angular.y != 0 || twist_msg.twist.angular.z != 0) {
			state_msg.twist.linear.x = twist_msg.twist.linear.x; 
			state_msg.twist.linear.y = twist_msg.twist.linear.y;
			state_msg.twist.linear.z = twist_msg.twist.linear.z;
			state_msg.twist.angular.x = twist_msg.twist.angular.x;
			state_msg.twist.angular.y = twist_msg.twist.angular.y;
			state_msg.twist.angular.z = twist_msg.twist.angular.z;
		}
		else {
			// zero velocity
			state = 0;
		}
	}
	state_msg.state = state;
	control_state_pub.publish(state_msg);
}

float get_minimum_in_point(float x, float y, ros::ServiceClient mapClient) {
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

bool set_sim_baselink_transform(float x, float y, float z, float yaw, float angular_x, float angular_y) {
	tf::Vector3 origin = tf::Vector3(x, y, z);
	tf::Quaternion rotation = tf::createQuaternionFromYaw(yaw);
  	sim_baselink_transform.setOrigin( origin );
  	sim_baselink_transform.setRotation( rotation );
}

bool set_velocity_transform(float x, float y, float z, float yaw, float angular_x, float angular_y) {
	tf::Vector3 origin = tf::Vector3(x, y, z);
	tf::Quaternion rotation = tf::createQuaternionFromYaw(yaw);
  	velocity_transform.setOrigin( origin );
  	velocity_transform.setRotation( rotation );
}

bool shift_drone(ros::ServiceClient mapClient) {
	// desired position + yaw
	  try{
	      static tf::TransformListener listener;
	      if (listener.canTransform ("/world", "/bebop2/simulation/velocity", ros::Time(0))) {
			  listener.lookupTransform("/world", "/bebop2/simulation/velocity", ros::Time(0), stampedTransform);
			  float rounded_x = stampedTransform.getOrigin().x(); 
			  float rounded_y = stampedTransform.getOrigin().y(); 
			  float rounded_z = stampedTransform.getOrigin().z(); 
			  float min_z = get_minimum_in_point(rounded_x, rounded_y, mapClient);
			  rounded_z = rounded_z > min_z ? rounded_z : min_z;
			  double desired_yaw = tf::getYaw(stampedTransform.getRotation()); 
			  if (rounded_x != 0 || rounded_y != 0 || rounded_z != 0 || desired_yaw != 0) {
				  move_drone(rounded_x, rounded_y, rounded_z, desired_yaw);
				  // update simulated base link
				  update_sim_base_link(rounded_x, rounded_y, rounded_z, desired_yaw);
			  }
			  return true;
			  // ROS_INFO("Transform msg: time: %f, x: %f, y: %f, z: %f, yaw: %f", stampedTransform.stamp_, stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y(), stampedTransform.getOrigin().z(), tf::getYaw(stampedTransform.getRotation()));
			  // ROS_INFO("Base link msg: time: %f, x: %f, y: %f, z: %f, yaw: %f", stampedTransformBaseLink.stamp_, stampedTransformBaseLink.getOrigin().x(), stampedTransformBaseLink.getOrigin().y(), stampedTransformBaseLink.getOrigin().z(), tf::getYaw(stampedTransformBaseLink.getRotation()));
		}
	      else {
		reset_baselink_frame();
		return false;
		}
	    }
	    catch (tf::TransformException &ex) {
	      ROS_ERROR("%s",ex.what());
	      ros::Duration(1.0).sleep();
	      return false;
	    }
}

bool renew_velocity() {
	static tf::TransformBroadcaster br;
  	br.sendTransform(tf::StampedTransform(velocity_transform, ros::Time::now(), "bebop2/simulation/base_link", "bebop2/simulation/velocity"));

}

bool renew_sim_baselink() {
	static tf::TransformBroadcaster br;
  	br.sendTransform(tf::StampedTransform(sim_baselink_transform, ros::Time::now(), "world", "bebop2/simulation/base_link"));
}


bool renew_sim_frames() {
	renew_sim_baselink();
	renew_velocity();
}

bool velocity_update() {
	static tf::TransformBroadcaster br; 
	dt = (ros::Time::now() - prev_time).toSec();
  	prev_time = ros::Time::now();
  	// creation/change of the vel frame
  	tf::Vector3 temp_origin = tf::Vector3(twist_msg.twist.linear.x, twist_msg.twist.linear.y, twist_msg.twist.linear.z);
  	double temp_yaw = twist_msg.twist.angular.z * axis_direction_yaw;

  	// scaling by time
  	temp_origin = temp_origin * max_vel * dt;  
  	temp_yaw = temp_yaw * max_yawrate * dt;

	set_velocity_transform(temp_origin.x(), temp_origin.y(), temp_origin.z(), temp_yaw, 0.0, 0.0);
	renew_velocity();
	//ROS_INFO("Twist msg: x: %f, y: %f, z: %f, yaw: %f", twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z, twist_msg.angular.z);
	//ROS_INFO("Update velocity x: %f, y: %f, z: %f, yaw: %f", twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z, twist_msg.angular.z);
	return true;
}


void set_takeoff_params(ros::ServiceClient mapClient) {
	reset_velocity_frame();
	// get source position (real base_link) and set final position (simulated base_link)
        static tf::TransformListener listener;
	if (listener.canTransform ("/world", "/bebop2/base_link", ros::Time(0))) {
		     // simulated base_link 
		     listener.lookupTransform("/world", "/bebop2/base_link", ros::Time(0), stampedTransformBaseLink);
		     float min_z = get_minimum_in_point(stampedTransform.getOrigin().x(), stampedTransform.getOrigin().y(), mapClient);
		     // already in air
		     if (stampedTransformBaseLink.getOrigin().z() > min_z + 0.3) {
			mutex = false;
			takeoff = false;
			return;
		     }
		     // still on the ground
		     update_sim_base_link(stampedTransformBaseLink.getOrigin().x(), stampedTransformBaseLink.getOrigin().y(), min_z + 1, tf::getYaw(stampedTransformBaseLink.getRotation()));
		     parameters_set = true;
	}
}


bool reset_baselink_frame() {
        static tf::TransformListener listener;
	static tf::TransformBroadcaster br; 
	if (listener.canTransform ("/world", "/bebop2/base_link", ros::Time(0))) {
		     // simulated base_link 
		     listener.lookupTransform("/world", "/bebop2/base_link", ros::Time(0), stampedTransformBaseLink);
		     update_sim_base_link(stampedTransformBaseLink.getOrigin().x(), stampedTransformBaseLink.getOrigin().y(), stampedTransformBaseLink.getOrigin().z(), tf::getYaw(stampedTransformBaseLink.getRotation()));
		     if (!reset_velocity_frame()) {return false;}
		     return true;
	}
	return false;
}

bool reset_velocity_frame() {
        static tf::TransformListener listener;
	static tf::TransformBroadcaster br; 
	set_velocity_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	renew_velocity();
	return true;
}

bool real_to_sim() {
	float limit_value = 0.1;
        static tf::TransformListener listener;
 	if (listener.canTransform ("/world", "/bebop2/base_link", ros::Time(0))) {
	          listener.lookupTransform("/world", "/bebop2/base_link", ros::Time(0), stampedTransformBaseLink);
		  move_drone(sim_baselink_transform.getOrigin().x(), sim_baselink_transform.getOrigin().y(), sim_baselink_transform.getOrigin().z(), tf::getYaw(sim_baselink_transform.getRotation()));
		  float x = stampedTransformBaseLink.getOrigin().x(); 
		  float y = stampedTransformBaseLink.getOrigin().y(); 
		  float z = stampedTransformBaseLink.getOrigin().z(); 
		  double yaw = tf::getYaw(stampedTransformBaseLink.getRotation());  
		  if(abs(sim_baselink_transform.getOrigin().x() - x) < limit_value && abs(sim_baselink_transform.getOrigin().y() - y) < limit_value && abs(sim_baselink_transform.getOrigin().z() - z) < limit_value && abs(tf::getYaw(sim_baselink_transform.getRotation()) - yaw) < limit_value) {
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

bool move_drone(float x, float y, float z, float yaw) {
	  // tf::Vector3 -> Eigen::Vector3d, yaw
	  Eigen::Vector3d desired_position(
		x,
		y, 
		z);
	  static trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
	  trajectory_msg.header.stamp = ros::Time::now();
	  trajectory_msg.header.seq++;

	  mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(desired_position,
	      yaw, &trajectory_msg);

	  trajectory_msg.points[0].time_from_start = ros::Duration(1.0); 
	  trajectory_pub.publish(trajectory_msg); 
	  return true;
}

bool update_sim_base_link(float x, float y, float z, float yaw) {
	set_sim_baselink_transform(x, y, z, yaw, 0.0, 0.0);
	renew_sim_baselink();
	return true;
}

// callbacks
void twist_callback(const geometry_msgs::TwistWithCovarianceConstPtr& msg){
  twist_msg = *msg;
  //ROS_INFO("Linear x: %f, y: %f, z: %f; angular: z: %f", twist_msg.linear.x,  twist_msg.linear.y,  twist_msg.linear.z,  twist_msg.angular.z);
  twist_msg_ready = true;
}

void take_off_callback(const std_msgs::EmptyConstPtr& msg) {
	// reset velocity
	twist_msg_ready = false;
	mutex = true;
	takeoff = true;
}


void land_callback(const std_msgs::EmptyConstPtr& msg) {}
void emergency_callback(const std_msgs::EmptyConstPtr& msg) {}
