#include <ros/ros.h>
#include <geometry_msgs/TwistWithCovariance.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/utils.h>

struct Goal {
  float x;
  float y;
  float z;
  float yaw;
} ;

ros::Publisher twist_pub, takeoff_pub;
ros::Subscriber odom_sub;
geometry_msgs::Pose pose;
geometry_msgs::Pose abs_pose;

bool pose_updated = false;

void send_message(float x, float y, float z, float yaw, ros::Duration duration);
void send_message(float x, float y, float z, float yaw);
bool get_to_pose(geometry_msgs::Pose pose);
void send_pose(float x, float y, float z, float yaw);
void set_goal_position(float x, float y, float z, float yaw);
bool get_goal_position();
bool set_goal_pos_safe(float x, float y, float z, float yaw);
int signum(float x);


int main(int argc, char ** argv) {
	ros::init(argc, argv, "Square");
	ros::NodeHandle nh;
	int position_counts = 9;
	// init goals
	Goal goals[position_counts] = {{3.0, 0.0, 1.0, 0},
					{3.0, 0.0, 1.0, 1.5708},
                               		{3.0, 3.0, 1.0, 1.5708},
                               		{3.0, 3.0, 1.0, 3.14},
                               		{0.0, 3.0, 1.0, 3.14},
                               		{0.0, 3.0, 1.0, -1.5708},
                               		{0.0, 0.0, 1.0, -1.5708},
                               		{0.0, 0.0, 1.0, 0.0},
					{0.0, 0.0, 0.0, 0.0}};
	twist_pub = nh.advertise<geometry_msgs::TwistWithCovariance>("/bebop2/cmd_vel", 10);
	takeoff_pub = nh.advertise<std_msgs::Empty>("/bebop2/takeoff", 10);
	ros::Rate rate(10.0);
	for( int a = 0; a < position_counts; a = a + 1 ) {
		ROS_INFO("Goal: x: %f, y: %f, z: %f, yaw: %f", goals[a].x, goals[a].y, goals[a].z, goals[a].yaw);
		while(nh.ok()) {
			ros::spinOnce();     			
			if (set_goal_pos_safe(goals[a].x, goals[a].y, goals[a].z, goals[a].yaw)) {
				if (get_to_pose(pose)) {break;}
			}else {
				send_message(0, 0, 0, 0);	
			} 		
			ros::spinOnce();
			rate.sleep();
		}
   	}
	send_message(0, 0, 0, 0);	
	return 0;
}

bool set_goal_pos_safe(float x, float y, float z, float yaw) {
	set_goal_position(x, y, z, yaw); 		
	ros::spinOnce();
	bool success = get_goal_position();
	if (success) {
		if (abs_pose.position.x == x &&
		    abs_pose.position.y == y &&
		    abs_pose.position.z == z &&
		    tf2::getYaw(abs_pose.orientation) == yaw) {
			return true;
		}
		else {
			return false;
		}
	}
	else {
		return false;	
	}
}

void set_goal_position(float x, float y, float z, float yaw) {
	static tf2_ros::TransformBroadcaster br;
	geometry_msgs::TransformStamped transformStamped;
 	transformStamped.header.stamp = ros::Time::now();
	transformStamped.header.frame_id = "world";
	transformStamped.child_frame_id = "square/goal";
	transformStamped.transform.translation.x = x;
	transformStamped.transform.translation.y = y;
	transformStamped.transform.translation.z = z;
	tf2::Quaternion q;
	q.setRPY(0, 0, yaw);
	transformStamped.transform.rotation.x = q.x();
	transformStamped.transform.rotation.y = q.y();
	transformStamped.transform.rotation.z = q.z();
	transformStamped.transform.rotation.w = q.w();
	br.sendTransform(transformStamped);
}

bool get_goal_position() {
	static tf2_ros::Buffer tfBuffer;
	static tf2_ros::TransformListener listener(tfBuffer);
        geometry_msgs::TransformStamped stampedTransform;  
	if (tfBuffer.canTransform ("bebop2/simulation/base_link", "square/goal", ros::Time(0))) {
		// relative position to bebop
		stampedTransform = tfBuffer.lookupTransform("bebop2/simulation/base_link", "square/goal", ros::Time(0));
		geometry_msgs::Point point;
		point.x = stampedTransform.transform.translation.x;
		point.y = stampedTransform.transform.translation.y;
		point.z = stampedTransform.transform.translation.z;
		pose.position = point;
		pose.orientation = stampedTransform.transform.rotation;
		// world position
		stampedTransform = tfBuffer.lookupTransform("world", "square/goal", ros::Time(0));
		geometry_msgs::Point abs_point;
		abs_point.x = stampedTransform.transform.translation.x;
		abs_point.y = stampedTransform.transform.translation.y;
		abs_point.z = stampedTransform.transform.translation.z;
		abs_pose.position = abs_point;
		abs_pose.orientation = stampedTransform.transform.rotation;
		return true;
	}
	else {
	     return false;
	}
	
}

void send_pose(float x, float y, float z, float yaw) {
	tf2::Quaternion q;
	q.setRPY(0, 0, yaw);
	geometry_msgs::Pose pose;
	pose.position.x = x;
	pose.position.y = y;
	pose.position.z = z;
	pose.orientation.x = q.x();
	pose.orientation.y = q.y();
	pose.orientation.z = q.z();
	pose.orientation.w = q.w();
	get_to_pose(pose);
}

int signum(float x) {
	return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

bool get_to_pose(geometry_msgs::Pose pose) {
	static tf2_ros::Buffer tfBuffer;
	static tf2_ros::TransformListener listener(tfBuffer);
	geometry_msgs::TransformStamped stampedTransformBaseLink;
	if (tfBuffer.canTransform ("world", "bebop2/simulation/base_link", ros::Time(0))) {
	     stampedTransformBaseLink = tfBuffer.lookupTransform("world", "bebop2/simulation/base_link", ros::Time(0));
	     float x = 0.0;
	     float y = 0.0;
	     float z = 0.0;
	     float yaw = 0.0;
	     float diffZ = pose.position.z - stampedTransformBaseLink.transform.translation.z;
	     if (abs(pose.position.z) > 0.05) {
		z = signum(pose.position.z) * 0.4;
	     }
	     if (z != 0.0) {
		     send_message(x, y, z, yaw);
		     return false;
	     }
	     float diffX = pose.position.x - stampedTransformBaseLink.transform.translation.x;
	     if (abs(pose.position.x) > 0.05) {
		x = signum(pose.position.x) * 0.4;
	     }
	     if (x != 0.0) {
		     send_message(x, y, z, yaw);
		     return false;
	     }
	     float diffY = pose.position.y - stampedTransformBaseLink.transform.translation.y;
	     if (abs(pose.position.y) > 0.05) {
		y = signum(pose.position.y) * 0.4;
	     }
	     if (y != 0.0) {
		     send_message(x, y, z, yaw);
		     return false;
	     }
	     float diffYaw = tf2::getYaw(pose.orientation) - tf2::getYaw(stampedTransformBaseLink.transform.rotation);
	     if (abs(tf2::getYaw(pose.orientation)) > 0.05) {
		yaw = signum(tf2::getYaw(pose.orientation)) * 0.2;
	     }
	     if (yaw != 0.0) {
		     send_message(x, y, z, yaw);
		     return false;
	     }
	     ROS_INFO("Desired: position x: %f, y: %f, z: %f; angular: z: %f", pose.position.x,  pose.position.y,  pose.position.z,  tf2::getYaw(pose.orientation));
	     ROS_INFO("Current: position x: %f, y: %f, z: %f; angular: z: %f", stampedTransformBaseLink.transform.translation.x,  stampedTransformBaseLink.transform.translation.y,  stampedTransformBaseLink.transform.translation.z,  tf2::getYaw(stampedTransformBaseLink.transform.rotation));
	     ROS_INFO("Differences: position x: %f, y: %f, z: %f; angular: z: %f", diffX,  diffY,  diffZ,  diffYaw);
	     return true;
	}
	else {
	     send_message(0, 0, 0, 0);	
	}
	return false;
}

void send_message(float x, float y, float z, float yaw) {
	geometry_msgs::TwistWithCovariance twist;
	twist.twist.linear.x = x;
	twist.twist.linear.y = y;
	twist.twist.linear.z = z;
	twist.twist.angular.z = yaw;
	twist_pub.publish(twist);		
}

void send_message(float x, float y, float z, float yaw, ros::Duration duration) {
	ros::Time start = ros::Time::now();
	ros::Duration time = ros::Time::now() - start;
	ros::Rate rate(10.0);
	geometry_msgs::TwistWithCovariance twist;
	twist.twist.linear.x = x;
	twist.twist.linear.y = y;
	twist.twist.linear.z = z;
	twist.twist.angular.z = yaw;
	ROS_INFO("Time: %f", ros::Time::now());
	ROS_INFO("Linear x: %f, y: %f, z: %f; angular: z: %f", twist.twist.linear.x,  twist.twist.linear.y,  twist.twist.linear.z,  twist.twist.angular.z);
	while (time < duration) {
		twist_pub.publish(twist);
		time = ros::Time::now() - start;
		rate.sleep();
	} 
}
