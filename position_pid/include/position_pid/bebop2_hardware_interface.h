#ifndef ROS_CONTROL__TR1_HARDWARE_INTERFACE_H
#define ROS_CONTROL__TR1_HARDWARE_INTERFACE_H

#include <hardware_interface/joint_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <joint_limits_interface/joint_limits_interface.h>
#include <joint_limits_interface/joint_limits.h>
#include <joint_limits_interface/joint_limits_urdf.h>
#include <joint_limits_interface/joint_limits_rosparam.h>
#include <controller_manager/controller_manager.h>
#include <boost/scoped_ptr.hpp>
#include <ros/ros.h>
#include <position_pid/bebop2_hardware.h>
#include <pesat_msgs/JointStates.h>
#include <vector>

using namespace hardware_interface;
using joint_limits_interface::JointLimits;
using joint_limits_interface::SoftJointLimits;
using joint_limits_interface::PositionJointSoftLimitsHandle;
using joint_limits_interface::PositionJointSoftLimitsInterface;
using std::vector;

namespace bebop2_hardware_interface
{
    static const double POSITION_STEP_FACTOR = 10;
    static const double VELOCITY_STEP_FACTOR = 10;

    class Bebop2HardwareInterface: public bebop2_hardware_interface::Bebop2Hardware
    {
        public:
            Bebop2HardwareInterface(ros::NodeHandle& nh);
            ~Bebop2HardwareInterface();
            void init();
            void update(const ros::TimerEvent& e);
            void read();
            void callback_positions(const pesat_msgs::JointStatesConstPtr& joints);
            void callback_velocities(const pesat_msgs::JointStatesConstPtr& joints);
            void write(ros::Duration elapsed_time);

        protected:
            ros::NodeHandle nh_;
            ros::Timer non_realtime_loop_;
            ros::Duration control_period_;
            ros::Duration elapsed_time_;
            PositionJointInterface positionJointInterface;
            PositionJointSoftLimitsInterface positionJointSoftLimitsInterface;
            double loop_hz_;
            boost::shared_ptr<controller_manager::ControllerManager> controller_manager_;
            double p_error_, v_error_, e_error_;
            ros::Publisher pub_;
            ros::Subscriber joints_positions_sub_;
            ros::Subscriber joints_velocities_sub_;
            vector<float> updates_positions_;
            pesat_msgs::JointStates current_positions_;
            pesat_msgs::JointStates current_velocities_;
    };

}

#endif
