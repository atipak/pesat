#include <sstream>
#include <position_pid/bebop2_hardware_interface.h>
#include <joint_limits_interface/joint_limits_interface.h>
#include <joint_limits_interface/joint_limits.h>
#include <joint_limits_interface/joint_limits_urdf.h>
#include <joint_limits_interface/joint_limits_rosparam.h>

using namespace hardware_interface;
using joint_limits_interface::JointLimits;
using joint_limits_interface::SoftJointLimits;
using joint_limits_interface::PositionJointSoftLimitsHandle;
using joint_limits_interface::PositionJointSoftLimitsInterface;

namespace bebop2_hardware_interface
{
    Bebop2HardwareInterface::Bebop2HardwareInterface(ros::NodeHandle& nh) : nh_(nh) {
        init();
        controller_manager_.reset(new controller_manager::ControllerManager(this, nh_));
        nh_.param("/bebop2/hardware_interface/loop_hz", loop_hz_, 0.1);
        ros::Duration update_freq = ros::Duration(1.0/loop_hz_);
        non_realtime_loop_ = nh_.createTimer(update_freq, &Bebop2HardwareInterface::update, this);
    }

    Bebop2HardwareInterface::~Bebop2HardwareInterface() {

    }

    void Bebop2HardwareInterface::init() {
        // Get joint names
        nh_.getParam("/bebop2/hardware_interface/joints", joint_names_);
        num_joints_ = joint_names_.size();

        // Resize vectors
        joint_position_.resize(num_joints_);
        joint_velocity_.resize(num_joints_);
        joint_effort_.resize(num_joints_);
        joint_position_command_.resize(num_joints_);
        joint_velocity_command_.resize(num_joints_);
        joint_effort_command_.resize(num_joints_);

        // Initialize Controller
        for (int i = 0; i < num_joints_; ++i) {

             // Create joint state interface
            JointStateHandle jointStateHandle(joint_names_[i], &joint_position_[i], &joint_velocity_[i], &joint_effort_[i]);
             joint_state_interface_.registerHandle(jointStateHandle);

            // Create position joint interface
            JointHandle jointPositionHandle(jointStateHandle, &joint_position_command_[i]);
            JointLimits limits;
            SoftJointLimits softLimits;
            getJointLimits(joint_names_[i], nh_, limits);
            PositionJointSoftLimitsHandle jointLimitsHandle(jointPositionHandle, limits, softLimits);
            positionJointSoftLimitsInterface.registerHandle(jointLimitsHandle);
            position_joint_interface_.registerHandle(jointPositionHandle);

            // Create effort joint interface
            //JointHandle jointEffortHandle(jointStateHandle, &joint_effort_command_[i]);
            //effort_joint_interface_.registerHandle(jointEffortHandle);

            // Create velocity joint interface
            JointHandle jointVelocityHandle(jointStateHandle, &joint_velocity_command_[i]);
            velocity_joint_interface_.registerHandle(jointVelocityHandle);
        }

        // create topic for publishing commands
        pub_ = nh_.advertise<pesat_msgs::JointStates>("/bebop2/current_commands", 5);
        joints_positions_sub_ = nh_.subscribe("/bebop2/current_positions", 1, &Bebop2HardwareInterface::callback_positions, this);
        joints_velocities_sub_ = nh_.subscribe("/bebop2/current_velocities", 1, &Bebop2HardwareInterface::callback_velocities, this);

        registerInterface(&joint_state_interface_);
        registerInterface(&position_joint_interface_);
        //registerInterface(&effort_joint_interface_);
        registerInterface(&velocity_joint_interface_);
        registerInterface(&positionJointSoftLimitsInterface);
    }

    void Bebop2HardwareInterface::update(const ros::TimerEvent& e) {
        elapsed_time_ = ros::Duration(e.current_real - e.last_real);
        read();
        controller_manager_->update(ros::Time::now(), elapsed_time_);
        write(elapsed_time_);
    }

    void Bebop2HardwareInterface::read() {
        for (int i = 0; i < num_joints_; i++) {
            joint_position_[i] = current_positions_.values[i];
            joint_velocity_[i] = current_velocities_.values[i];
        }
    }

    void Bebop2HardwareInterface::write(ros::Duration elapsed_time) {
        positionJointSoftLimitsInterface.enforceLimits(elapsed_time);
        pesat_msgs::JointStates joints_states;
        for (int i = 0; i < num_joints_; i++) {
            joints_states.values.push_back(joint_velocity_command_[i]);
        }
        pub_.publish(joints_states);
    }

    void Bebop2HardwareInterface::callback_positions(const pesat_msgs::JointStatesConstPtr& joints)
    {
        current_positions_ = *joints;
    }

    void Bebop2HardwareInterface::callback_velocities(const pesat_msgs::JointStatesConstPtr& joints)
    {
        current_velocities_ = *joints;
    }

}