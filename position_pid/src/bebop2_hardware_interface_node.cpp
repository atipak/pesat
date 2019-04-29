#include <position_pid/bebop2_hardware_interface.h>
int main(int argc, char** argv)
{
    ros::init(argc, argv, "bebop2_hardware_interface");
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(1);
    spinner.start();
    bebop2_hardware_interface::Bebop2HardwareInterface ROBOT(nh);
    ros::waitForShutdown();
    return 0;
}