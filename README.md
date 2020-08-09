# Persuer-evader search and track

## How to launch algorithm
### Installation
Lists of packages which are needed for successful launch of project.   

#### A new random world creation
Python packages for python interpreter
1. cv2
2. deap
3. matplotlib
4. numpy
5. pyoctree
6. scipy
7. geometry_msgs (ROS instalation)
8. sensor_msgs (ROS instalation)
9. shapely
10. pesat


#### Simulation
ROS packages 
0. ROS Melodic
1. catkin_simple
2. geographic_info
3. geometric_shapes
4. mav_comm
5. moveit
6. rotors_simulator
7. rpg_public_dronet
8. rviz_visual_tools
9. srdfdom
10. unique_identifier
11. ros_control
12. control_msgs
13. control_toolbox
14. four_wheel_steering_msgs
15. ros_controllers
16. urdf_geometry_parser
17. geometry2
18. pid
19. moveit_msgs
20. moveit_resources
21. moveit_tutorials
22. moveit_visual_tools
23. pesat

### World creation
There is building_blocks.py python source code in package helper_pkg/src/helper_pkg which is used to create a new random world. After launch this python code all configuration files are replaced and new world is used in simulation. This script doesnt have console interface and parameters of new world has to be changed inside code. 

create_new_map(True, True, 100, 0.1, -1, "n", "n") - (substitute_paths, create_section_file, size [x * 50, where x is whole positive number], built [0.1 - 0.3], height of obstacles [-1 (random), 1-20], position of drone and target ["f", "n"], dynamical_obstacle ["y", "n"])

### Simulation
There are 3 launch files in Pesat_core package. Next launch file has to be launched after finish of initialization of previous one.

drone.launch file is used for launching simulator, drone and MoveIt instance for drone.

target.launch file is used for launching target, target logic and MoveIt instance for target.

logic.launch file is used for launching logic of drone - dynamic avoidance, searching, tracking

#### Launch process
Commands:
1. roscore
2. roslaunch pesat_core drone.launch - wait until all components are initialized (there are no logging information)
3. roslaunch pesat_core target.launch - wait until all components are initialized (there are no logging information)
4. manually unpause Gazebo simulator - we have to initialize MoveIt (which is dependent on simulation time) so we dont get error messages about unavaible component - wait until all obstacles are added into planning scene
5. manually pause Gazebo simulator
6. roslaunch pesat_core logic.launch - wait until message "Section algorithm loaded" is shown
7. launch target logic by sending message: 'rostopic pub /user/target/strategy std_msgs/Int64 "data: 6"'
8. manually unpause Gazebo simulator


##x Configuration
- pesat_resources/config/*_configuration.yaml
- environment, target, logic, drone -- name = type of configuration

### Target logic 
Sending message 'rostopic pub /user/target/strategy std_msgs/Int64 "data: 6"'



