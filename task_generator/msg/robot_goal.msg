task_generator/crate_action crate_action 
uint8 crate_id # negative if no crate
string ns # e.g. /sim_1
string robot_id # e.g. '/robot1' or '' if no robot is assigned
string robot_type # e.g. 'burger' or '' if no robot is assigned
geometry_msgs/Pose2D robot_goal # goal where robot is headed / where it arrived
geometry_msgs/Pose2D crate_goal # goal where the crate is headed / same as robot_goal if robot has crate