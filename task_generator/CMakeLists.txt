cmake_minimum_required(VERSION 2.8.3)
project(task_generator)

find_package(catkin REQUIRED COMPONENTS
        rospy
        message_generation
        std_msgs
        actionlib_msgs
        geometry_msgs
)
catkin_python_setup()

add_message_files(
  FILES
  crate_action.msg
  robot_goal.msg
  robot_goal_list.msg
)

generate_messages(
  DEPENDENCIES
  task_generator
  std_msgs
  actionlib_msgs
  geometry_msgs
)



catkin_package(
        CATKIN_DEPENDS message_runtime std_msgs actionlib_msgs geometry_msgs
        )