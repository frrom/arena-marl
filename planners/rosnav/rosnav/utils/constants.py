import rospy

RosnavEncoder = {
    "RobotSpecificEncoder": {
        "lasers": rospy.get_param("laser/num_beams"),
        "meta": 2 + 3 # Goal + Vel
    },
    "UniformEncoder": {
        "lasers": 1200,
        "meta": 2 + 3 + 1 + 6, # Goal + Vel + Radius + Max Vel 
        "maxVelocity": {
            "x": [-5, 5],
            "y": [-5, 5],
            "angular": [-10, 10]
        }
    },
    "WarehouseEncoder": {
        "lasers": rospy.get_param("laser/num_beams"),
        "meta": int(rospy.get_param("observable_task_goals"))*2 + 3 + 3 # Goal + Vel
    }
}
