#! /usr/bin/env python3
import numpy as np
import cv2 as cv
from Grid import *
import rospkg
import os
import yaml

import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

if __name__=='__main__':
    rospack = rospkg.RosPack()
    path2setup = os.path.join(rospack.get_path('arena-simulation-setup'),'maps','gridworld', 'grid.npy')
    map = np.load(path2setup)
    

    w,h = map.shape
    rospy.init_node('gridworld')
    rate = rospy.Rate(1)
    rospy.set_param('/gridworld/width', w)
    rospy.set_param('/gridworld/height', h)

    pub = rospy.Publisher('gridworld_base', numpy_msg(Floats), queue_size=10)
    counter=1
    while not rospy.is_shutdown():
        connections = pub.get_num_connections()
        try:
            if connections>=1:
                a = map.astype(np.float32).flatten()
                pub.publish(a)
                #print('Gridworld published')
            else:
                rate.sleep()
        except rospy.ROSException as e:
            print(e)