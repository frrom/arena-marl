#! /usr/bin/env python3
import rospy
from std_msgs.msg import String
import argparse
import time

parser = argparse.ArgumentParser(description='Topic of simulation')
parser.add_argument('simulation_string', metavar='N',
                    help='topic of the simulation you want to visualize')
args = parser.parse_args()
data = args.simulation_string

rospy.init_node('Switcher')
sim_changer = rospy.Publisher('change_sim', String)
time.sleep(1)
sim_changer.publish(data)