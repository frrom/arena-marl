#! /usr/bin/env python3

import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import rospkg
import yaml
from Grid import *
from task_gen1 import TaskManager, Robot
import numpy as np
import copy
from Crate import CrateStack

import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import sys

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from task_generator.msg import robot_goal, crate_action, robot_goal_list

WALL_COLOR=(0,0,0)
FREE_SHELF_COLOR=(0.7,0.7,1)
OCCUPIED_SHELF_COLOR=(0,0,1)
FREE_GOAL_COLOR =(1,0.5,0)
OCCUPIED_GOAL_COLOR=(1,0,0)
CRATE_COLOR=(0,1,0)
path= '/home/patrick/catkin_ws/src/utils/arena-simulation-setup/maps/ignc/map.png'

class Visualizer:
    def __init__(self, ns=1, yaml=None):
        #TODO: replace this with a subscriber to /gridworld, taskmanager should update /gridworld
        
        #switching between envs
        self.ns=ns
        self.ns_prefix = f"/{ns}/" if ns else ""
        self._curr_stage = 1

        #yaml resolution and origin information
        self.resolution = yaml['resolution']
        self.origin = yaml['origin']
        path = rospkg.RosPack().get_path('arena-simulation-setup')+'/maps/gridworld/grid.npy'
        self.map_original = np.load(path)#None#Map(path=path)
        self.map = copy.deepcopy(self.map_original)

        self.scale = 100 #get path to image and extract size image_size[0]/map_size[0] = scale
        

        self.new_map_received = False
        self.markerArray = MarkerArray()
        self.map_ID2COLOR={'1': CRATE_COLOR,
                           '2': WALL_COLOR,
                           '3': FREE_GOAL_COLOR,
                           '4': OCCUPIED_GOAL_COLOR,
                           '5': FREE_SHELF_COLOR,
                           '6': OCCUPIED_SHELF_COLOR}

        # subs for triggers
        self._sub_next = rospy.Subscriber(
            f"{self.ns_prefix}next_stage", Bool, self.next_stage
        )
        self._sub_previous = rospy.Subscriber(
            f"{self.ns_prefix}previous_stage", Bool, self.previous_stage
        )

        
    def goal_list_to_map(self, data):
        print('________goal list recieved_________')
        self.new_map_received = True
        tmp_map = copy.deepcopy(self.map_original)
        for robot_goal_ in data.open_tasks:
            idx = robot_goal_.robot_goal
            print(idx)
            shelf_or_goal = tmp_map[int(idx.y), int(idx.x)]
            if shelf_or_goal == FREE_GOAL:
                tmp_map[int(idx.y), int(idx.x)]=OCCUPIED_GOAL
            elif shelf_or_goal == FREE_SHELF:
                tmp_map[int(idx.y), int(idx.x)]=OCCUPIED_SHELF
            else:
                print('out:',self.map[int(idx.y), int(idx.x)])
        self.map = tmp_map

            

    def next_stage(self, *args, **kwargs):
        if self._curr_stage < len(self._stages):
            self._curr_stage = self._curr_stage + 1
            self._initiate_stage()

            if self.ns == "eval_sim":
                rospy.set_param("/curr_stage", self._curr_stage)
                # if not rospy.get_param("debug_mode"):
                #     with self._lock_json:
                #         self._update_curr_stage_json()

                if self._curr_stage == len(self._stages):
                    rospy.set_param("/last_stage_reached", True)
        else:
            print(
                f"({self.ns}) INFO: Tried to trigger next stage but already reached last one"
            )

    def previous_stage(self, *args, **kwargs):
        if self._curr_stage > 1:
            rospy.set_param("/last_stage_reached", False)

            self._curr_stage = self._curr_stage - 1
            self._initiate_stage()

            if self.ns == "eval_sim":
                rospy.set_param("/curr_stage", self._curr_stage)
                with self._lock_json:
                    self._update_curr_stage_json()
        else:
            print(
                f"({self.ns}) INFO: Tried to trigger previous stage but already reached first one"
            )

    def callback(self,data):
        w = data.info.width
        h = data.info.height
        self.map = np.reshape(np.asarray(data.data), [w,h])#.astype(np.int32)

        

    def create_marker(self, x,y, color):
        '''create a marker for rviz'''
        #rviz.x = arr.y
        # rviz.y = arr.x
        #need to switch x,y when calling this function 
        r,g,b = color
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.scale.x = self.resolution*(self.scale-1)
        marker.scale.y = self.resolution*(self.scale-1)
        marker.scale.z = 1
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x + self.origin[0]
        marker.pose.position.y = y + self.origin[1]
        marker.pose.position.z = 0.1  

        return marker

    def map_coord_to_image(self,array_coordinates):
        '''return a tuple for image coordinates'''
        scale_center = np.floor(self.scale/2)
        x,y = array_coordinates
        return [(x*self.scale+scale_center)*self.resolution, (y*self.scale+ scale_center+1)*self.resolution]

    def add_markers(self):
        '''adds markers for all shelf/goal cells'''
        grid = np.flip(self.map)
        #print(grid)
        
        indices = np.stack(np.where(grid>0),1)
        for x,y in indices:
            grid_id = grid[x,y]
            x,y = self.map_coord_to_image([x,y])
            color = self.map_ID2COLOR[str(int(grid_id))]
            self.markerArray.markers.append(self.create_marker(y,x,color))  


    def main(self):
        '''runs the node to update all the markers'''
        MARKERS_MAX = 100
        step = 0
        working_grid = copy.deepcopy(self.map)
        
        #Map Listener
        #Subscriber = rospy.Subscriber("gridworld", OccupancyGrid, self.callback)
        open_tasks = rospy.Subscriber('sim_1/open_tasks', robot_goal_list, self.goal_list_to_map)

        #Marker Publisher
        topic = 'visualization_marker_array'
        publisher = rospy.Publisher(topic, MarkerArray, queue_size=1000)

        rospy.init_node(f"{self.ns}/visualizer", anonymous=True)
        rate = rospy.Rate(10)
        print('----initialized nodes---')
        
        while not rospy.is_shutdown():
            if self.new_map_received == True:
                # We add the new marker to the MarkerArray, removing the oldest
                # marker from it when necessary
                #self.delete_all_marker()

                self.markerArray = MarkerArray()
                self.add_markers()

                id = 0
                for m in self.markerArray.markers:
                    m.id = id
                    id += 1
                # Publish the MarkerArray
                publisher.publish(self.markerArray)
                self.new_map_received = False
                
            

    def demo(self, grid, step=0):
        g = Grid()
        g.grid = grid
        
        robot = Robot('Facu', 0)
        tm = TaskManager(g)
        #print(tm.g.grid)
        if step==0:
            return tm.g.grid
        #%%
        tm.generate_new_task('pack')
        if step==1:
            return tm.g.grid
        tm.active_crates[0]
        #%%
        crate_index, goal = tm.pickup_crate(tm.active_crates[0].current_location, robot.name)
        robot.crate_index = crate_index
        robot.goal = goal
        if step==2:
            return tm.g.grid
        #%%
        tm.drop_crate(robot.crate_index, robot.goal)
        if step==3:
            return tm.g.grid
        #%%
        tm.generate_new_task('unpack')

        if step==4:
            return tm.g.grid
        tm.active_crates._crate_map.items()
        #%%
        crate_index, goal = tm.pickup_crate(tm.active_crates[0].current_location, robot.name)
        robot.crate_index = crate_index
        robot.goal = goal
        if step==5:
            return tm.g.grid
        #%%
        tm.drop_crate(robot.crate_index, robot.goal)
        if step==6:
            return tm.g.grid
        #%%
        tm.empty_delivered_goal()
        tm.active_crates
        if step==7:
            return tm.g.grid


if __name__ == "__main__":
    #mapyaml = rospy.get_param('/grid_vis/map_path')
    args = rospy.myargv(args=sys.argv)
    if len(args) != 2:
        print("no arguments found")
        ns = 'sim_1'
    else:
        ns = args[1]
    print(ns)
    mapyaml = '/'.join([rospkg.RosPack().get_path('arena-simulation-setup'), 'maps', 'gridworld', 'map.yaml'])
    #rospy.get_param('/ns')
    print('----------------'+mapyaml+'-------------------------------')
    for i in range(0,5):
        print('--------------------------------')
    with open(mapyaml, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    vis = Visualizer(ns,yaml_dict)
    vis.main()