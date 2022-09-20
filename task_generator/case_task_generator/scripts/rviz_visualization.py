#%%
#!/usr/bin/env python

import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import math
from gridworld import Map
from Grid import *
from task_gen1 import TaskManager, Robot
import numpy as np
import copy
from Crate import CrateStack

WALL_COLOR=(0,0,0)
FREE_SHELF_COLOR=(0.7,0.7,1)
OCCUPIED_SHELF_COLOR=(0,0,1)
FREE_GOAL_COLOR =(1,0.5,0)
OCCUPIED_GOAL_COLOR=(1,0,0)
CRATE_COLOR=(0,1,0)
path= 'map.png'

class Visualizer:
    def __init__(self):
        #TODO: replace this with a subscriber to /gridworld, taskmanager should update /gridworld
        #Add your path to ignc map
        self.map = Map(path=path)
        topic = 'visualization_marker_array'
        self.publisher = rospy.Publisher(topic, MarkerArray, queue_size=self.map.grid_size[0]*self.map.grid_size[1])
        rospy.init_node('markers_patrick')

        self.markerArray = MarkerArray()
        self.map_ID2COLOR={'1': CRATE_COLOR,
                           '2': WALL_COLOR,
                           '3': FREE_GOAL_COLOR,
                           '4': OCCUPIED_GOAL_COLOR,
                           '5': FREE_SHELF_COLOR,
                           '6': OCCUPIED_SHELF_COLOR}

    def create_marker(self, x,y, color):
        #rviz.x = arr.y
        # rviz.y = arr.x
        #need to switch x,y when calling this function 
        r,g,b = color
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.scale.x = self.map.scale-1
        marker.scale.y = self.map.scale-1
        marker.scale.z = 1
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1  

        return marker

    def delete_all_marker(self):
        r,g,b = (0,0,0)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.CUBE
        marker.action = marker.DELETEALL
        marker.scale.x = self.map.scale-1
        marker.scale.y = self.map.scale-1
        marker.scale.z = 1
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0.1  
        self.markerArray.markers.append(marker)

    def add_markers(self):
        grid = np.flip(self.map.grid)
        indices = np.stack(np.where(grid>0),1)
        for x,y in indices:
            grid_id = grid[x,y]
            x,y = self.map.map_coord_to_image([x,y])
            color = self.map_ID2COLOR[str(int(grid_id))]
            self.markerArray.markers.append(self.create_marker(y,x,color))  


    def main(self):
        MARKERS_MAX = 100
        step = 0
        working_grid = copy.deepcopy(self.map.grid)
        while not rospy.is_shutdown():

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
            self.publisher.publish(self.markerArray)
            #self.map.grid = np.random.randint(0, 6, size=(self.map.grid_size[0], self.map.grid_size[1]))
            
            self.map.grid = self.demo(copy.deepcopy(working_grid), step)
            print(step)

            print('------------------')
            print(self.map.grid, working_grid)
            print('------------------')
            step = (step+1)%7
            rospy.sleep(2)

    def demo(self, grid, step=0):
        g = Grid()
        g.grid = grid
        
        robot = Robot('Facu', 0)
        tm = TaskManager(g)
        #print(tm.g.grid)
        if step==0:
            return tm.g.grid

        tm.generate_new_task('pack')
        if step==1:
            return tm.g.grid
        tm.active_crates[0]

        crate_index, goal = tm.pickup_crate(tm.active_crates[0].current_location, robot.name)
        robot.crate_index = crate_index
        robot.goal = goal
        if step==2:
            return tm.g.grid

        tm.drop_crate(robot.crate_index, robot.goal)
        if step==3:
            return tm.g.grid

        tm.generate_new_task('unpack')

        if step==4:
            return tm.g.grid
        tm.active_crates._crate_map.items()
        crate_index, goal = tm.pickup_crate(tm.active_crates[0].current_location, robot.name)
        robot.crate_index = crate_index
        robot.goal = goal
        if step==5:
            return tm.g.grid

        tm.drop_crate(robot.crate_index, robot.goal)
        if step==6:
            return tm.g.grid

        tm.empty_delivered_goal()
        tm.active_crates
        if step==7:
            return tm.g.grid
#%%
vis = Visualizer()

#%%
if __name__ == "__main__":
    vis = Visualizer()
    vis.main()
# %%
