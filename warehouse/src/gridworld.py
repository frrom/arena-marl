#! /usr/bin/env python3
import numpy as np
import cv2 as cv
from Grid import *
import rospkg
import os
import yaml
import rospy
import copy 

#from map_generator_node.py
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String

import subprocess

class Map:
    def __init__(
                self, shelf_columns=3,
                shelf_rows=3,
                column_height=3,
                bigger_highways=False,
                scale=100,
                setup_name= None,
                random_map=False,
                additional_goals = None
                ):

        #Grid Shape Arguments
        self.shelf_columns = shelf_columns
        self.shelf_rows = shelf_rows
        self.column_height = column_height
        self.scale = scale
        
        #Addtional Map Changes
        self.bigger_highways = bigger_highways
        self.additional_goals = None

        #Meta Stuff
        self.setup_name = setup_name
        self.random_map = random_map
        self.grid = None 
        self.grid_size = None 
        self.map = None 
        self.occupancy_grid = OccupancyGrid()

        #Subscriber & Publishers
        rospack = rospkg.RosPack()
        self.path2setup = os.path.join(rospack.get_path('arena-simulation-setup'),'maps')
        
        rospy.init_node('gridworld')
        self.gridpub = rospy.Publisher('gridworld_base', OccupancyGrid, queue_size=1)
        
        print(self.random_map)
        if self.random_map:
            #self.setup_name = 'gridworld_random'
            rospy.Subscriber('/map', OccupancyGrid, self.get_occupancy_grid)
            rospy.Subscriber('/demand', String, self.new_episode_callback) # generate new random map for the next episode when entering new episode
            self.mappub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)
        else:
            self.make_gridworld()
            self.make_setup_folder()
            self.y_origin = 0 #- 4
            self.x_origin = 0 #- self.grid.shape[1]/2
            self.gridpub.publish(self.create_gridworld_message())
        

    def make_gridworld(self):
        self.grid = self.make_rware(self.shelf_columns, self.shelf_rows, self.column_height)

        if self.bigger_highways:
            self.increase_highways()
        
        if self.additional_goals is not None:
            print("add additional goals")
            self.add_goals_NWO(self.additional_goals )

        self.grid = self.add_walls(self.grid)
        print(self.grid)
        normal = True
        #print(self.scale)
        #print(self.column_height)
        if "crossroad" in self.setup_name:
            self.map = self.generate_map2(self.scale, self.shelf_rows)
            #self.map = self.generate_map(self.upscale_grid(np.flip(self.grid,axis=0),self.scale), self.column_height)
        elif "empty" in self.setup_name:
            self.map = self.generate_map3(self.scale, self.shelf_rows)
        else:
            self.map = self.generate_map(self.upscale_grid(np.flip(self.grid,axis=0),self.scale), self.column_height)
        self.grid_size = self.grid.shape


    def make_setup_folder(self):
        '''create a setup folder including map.yaml, map.world.yaml, 
            grid as grid.npy and the map.png in arena-simulation-setup
            Folder will be named gridworld+(custom_name)
            '''
        if self.setup_name is None:
            self.setup_name = 'default_gridworld'#_sc'+str(self.shelf_columns)+'_sr'+str(self.shelf_rows)+'_ch'+str(self.column_height)
        setup_pth = os.path.join(self.path2setup,self.setup_name)

        if not os.path.isdir(setup_pth):
            print('path already exists')
            print('overwrite map...')

            os.mkdir(setup_pth)

        self.y_origin = 0
        self.x_origin = 0
        map_world_yaml = {'properties': {'velocity_iterations': 10, 'position_iterations': 10},
                    'layers': [{'name': 'static', 'map': 'map.yaml', 'color': [0, 1, 0, 1]}]}
        map_yaml = {'image': 'map.png', 'resolution': 0.01, 'origin': [self.x_origin, self.y_origin, 0.0],
                    'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196, 'corridors' : self.shelf_rows}    

        print(setup_pth)
        with open(setup_pth+'/map.world.yaml', 'w') as file:
            documents1= yaml.dump(map_world_yaml, file, default_flow_style=None)

        with open(setup_pth+'/map.yaml', 'w') as file:
            documents2 = yaml.dump(map_yaml, file)

       
        cv.imwrite(os.path.join(self.path2setup, self.setup_name, 'map.png'), self.map)    
        np.save(setup_pth+'/grid', self.grid)
        
        
    def map_coord_to_image(self,array_coordinates):
        '''returns the coordinates of the grid on the image'''
        scale_center = np.floor(self.scale/2)
        x,y = array_coordinates
        return [x*self.scale +scale_center, y*self.scale + scale_center+1]

    def upscale_grid(self,grid, n = 1):
        grid = np.kron(grid, np.ones((n,n)))

        return grid

    def make_rware(self,shelf_columns, shelf_rows, column_height):
        '''Implementation of the warehouse from rware
            shelf columns: Number of shelfs vertically
            shelf_rows: Number of shelfs horizontally
            column_height: Number of Shelf spots 

            return: numpy array of the warehouse
            '''

        #assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        grid_size = (
            (column_height+1)*shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        #print(grid_size)
        column_height = column_height
        grid = np.zeros((2, *grid_size), dtype=np.int32)
        goals = [
            (grid_size[1] // 2 - 1, grid_size[0] - 1),
            (grid_size[1] // 2, grid_size[0] - 1),
        ]

        highways = np.zeros(grid_size, dtype=np.int32)

        highway_func = lambda x, y: (
            (x % 3 == 0)  # vertical highways
            or (y % (column_height + 1 ) == 0)  # horizontal highways
            or (y == grid_size[0] - 1)  # delivery row
            or (  # remove a box for queuing
                (y > (grid_size[0] - 3)) and
                ((x == grid_size[1] // 2 - 1) or (x == grid_size[1] // 2))
            )
        )
        for x in range(grid_size[1]):
            for y in range(grid_size[0]):
                highways[y, x] = highway_func(x, y)
        #print(highways)
        for x,y in goals:
            highways[y,x]=FREE_GOAL

        highways[highways==0]=FREE_SHELF
        highways[highways==1]=EMPTY

        return highways

    def generate_map(self, arr, shelf_rows, include_vert_seperators=True, include_hor_seperators=True):
        '''
            creates the map.png from grid array
            draw seperators between the individual shelf cells
        '''
        
        tmp_arr = copy.deepcopy(arr)
        tmp_arr[tmp_arr==FREE_GOAL]=0
        tmp_arr[tmp_arr==WALL]=0
        
        _,thresh = cv.threshold(tmp_arr,1, 255,0)
        print("thresh:")
        print(thresh.astype(np.uint8),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        cntrs, _ = cv.findContours(thresh.astype(np.uint8), cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        lines_vert = []
        lines_hori = []
        
        img_with_seperators = np.zeros_like(arr)
        img_with_seperators += 255

        for cnt in cntrs:
            #add seperator line horizontally for each shelf
            x,y,w,h = cv.boundingRect(cnt)
            line_start = (int(x+w/2), int(y))
            line_end = (int(x+w/2), int(y+h))
            lines_vert.append((line_start, line_end))

            #add seperator line horizontally for each shelf
            shelf_vertical_distance = h/shelf_rows

            for i in range(0,shelf_rows+1):
                shelf_seperation_horizontal_start = (int(x),int(y+shelf_vertical_distance*i)) 
                shelf_seperation_horizontal_end = (int(x+w),int(y+shelf_vertical_distance*i)) 
                lines_hori.append((shelf_seperation_horizontal_start, shelf_seperation_horizontal_end))

        #print(lines_hori)
        #print(lines_vert)
        if include_vert_seperators:
            for line in lines_vert:
                cv.line(img_with_seperators, line[0], line[1],0, 5)

        if include_hor_seperators:
            for line in lines_hori:
                cv.line(img_with_seperators, line[0], line[1],0, 5)
        img_with_seperators[arr == WALL] = 0
        
        return img_with_seperators

    def generate_map2(self, scale, rows=5, corridor_width=150, include_vert_separators=True, include_hor_separators=True):
        #tmp_arr = copy.deepcopy(arr)
        tmp_arr = np.zeros((scale*10,scale*10))
        tmp_arr[0:50,:] = 2
        tmp_arr[-50:-1,:]= 2
        tmp_arr[:,0:50] = 2
        tmp_arr[:,-50:-1] = 2
        #tmp_arr[tmp_arr != WALL] = 0
        _, thresh = cv.threshold(tmp_arr, 1, 255, 0)
        contours, _ = cv.findContours(thresh.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img_with_separators = np.zeros_like(tmp_arr)
        ver = rows // 2 + 1
        hor = rows//2 + rows%2 + 1
        print(ver, hor)
        print(img_with_separators)
        x, y, w, h = 0,0,0,0
        #cv.rectangle(img_with_separators, (corridor_start, y), (corridor_end, y + h), 255, thickness=-1)
        for cnt in contours:
            #print(cnt)
            xn,yn,wn,hn = cv.boundingRect(cnt)
            if xn >= x:
                x, y, w, h = xn, yn, wn, hn
            #x, y, w, h = cv.boundingRect(cnt)
            print(x,y,w,h)
            # Add vertical corridor lines
        for i in range(ver-1):
            cv.rectangle(img_with_separators, (int(x + (i+1)/ver*w - corridor_width / 2), y), 
                            (int(x + (i+1)/ver*w + corridor_width / 2), y + h), 255, thickness=-1)
        for i in range(hor-1):
            cv.rectangle(img_with_separators, (x, int(y + (i+1)/hor*h - corridor_width / 2)),
                                (x + w, int(y +  (i+1)/hor*h + corridor_width / 2)), 255, thickness=-1)
            

            # Add horizontal corridor lines for each shelf
            # shelf_vertical_distance = h / shelf_rows
            # for i in range(shelf_rows + 1):
            #     if include_hor_separators:
            #         cv.rectangle(img_with_separators, (x, int(y + shelf_vertical_distance * i - corridor_width / 2)),
            #                     (x + w, int(y + shelf_vertical_distance * i + corridor_width / 2)), 255, thickness=-1)
        #img_with_separators[arr == WALL] = 0
        return img_with_separators
    
    def generate_map3(self, scale, rows=5, corridor_width=150, include_vert_separators=True, include_hor_separators=True):
        tmp_arr = np.zeros((scale*10,scale*10))
        tmp_arr[0:50,:] = 2
        tmp_arr[-50:-1,:]= 2
        tmp_arr[:,0:50] = 2
        tmp_arr[:,-50:-1] = 2
        #tmp_arr[tmp_arr != WALL] = 0
        _, thresh = cv.threshold(tmp_arr, 1, 255, 0)
        contours, _ = cv.findContours(thresh.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img_with_separators = np.zeros_like(tmp_arr)
        x, y, w, h = 0,0,0,0
        for cnt in contours:
            xn,yn,wn,hn = cv.boundingRect(cnt)
            if xn > x:
                x, y, w, h = xn, yn, wn, hn
                print(x,y,w,h)
                cv.rectangle(img_with_separators, (w,y), 
                (x,h), 255, thickness=-1)
        

        return img_with_separators
    def increase_highways(self):
        ''' We need to build a bigger highway so more robots can move between shelfs'''
        w,h = self.grid.shape
        added_col_counter = 0
        #add additional zeros cols
        highway_cols = np.where(np.sum(self.grid, 0)==0)[0] # axis needs testing
        highway_cols2 = np.where(np.sum(self.grid, 0)==3)[0]
        highway_cols = np.hstack((highway_cols, highway_cols2))
        for i in highway_cols:
            self.grid = np.insert(self.grid, i+added_col_counter, 0, axis=1)
            added_col_counter+=1
        added_col_counter = 0
        highway_rows = np.where(np.sum(self.grid, 1)==0)[0] # axis needs testing
        for j in highway_rows:
            self.grid = np.insert(self.grid, j+added_col_counter, 0, axis=0)
            added_col_counter+=1

    def create_map_message(self):
        w,h= self.map.shape
        occgrid = OccupancyGrid()
        occgrid.header.frame_id = 'map'
        occgrid.info.resolution = 10.0
        occgrid.info.width = w
        occgrid.info.height = h

        occgrid.info.origin.orientation.w = 1
        print(self.map.min())
        occgrid.data = (self.map/255).astype(np.int32).flatten()
        return occgrid

    def create_gridworld_message(self):
        w,h = self.grid.shape
        occ_grid = OccupancyGrid()
        occ_grid.info.width = w
        occ_grid.info.height = h
        occ_grid.data = self.grid.flatten()
        return occ_grid

    def get_occupancy_grid(self, occgrid_msg: OccupancyGrid): # a bit cheating: copy OccupancyGrid meta data from map_server of initial map
        #self.occupancy_grid = occgrid_msg
        #dont need the message
        return occgrid_msg

    def new_episode_callback(self, msg: String):
        
        #print('old grid', self.grid.shape)
        print('we are in random')
        if self.random_map:
            self.shelf_rows = np.random.randint(1,5)*2 + 1
            self.shelf_columns = np.random.randint(1,5)*2 + 1
            self.column_height = np.random.randint(2,8)
            
            print(self.shelf_rows,  self.shelf_columns, self.column_height)
            self.make_gridworld()
            self.make_setup_folder()
            

            self.gridpub.publish(self.create_gridworld_message())
            self.occupancy_grid= self.create_map_message()

            print(self.occupancy_grid.data)
            print('new grid', self.grid.shape)
            print()
            # rospy.loginfo("New random map generated for episode {}.".format(self.nr))
            self.mappub.publish(self.occupancy_grid)

            try:   
                bashCommand = "rosservice call /move_base/clear_costmaps"
                subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                rospy.loginfo("New random map published and costmap cleared.")
            except rospy.ROSException as e:
                print(e)

    def add_walls(self,grid):
        grid = np.pad(grid, pad_width=((1,1), (1,1)), mode='constant', constant_values=WALL) 
        return grid

    def add_goals_NWO(self, grid, distance_goal_shelf = 3, orientation='T'):
        h,w = self.grid.shape
        w_half = int(w/2)
        h_half = int(h/2)
        if orientation == 'T':
            goal_row = np.zeros((distance_goal_shelf,w))
            goal_row[0,w_half-1:w_half+1] = FREE_GOAL
            self.grid = np.vstack([goal_row, self.grid])

        elif orientation == 'L':
            goal_col = np.zeros((h,distance_goal_shelf))
            goal_col[w_half-1:w_half+1, 0] = FREE_GOAL
            self.grid = np.hstack([goal_row, self.grid])

        elif orientation == 'R':
            goal_col = np.zeros((h,distance_goal_shelf))
            goal_col[w_half-1:w_half+1, 2] = FREE_GOAL
            self.grid = np.hstack([self.grid, goal_row])


            


import roslaunch
from nav_msgs.msg import OccupancyGrid
import time
import sys

if __name__ == '__main__':
    grid_args = sys.argv[1:8]
    shelf_cols, shelf_rows, col_height, scale  = [int(a.split(':=')[-1]) for a in grid_args[0:4]]
    bigger_highways, rand_map = [bool(a.split(':=')[-1]) for a in grid_args[4:6]]
    additional_goals = grid_args[6]
    rand_map = False
    folder = sys.argv[8].split(':=')[-1]
    print(sys.argv[8])

    grid = Map( shelf_columns=shelf_cols,
                shelf_rows=shelf_rows,
                column_height=col_height, 
                bigger_highways=bigger_highways,
                scale=scale,
                setup_name= folder,
                random_map = rand_map,
                additional_goals = additional_goals
                )

    time.sleep(2)

    if sys.argv[-1].split(':=')[0]=='__log':
        if rand_map:
            msg = String()
            grid.new_episode_callback(msg)

        #Launch the remaining node
        launch_args = sys.argv[8:-2]
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)

        cli_args = [f"{rospkg.RosPack().get_path('warehouse')}/launch/rest.launch"] + launch_args
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], launch_args)]
        launch = roslaunch.parent.ROSLaunchParent(uuid,roslaunch_file )
        launch.start()
        rospy.loginfo("started")


        while not rospy.is_shutdown():
            grid.gridpub.publish(grid.create_gridworld_message())
        rospy.spin()
            






    
