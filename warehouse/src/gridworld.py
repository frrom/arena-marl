#! /usr/bin/env python3
import numpy as np
import cv2 as cv
from Grid import *
import rospkg
import os
import yaml
import rospy

class Map:
    def __init__(self, shelf_columns=3, shelf_rows=3, column_height=3, bigger_highways=False, scale=11,
                setup_name= None):
        self.shelf_columns = shelf_columns
        self.shelf_rows = shelf_rows
        self.column_height = column_height
        self.scale = scale
        self.setup_name = setup_name

        self.grid = self.make_rware(self.shelf_columns, self.shelf_rows, self.column_height)

        if bigger_highways:
            self.increase_highways()

        self.grid_size = self.grid.shape

        self.map = self.generate_map(self.upscale_grid(self.grid,self.scale), self.column_height)
        self.map = self.map2Occupancy()

        rospack = rospkg.RosPack()
        self.path2setup = os.path.join(rospack.get_path('arena-simulation-setup'),'maps')

        self.make_setup_folder()
        

    def make_setup_folder(self):
        '''create a setup folder including map.yaml, map.world.yaml, 
            grid as grid.npy and the map.png in arena-simulation-setup
            Folder will be named gridworld+(custom_name)
            '''
        if self.setup_name is None:
            self.setup_name = 'gridworld'#_sc'+str(self.shelf_columns)+'_sr'+str(self.shelf_rows)+'_ch'+str(self.column_height)
        setup_pth = os.path.join(self.path2setup,self.setup_name)

        if not os.path.isdir(setup_pth):
            print('path already exists')
            print('overwrite map...')

            os.mkdir(setup_pth)

        map_world_yaml = {'properties': {'velocity_iterations': 10, 'position_iterations': 10},
                    'layers': [{'name': 'static', 'map': 'map.yaml', 'color': [0, 1, 0, 1]}]}
        map_yaml = {'image': 'map.png', 'resolution': 10.0, 'origin': [0, 0, 0.0],
                    'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196}    

        
        with open(setup_pth+'/map.world.yaml', 'w') as file:
            documents1= yaml.dump(map_world_yaml, file)

        with open(setup_pth+'/map.yaml', 'w') as file:
            documents2 = yaml.dump(map_yaml, file)

        cv.imwrite(os.path.join(self.path2setup, self.setup_name, 'map.png'),self.map)    
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

        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        column_height = column_height
        grid = np.zeros((2, *grid_size), dtype=np.int32)
        goals = [
            (grid_size[1] // 2 - 1, grid_size[0] - 1),
            (grid_size[1] // 2, grid_size[0] - 1),
        ]

        highways = np.zeros(grid_size, dtype=np.int32)

        highway_func = lambda x, y: (
            (x % 3 == 0)  # vertical highways
            or (y % (column_height + 1) == 0)  # horizontal highways
            or (y == grid_size[0] - 1)  # delivery row
            or (  # remove a box for queuing
                (y > grid_size[0] - (column_height + 3))
                and ((x == grid_size[1] // 2 - 1) or (x == grid_size[1] // 2))
            )
        )
        for x in range(grid_size[1]):
            for y in range(grid_size[0]):
                highways[y, x] = highway_func(x, y)
        
        for x,y in goals:
            highways[y,x]=FREE_GOAL

        highways[highways==0]=FREE_SHELF
        highways[highways==1]=EMPTY

        return highways

    def generate_map(self,arr, shelf_rows, include_vert_seperators=True, include_hor_seperators=True):
        '''
            creates the map.png from grid array
            draw seperators between the individual shelf cells
        '''

        tmp_arr = arr
        tmp_arr[tmp_arr==3]=0
        _,thresh = cv.threshold(tmp_arr,1, 255,0)
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
                cv.line(img_with_seperators, line[0], line[1],0, 1)

        if include_hor_seperators:
            for line in lines_hori:
                cv.line(img_with_seperators, line[0], line[1],0, 1)
        
        return img_with_seperators

    def increase_highways(self):
        ''' We need to build a bigger highway so more robots can move between shelfs'''
        w,h = self.grid.shape
        added_col_counter = 0
        #add additional zeros cols
        highway_cols = np.where(np.sum(self.grid, 0)==0)[0] # axis needs testing
        
        for i in highway_cols:
            self.grid = np.insert(self.grid, i+added_col_counter, 0, axis=1)
            added_col_counter+=1
        
        added_col_counter = 0
        highway_rows = np.where(np.sum(self.grid, 1)==0)[0] # axis needs testing
        
        for j in highway_rows:
            self.grid = np.insert(self.grid, j+added_col_counter, 0, axis=0)
            added_col_counter+=1

    def map2Occupancy(self):
        return np.clip(self.map, -1, 100)        



import roslaunch
from nav_msgs.msg import OccupancyGrid


if __name__ == '__main__':
    shelf_rows = rospy.get_param('/map_creator/shelf_rows')
    shelf_cols = rospy.get_param('/map_creator/shelf_cols')
    col_height = rospy.get_param('/map_creator/col_height')
    bigger_highways = rospy.get_param('/map_creator/bigger_highways')


    grid = Map(shelf_columns=shelf_cols, shelf_rows=shelf_rows, column_height=col_height, bigger_highways=bigger_highways)
    grid.increase_highways()




    