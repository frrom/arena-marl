
import numpy as np
import cv2 as cv
from Grid import *

class Map:
    def __init__(self, shelf_columns=3, shelf_rows=3, column_height=3, scale=11,
                path= '/home/patrick/catkin_ws/src/utils/arena-simulation-setup/maps/ignc/map.png'):
        self.shelf_columns = shelf_columns
        self.shelf_rows = shelf_rows
        self.column_height = column_height
        self.scale = scale
        self.path = path

        self.grid = self.make_rware(self.shelf_columns, self.shelf_rows, self.column_height)
        self.grid_size = self.grid.shape
        cv.imwrite(path,self.generate_map(self.upscale_grid(self.grid,self.scale), self.shelf_rows))


    def map_coord_to_image(self,array_coordinates):
        '''return a tuple for image coordinates'''
        scale_center = np.floor(self.scale/2)
        x,y = array_coordinates
        return [x*self.scale +scale_center, y*self.scale + scale_center+1]

    def upscale_grid(self,grid, n = 1):
        grid = np.kron(grid, np.ones((n,n)))

        return grid

    def make_rware(self,shelf_columns, shelf_rows, column_height):
        '''rware warehouse'''
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
        tmp_arr = arr
        tmp_arr[tmp_arr==3]=0
        _,thresh = cv.threshold(tmp_arr,1, 255,0)
        cntrs = cv.findContours(thresh.astype(np.uint8), cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
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