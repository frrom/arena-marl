#%%
import numpy as np
import numpy.ma as ma
from utils.warehouse.src.Grid_utils import *
import matplotlib.pyplot as plt
import cv2 as cv
# configs

import numpy as np
pixels_per_meter = 100
resolution = 1/pixels_per_meter
grid_cell_size = 1 # in meters
min_shelf_separation = 2 # in meters
clearing_for_goal = 4
shelf_size = 1
possible_layouts = ['horizontal', 'vertical']
possible_separations = ['max', 'center', 'near', 'far']


# hyper-parameters

size = (10, 10) # meters
size_pixels = tuple(a/resolution for a in size)
layout = 'vertical'  
nr_shelf_groups = 2
separation = 'center'
nr_goals = 5

# functions

def add_walls(grid):
    grid.has_walls = True
    grid.grid = np.pad(grid.grid, pad_width=((1,1), (1,1)), mode='constant', constant_values=WALL)
    
    return grid

def remove_walls(grid):
    grid.has_walls = False
    grid.grid = grid.grid[1:-1, 1:-1]

    return grid

def add_goals(grid):
    if layout == 'vertical':
        return _add_goals_vertical(grid)
    else:
        return NotImplementedError

def _add_goals_vertical(grid):
    row = -2 if grid.has_walls else -1

    m = grid.grid.shape[1]//2 # middle
    if grid.grid.shape[1] % 2 == 0:
        grid.grid[row, m-1:m+1] = FREE_GOAL
    else:
        grid.grid[row, m-1:m+2] = FREE_GOAL


    return grid

def add_shelves(grid):
    if layout == 'vertical':
        return _add_shelves_vertical(grid)
    else:
        return NotImplementedError

def _add_shelves_vertical(grid):
    if separation == 'center':
        return _add_shelves_vertical_center(grid)
    else:
        return NotImplementedError

def _add_shelves_vertical_center(grid):
    nr_rows = size[0] - clearing_for_goal # 3 meter clearing from wall
    row_min = 0 # start from other wall
    nr_cols = (nr_shelf_groups * shelf_size) + (nr_shelf_groups - 1)*min_shelf_separation
    col_min = (size[1] - nr_cols)//2

    if grid.has_walls:
        cols = np.arange(col_min+1, col_min + 1 + nr_cols + 1, 1 + min_shelf_separation)
        rows = slice(row_min+1, row_min + 1 + nr_rows + 1, 1)
    else:
        cols = np.arange(col_min, col_min + nr_cols + 1, 1 + min_shelf_separation)
        rows = slice(row_min, row_min + 1 + nr_rows, 1)

    m = grid.grid.shape[1]//2 # middle
    if grid.grid.shape[1] % 2 == 0:
        grid.grid[rows, tuple(cols)] = FREE_SHELF
    else:
        grid.grid[rows, tuple(cols)] = FREE_SHELF

    return grid

def upscale_grid(grid, n = pixels_per_meter):

    grid.grid = np.kron(grid.grid, np.ones((n,n)))

    return grid

def make_rware(shelf_columns, shelf_rows, column_height):
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

    def upscale_grid(self,grid, n = pixels_per_meter):
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
    
    

# script
if __name__ == "__main__":
    grid_size = np.array(size)/grid_cell_size
    g = Grid()
    g.grid = np.zeros(grid_size.astype(np.int32))

    #g = add_walls(g)
    g = add_goals(g)
    g = add_shelves(g)

    shelf_columns=3
    shelf_rows=3
    column_height= 3
    g.grid = make_rware(shelf_columns,shelf_rows,column_height)
    path = '/home/patrick/catkin_ws/src/utils/arena-simulation-setup/maps/ignc/map.png'

    print(g.grid.shape)
    scale = 11
    print(upscale_grid(g,scale).grid.shape)
    #cv.imwrite(path, generate_map(upscale_grid(g,scale).grid, shelf_rows))
    print('saved')
    #plt.imshow(make_rware(3,2,2))
    map = Map()
    #np.save('wh1', g.grid)




# %%
#plt.imshow(upscale_grid(g, 100))

# %%