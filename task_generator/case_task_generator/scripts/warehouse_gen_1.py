#%%
import numpy as np
import numpy.ma as ma
from .Grid import *
import matplotlib.pyplot as plt
import rospkg

# configs
#%%
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


    
    

# script

grid_size = np.array(size)/grid_cell_size
g = Grid()
g.grid = np.zeros(grid_size.astype(np.int32))

print(g.grid.shape)
#g = add_walls(g)
g = add_goals(g)
g = add_shelves(g)
g.grid = np.flip(make_rware(3,2,2), axis= 0)
plt.imshow(g.grid, origin='lower')

np.save(f"{rospkg.RosPack().get_path('arena-simulation-setup')}maps/warehouse_1_cases/map.npy", g.grid)
from PIL import Image
img = np.rint(upscale_grid(g,100).grid/5*255).astype(np.uint8)
im = Image.fromarray(np.flip(img, axis= 0))
#/home/u20/MARL_ws/src/forks/arena-simulation-setup/maps/warehouse_1_cases/
im.save(f"{rospkg.RosPack().get_path('arena-simulation-setup')}maps/warehouse_1_cases/map.png")



# %%
plt.imshow(upscale_grid(g.grid, 100))

# %%
