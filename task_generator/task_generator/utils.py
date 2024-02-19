import math
import numpy as np
from nav_msgs.msg import OccupancyGrid
import random
from geometry_msgs.msg import Pose2D


def generate_freespace_indices(map_: OccupancyGrid) -> tuple:
    """generate the indices(represented in a tuple) of the freesapce based on the map

    Returns:
        indices_y_x(tuple): indices of the non-occupied cells, the first element is the y-axis indices,
        the second element is the x-axis indices.
    """
    width_in_cell, height_in_cell = map_.info.width, map_.info.height
    map_2d = np.reshape(map_.data, (height_in_cell, width_in_cell))
    indices_y_x = np.where(map_2d == 0)
    return indices_y_x


def get_random_pos_on_map(free_space_indices, map_: OccupancyGrid, safe_dist: float, forbidden_zones: list = None, no_zones: list = None, start_pos = None, max_dist = 1000, dont_care = False):
    """
    Args:
        indices_y_x(tuple): a 2 elementary tuple stores the indices of the non-occupied cells, the first element is the y-axis indices,
            the second element is the x-axis indices.
        map (OccupancyGrid): map proviced by the ros map service
        forbidden_zones (list of 3 elementary tuple(x,y,r)): a list of zones which is forbidden
    Returns:
       x_in_meters,y_in_meters,theta
    """

    def is_pos_valid(x_in_meters, y_in_meters, start_pos, max_dist):
        for forbidden_zone in forbidden_zones:
            if (
                forbidden_zone
                and (x_in_meters - forbidden_zone[0]) ** 2 + (y_in_meters - forbidden_zone[1]) ** 2
                < (forbidden_zone[2] + safe_dist) ** 2
            ):
                return False
        for forbidden_zone in no_zones:
            if (forbidden_zone and (forbidden_zone[0]-forbidden_zone[2] < x_in_meters < forbidden_zone[0]+forbidden_zone[2])
                and (forbidden_zone[1]-forbidden_zone[3]< y_in_meters < forbidden_zone[1]+forbidden_zone[3])):
                return False
        if start_pos is not None and np.sqrt((x_in_meters-start_pos.x)**2 + (y_in_meters-start_pos.y)**2) > max_dist:
            return False
        # in pixel
        cell_radius = int(safe_dist / map_.info.resolution)
        x_index = int((x_in_meters - map_.info.origin.position.x) // map_.info.resolution)
        y_index = int((y_in_meters - map_.info.origin.position.y) // map_.info.resolution)

        # check occupancy around (x_index,y_index) with cell_radius
        # TODO use numpy for checking
        for i in range(x_index - cell_radius, x_index + cell_radius, 1):
            for j in range(y_index - cell_radius, y_index + cell_radius, 1):
                index = j * map_.info.width + i
                if index >= len(map_.data):
                    return False
                try:
                    value = map_.data[index]
                except IndexError:
                    print("IndexError: index: %d, map_length: %d" % (index, len(map_.data)))
                    return False
                if value != 0:

                    return False
        return True

    assert len(free_space_indices) == 2 and len(free_space_indices[0]) == len(
        free_space_indices[1]
    ), "free_space_indices is not correctly setup"
    if forbidden_zones is None:
        forbidden_zones = []
    if no_zones is None:
        no_zones = []

    n_freespace_cells = len(free_space_indices[0])
    pos_valid = False
    n_check_failed = 0
    x_in_meters, y_in_meters = None, None
    while not pos_valid:
        idx = random.randint(0, n_freespace_cells - 1)
        # in cells
        y_in_cells, x_in_cells = free_space_indices[0][idx], free_space_indices[1][idx]
        # convert x, y in meters
        y_in_meters = y_in_cells * map_.info.resolution + map_.info.origin.position.y
        x_in_meters = x_in_cells * map_.info.resolution + map_.info.origin.position.x
        if dont_care:
            break
        pos_valid = is_pos_valid(x_in_meters, y_in_meters, start_pos, max_dist)
        if not pos_valid:
            n_check_failed += 1
            if n_check_failed > 400:
                print("no valid position found")
                max_dist = 1.1*max_dist if max_dist else None
                forbidden_zones = [(forbidden[0], forbidden[1], forbidden[2]*0.8) for forbidden in forbidden_zones if forbidden]
                return get_random_pos_on_map(free_space_indices, map_, safe_dist/2, forbidden_zones=forbidden_zones, no_zones = no_zones, start_pos=start_pos, max_dist=max_dist)
                return 0,0,0
                raise Exception("can't find any no-occupied space please check the map information")
        # in radius
    theta = random.uniform(-math.pi, math.pi)

    return x_in_meters, y_in_meters, theta
