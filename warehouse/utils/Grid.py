#! /usr/bin/env python3

EMPTY = 0.
CRATE = 1.
WALL = 2.
FREE_GOAL = 3.
OCCUPIED_GOAL = 4.
FREE_SHELF = 5.
OCCUPIED_SHELF = 6.

class Grid:
    grid = None
    has_walls = False