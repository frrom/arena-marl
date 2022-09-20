
EMPTY = 0.
CRATE = 1.
WALL = 2.
FREE_GOAL = 3.
OCCUPIED_GOAL = 4.
FREE_SHELF = 5.
OCCUPIED_SHELF = 6.

class Grid:
    has_walls = False
    def __init__(self, grid = None):
        self.grid = grid

    def __repr__(self):
        return repr(self.grid)