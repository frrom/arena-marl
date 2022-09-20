#%%
import numpy as np
from .Grid import *
from matplotlib import pyplot as plt
from .Crate import CrateStack, Crate
from typing import List, Dict, Union, Literal, TypedDict, Any
from task_generator.task_generator.robot_manager import RobotManager
from geometry_msgs.msg import Pose2D
from actionlib_msgs.msg import GoalStatusArray, GoalStatus, GoalID
#%%


class CaseTaskManager():
    def __init__(self, grid_save_path: np.ndarray, name: str = 'default', num_active_tasks: int= 5):
        """
        params:
            spawn_rate - every how many seconds a new crate is added to spawn queue
        """
        g = np.load(grid_save_path)
        self.grid_save_path = grid_save_path
        self.g = g.copy()
        self.active_crates = CrateStack(name)
        self.crate_index_to_robots:  Dict[int, RobotManager]= {}
        self._num_active_tasks = num_active_tasks # simultaneos active tasks to post in task manager
    

    ## MANAGE QUADRANTS ##
    def _get_random_quadrant_of_type(self, quadrant_type, nr_quadrants= 1, random=True):
        grid_inds = self._find(quadrant_type)
        if grid_inds.shape[0] <= 0:
            return np.array([])
        ind = np.random.permutation(np.arange(grid_inds.shape[0]))[:nr_quadrants]
        return grid_inds[ind, :]

    def _get_quadrant_type(self, coords: np.ndarray):
        return self.g[coords[0], coords[1]]

    def _free_quadrant(self, coords: np.ndarray, remove_crate: bool= False):
        if remove_crate:
            self.active_crates.remove(coords)
        quadrant_type = self._get_quadrant_type(coords)
        self.g[coords[0], coords[1]] = FREE_GOAL if quadrant_type == OCCUPIED_GOAL else FREE_SHELF

    def _occupy_quadrant(self, coords: np.ndarray, goal: bool= None, spawn_crate: bool= False):
        if spawn_crate:
            self.active_crates.add(coords, goal)
        quadrant_type = self._get_quadrant_type(coords)
        self.g[coords[0], coords[1]] = OCCUPIED_GOAL if quadrant_type == FREE_GOAL else OCCUPIED_SHELF


    def _find(self, quadrant_type):
        return np.argwhere(self.g == quadrant_type)

    ## MOVE CRATES ##
    def _can_spawn_crate(self):
        return self._find(FREE_GOAL).size > 0 # True if there is a FREE_GOAL, False otherwise

    def _spawn_crates(self, nr_crates: int= 1, goals: np.ndarray= None):
        grid_inds = self._find(FREE_GOAL)[:nr_crates,:] # Find free goals and restrict to however many crates we want to spawn
        if nr_crates > grid_inds.shape[0]:
            print(f"Can't spawn {nr_crates} because there are only {grid_inds.shape[0]} free Goals. Spawning {grid_inds.shape[0]} crates instead.")
            nr_crates = grid_inds.shape[0]
        
        if goals is None:
            goals = self._get_random_quadrant_of_type(FREE_SHELF, nr_crates)

        for start_coords, goal in zip(grid_inds, goals):
            self._occupy_quadrant(start_coords, goal, spawn_crate= True)

    def _spawn_crate_manual(self, starting_point: int, goal: np.ndarray):
        self._occupy_quadrant(starting_point, goal, True)


    def _move_from_to(self, current_location: np.ndarray, new_location: np.ndarray, remove_crate: bool= False):
        self.active_crates.move_crate((current_location), new_location)
        self._free_quadrant(current_location)
        self._occupy_quadrant(new_location)
        if remove_crate:
            self._free_quadrant(new_location, remove_crate= True)


    ## TASK GENERATORS ##
    def _generate_pack_task(self, goal: np.ndarray= None):
        if not self._can_spawn_crate():
            print('No free goals to spawn crate in')
            return False
        else:
            self._spawn_crates(1, goal)
            return True


    def _generate_unpack_task(self, force_free_goal= True):
        goal = self._get_random_quadrant_of_type(FREE_GOAL).squeeze()
        if not goal.size > 0:
            if not force_free_goal:
                print('WARNING: The goal is occupied!')
                goal = self._get_random_quadrant_of_type(OCCUPIED_GOAL).squeeze()
            else:
                print('No free goals')
                return False

        crate_location = self._get_random_quadrant_of_type(FREE_SHELF).squeeze()
        if not crate_location.size > 0:
            print('No free shelf')
            return False

        self._occupy_quadrant(crate_location, goal, spawn_crate= True)
        
        crate = self.active_crates.get_crate_at_location(crate_location)
        crate.set_new_goal(goal)

        return True

    def _generate_manual_task(self, starting_point: np.ndarray, goal: np.ndarray):
        if self.g[starting_point] not in [OCCUPIED_GOAL, OCCUPIED_SHELF]:
            print('No crate at starting_point. No task was generated.')
            return False
        elif self.g[goal] not in [FREE_GOAL, FREE_SHELF, EMPTY]:
            print('goal is occupied. No task was generated.')
            return False
        else:
            self._spawn_crate_manual(starting_point, goal)
            return True



    def pose2d_to_numpy(self, pose: Pose2D) -> np.ndarray:
        return np.ndarray([pose.x-0.5, pose.y-0.5])
    
    def numpy_to_pose2d(self, coords: np.ndarray, theta= 0) -> Pose2D:
        pose = Pose2D(coords[1]+0.5, coords[0]+0.5, theta)
        return pose



    ## PUBLIC FUNCTIONS ## 

    def generate_new_task(self, type: Literal['pack', 'unpack', 'manual'], **kwargs):
        if type not in ['pack', 'unpack', 'manual']:
            raise ValueError('Assignment not implemented')
        if type == 'pack':
            return self._generate_pack_task()            
        elif type == 'unpack':
            return self._generate_unpack_task(kwargs.get('force_free_goal', True))
        elif type == 'manual':
            return self._generate_manual_task(kwargs.get('starting_point'), kwargs.get('goal')) # passed argument like this, because this is meant to only be used explicitly.
        else:
            raise ValueError(f'type: {type} is not recognized. Accepted types are "pack" and "unpack".')

    def pickup_crate(self, crate_location: Union[Pose2D, np.ndarray], robot_manager: RobotManager) -> Crate:
        if type(crate_location) is Pose2D:
            crate_location = self.pose2d_to_numpy(crate_location)
        crate = self.active_crates.pickup_crate(crate_location)
        self._free_quadrant(crate_location)
        self.crate_index_to_robots[crate.index] = robot_manager

        return crate

    def drop_crate(self, crate_index: int, drop_location: Union[Pose2D, np.ndarray]) -> Union[int, None]:
        if type(drop_location) is Pose2D: 
            drop_location = self.pose2d_to_numpy(drop_location)
        drop_successful = self.active_crates.drop_crate(crate_index, drop_location)
        if drop_successful:
            self._occupy_quadrant(drop_location) 
            return self.crate_index_to_robots.pop(crate_index)
        
    def get_transit_crates(self):
        return self.active_crates._in_transit

    def get_open_tasks(self, resolution= 1, generate= False):
        """
        gets currently open tasks, if generate=True will generate new ones in the case of not having any available
        """
        crate_ids = []
        crate_locations = []
        crate_goals = []

        for crate in self.active_crates:
            if not crate.delivered:
                crate_ids.append(crate.index)
                crate_locations.append(self.numpy_to_pose2d(crate.current_location/resolution))
                crate_goals.append(self.numpy_to_pose2d(crate.goal/resolution))
        
        if generate:
            nr_tasks = np.clip(self._num_active_tasks - len(crate_ids), 0)
            self.empty_delivered_goal()
            self.generate_scenareo(nr_tasks, reset= False)
            return self.get_open_tasks(generate=False) # explicitly pass false in case it just gets stuck in loop

        
        return crate_ids, crate_locations, crate_goals

    def empty_delivered_goal(self, goal: Union[Pose2D, np.ndarray]= None):
        if type(goal) is Pose2D:
            goal = self.pose2d_to_numpy(goal)
        if goal is None:
            all_goals = self._find(OCCUPIED_GOAL)
            for g in all_goals:
                if self.active_crates.get_crate_at_location(g).delivered: # if crate is delivered
                    self._free_quadrant(g, True)
        
        else:
            self._free_quadrant(goal, True)

    def generate_scenareo(self, nr_tasks: int, type: Literal['random', 'manual']= 'random', reset= True, **kwargs):
        if reset:
            self.reset()
        if type == 'random':
            tasks = {True: 'pack', False: 'unpack'}

            generated_tasks = 0
            while generated_tasks < nr_tasks:
                free_goals = self._find(FREE_GOAL)
                task_type = free_goals.shape[0] > 1
                success = self.generate_new_task(tasks[task_type])
                generated_tasks += 1
                
        elif type == 'manual':
            raise NotImplementedError
            starts, goals = kwargs.get('starts'), kwargs.get('goals')
            if starts is None or goals is None:
                raise ValueError('starts and goals have to be passed to run Manual task')
            
    def reset(self):
        self.g = np.load(self.grid_save_path)
        self.active_crates = CrateStack(self.active_crates.name)
        self.crate_index_to_robots:  Dict[int, RobotManager]= {}
    
 



class Robot:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.crate_index = None
        self.goal = None



# file = 'wh1.npy'

# robot = Robot('Facu', 0)
# tm = CaseTaskManager(np.load(file))
# plt.imshow(tm.g.grid)
# #%%
# tm.generate_new_task('pack')
# plt.imshow(tm.g.grid)
# tm.active_crates[0]
# #%%
# crate_index, goal = tm.pickup_crate(tm.active_crates[0].current_location, robot)
# robot.crate_index = crate_index
# robot.goal = goal
# plt.imshow(tm.g.grid)
# #%%
# tm.drop_crate(robot.crate_index, robot.goal)
# plt.imshow(tm.g.grid)
# #%%
# tm.generate_new_task('unpack')
# plt.imshow(tm.g.grid)
# tm.active_crates._crate_map.items()
# #%%
# crate_index, goal = tm.pickup_crate(tm.active_crates[0].current_location, robot)
# robot.crate_index = crate_index
# robot.goal = goal
# plt.imshow(tm.g.grid)
# #%%
# tm.drop_crate(robot.crate_index, robot.goal)
# plt.imshow(tm.g.grid)
# #%%
# tm.empty_delivered_goal()
# plt.imshow(tm.g.grid)
# tm.active_crates


# # %%
# from nav_msgs.msg import OccupancyGrid
# import rospy
# import cv2
# import numpy as np

# var = None

# def store_in_np(map: OccupancyGrid):
#     arr = np.array(map.data)
#     print(map.info)
#     width = map.info.width
#     height = map.info.height

#     global var
#     arr = arr.reshape(height, width)
#     var = np.zeros([height, width, 3])
#     var[:,:,0] = arr*64/255.0
#     var[:,:,1] = arr*128/255.0
#     var[:,:,2] = arr*192/255.0

#     cv2.imwrite('color_img.jpg', var)
#     cv2.imshow("image", var)
#     cv2.waitKey()


# rospy.init_node('my_node')
# sub = rospy.Subscriber('/map', OccupancyGrid, store_in_np)


# # %%
# from geometry_msgs.msg import Pose2D
# a = Pose2D()
# a.x = 1
# a.y = 1
# a.theta = 0

# b = Pose2D()
# b.x = 1
# b.y = 1
# b.theta = 1

# print(a == b)
# b.theta = 0
# print(a == b)

# d = {}
# d[a] = b
# # %%

# %%
