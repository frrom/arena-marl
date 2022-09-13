#! /usr/bin/env python3
import numpy as np
from Grid import *
from matplotlib import pyplot as plt
from Crate import CrateStack
from typing import List, Dict, Union, Literal, TypedDict, Any

import rospy
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

class Robot:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.crate_index = None
        self.goal = None

class TaskManager():
    def __init__(self, g: Grid, name: str = 'default'):
        self.g = g
        self.active_crates = CrateStack(name)
        self.crate_index_to_robots:  Dict[int, Robot]= {}

    ## MANAGE QUADRANTS ##
    def _get_random_quadrant_of_type(self, quadrant_type, nr_quadrants= 1, random=True):
        grid_inds = self._find(quadrant_type)
        if grid_inds.shape[0] <= 0:
            return np.array([])
        ind = np.random.permutation(np.arange(grid_inds.shape[0]))[:nr_quadrants]
        return grid_inds[ind, :]

    def _get_quadrant_type(self, coords: np.ndarray):
        return self.g.grid[coords[0], coords[1]]

    def _free_quadrant(self, coords: np.ndarray, remove_crate: bool= False):
        if remove_crate:
            self.active_crates.remove(coords)
        quadrant_type = self._get_quadrant_type(coords)
        self.g.grid[coords[0], coords[1]] = FREE_GOAL if quadrant_type == OCCUPIED_GOAL else FREE_SHELF

    def _occupy_quadrant(self, coords: np.ndarray, goal: bool= None, spawn_crate: bool= False):
        if spawn_crate:
            self.active_crates.add(coords, goal)
        quadrant_type = self._get_quadrant_type(coords)
        self.g.grid[coords[0], coords[1]] = OCCUPIED_GOAL if quadrant_type == FREE_GOAL else OCCUPIED_SHELF

    def _find(self, quadrant_type):
        return np.argwhere(self.g.grid == quadrant_type)

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
        else:
            self._spawn_crates(1, goal)


    def _generate_unpack_task(self):
        if self.active_crates.isempty():
            print('No stashed crates to unpack.')

        else:
            crate_location = self._get_random_quadrant_of_type(OCCUPIED_SHELF).squeeze()
            goal = self._get_random_quadrant_of_type(FREE_GOAL).squeeze()
            if not goal.size > 0:
                print('No free goals')
            else:
                crate = self.active_crates.get_crate_at_location(crate_location)
                crate.set_new_goal(goal)

    def _generate_manual_task(self, starting_point: np.ndarray, goal: np.ndarray):
        if self.g.grid[starting_point] not in [OCCUPIED_GOAL, OCCUPIED_SHELF]:
            print('No crate at starting_point. No task was generated.')
        elif self.g.grid[goal] not in [FREE_GOAL, FREE_SHELF, EMPTY]:
            print('goal is occupied. No task was generated.')
        else:
            self._spawn_crate_manual(starting_point, goal)



    ## PUBLIC FUNCTIONS ## 
    def generate_new_task(self, type: Literal['pack', 'unpack', 'manual'], **kwargs):
        if type not in ['pack', 'unpack', 'manual']:
            raise ValueError('Assignment not implemented')
        if type == 'pack':
            self._generate_pack_task()            
        elif type == 'unpack':
            self._generate_unpack_task()
        elif type == 'manual':
            self._generate_manual_task(kwargs.get('starting_point'), kwargs.get('goal')) # passed argument like this, because this is meant to only be used explicitly.
        else:
            raise ValueError(f'type: {type} is not recognized. Accepted types are "pack" and "unpack".')

    def pickup_crate(self, crate_location: np.ndarray, robot_name: str='Default'):
        crate_index, goal = self.active_crates.pickup_crate(crate_location)
        self._free_quadrant(crate_location)
        self.crate_index_to_robots[crate_index] = robot_name

        return crate_index, goal

    def drop_crate(self, crate_index: int, drop_location: np.ndarray) -> Union[int, None]:
        drop_successful = self.active_crates.drop_crate(crate_index, drop_location)
        if drop_successful:
            self._occupy_quadrant(drop_location) 
            return self.crate_index_to_robots.pop(crate_index)
        
    def get_in_transit_crates(self):
        return self.active_crates._in_transit

    def empty_delivered_goal(self, goal: np.ndarray= None):
        if goal is None:
            all_goals = self._find(OCCUPIED_GOAL)
            for g in all_goals:
                if self.active_crates.get_crate_at_location(g).delivered: # if crate is delivered
                    self._free_quadrant(g, True)
        
        else:
            self._free_quadrant(goal, True)


    ###Patrick Stuff for demo

    def callback(self,data):
        w = rospy.get_param('/gridworld/width')
        h = rospy.get_param('/gridworld/height')
        self.g.grid = np.reshape(data.data, [w,h]).astype(np.int32)

    def main(self):   
        robot = Robot('Facu', 0)
        #Test Map publisher
        pub = rospy.Publisher('gridworld', numpy_msg(Floats), queue_size=10)
        
        #Map Listener
        subscriber = rospy.Subscriber("gridworld_base", numpy_msg(Floats), self.callback)


        rospy.init_node('task_manager_f', anonymous=True)
        print('----initialized nodes---')
        data = rospy.wait_for_message('/gridworld_base', numpy_msg(Floats))
        subscriber.unregister()

        while not rospy.is_shutdown():
            rospy.sleep(1)
            self.generate_new_task('pack')
            
            pub.publish(self.g.grid.astype(np.float32).flatten())
            rospy.sleep(1)
            

            self.active_crates[0]
        
            crate_index, goal = self.pickup_crate(self.active_crates[0].current_location, robot.name)
            robot.crate_index = crate_index
            robot.goal = goal

            pub.publish(self.g.grid.astype(np.float32).flatten())
            rospy.sleep(1)
            
        
            self.drop_crate(robot.crate_index, robot.goal)

            pub.publish(self.g.grid.astype(np.float32).flatten())
            rospy.sleep(1)
            

            self.generate_new_task('unpack')

            pub.publish(self.g.grid.astype(np.float32).flatten())
            rospy.sleep(1)
            

            self.active_crates._crate_map.items()
            
            crate_index, goal = self.pickup_crate(self.active_crates[0].current_location, robot.name)
            robot.crate_index = crate_index
            robot.goal = goal
            
            pub.publish(self.g.grid.astype(np.float32).flatten())
            rospy.sleep(1)
            
        
            self.drop_crate(robot.crate_index, robot.goal)
            
            pub.publish(self.g.grid.astype(np.float32).flatten())
            rospy.sleep(1)
            
            
            self.empty_delivered_goal()
            self.active_crates

            pub.publish(self.g.grid.astype(np.float32).flatten())
            rospy.sleep(1)
            



if __name__ == '__main__':
    try:
        tm = TaskManager(Grid())
        tm.main()
    except (rospy.ROSException, Exception) as e:
        print('#############')
        print(e)
        exit()