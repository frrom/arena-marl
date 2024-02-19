#%%
from time import sleep
from typing import Dict, Type, Union, List

import json
import os
import rospkg
import rospy
import yaml
import warnings
import subprocess

from abc import ABC, abstractmethod
from enum import Enum, unique
from threading import Lock
from filelock import FileLock

from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap, SetMap
from std_msgs.msg import Bool, Int8
from geometry_msgs.msg import Pose2D, PoseStamped
import rospy
from std_srvs.srv import Empty

from .obstacles_manager import ObstaclesManager
from .robot_manager import RobotManager
from .utils import get_random_pos_on_map
import random
import math
from scipy.optimize import linear_sum_assignment

class ABSMARLTask(ABC):
    """An abstract class for the DRL agent navigation tasks."""

    def __init__(
        self,
        obstacles_manager: ObstaclesManager,
        robot_manager: Dict[str, List[RobotManager]],
    ):
        self.obstacles_manager = obstacles_manager
        self.robot_manager = robot_manager
        self._service_client_get_map = rospy.ServiceProxy("/static_map", GetMap)
        self._map_lock = Lock()
        rospy.Subscriber("/map", OccupancyGrid, self._update_map)
        # a mutex keep the map is not unchanged during reset task.

    @abstractmethod
    def reset(self):
        """
        Funciton to reset the task/scenery. Make sure that _map_lock is used.
        """

    def _update_map(self, map_: OccupancyGrid):
        if self.obstacles_manager is not None:
            with self._map_lock:
                self.obstacles_manager.update_map(map_)
                for manager in self.robot_manager:
                    for rm in self.robot_manager[manager]:
                        rm.update_map(map_)

    def set_obstacle_manager(self, manager: ObstaclesManager):
        assert type(manager) is ObstaclesManager
        if self.obstacles_manager is not None:
            warnings.warn(
                "TaskManager was already initialized with a ObstaclesManager. "
                "Current ObstaclesManager will be overwritten."
            )
        self.obstacles_manager = manager


def count_robots(obstacles_manager_dict: Dict[str, List[RobotManager]]) -> int:
    return sum(len(manager) for manager in obstacles_manager_dict.values())


class RandomMARLTask(ABSMARLTask):
    """Sets a randomly drawn start and goal position for each robot episodically."""

    def __init__(
        self,
        obstacles_manager: ObstaclesManager,
        robot_manager: Dict[str, List[RobotManager]],
    ):
        super().__init__(obstacles_manager, robot_manager)
        print("using random tasks")
        self._num_robots = (
            count_robots(self.robot_manager) if type(self.robot_manager) is dict else 0
        )
        self.reset_flag = (
            {key: False for key in self.robot_manager.keys()}
            if type(self.robot_manager) is dict
            else {}
        )

    def add_robot_manager(self, robot_type: str, managers: List[RobotManager]):
        assert type(managers) is list
        if not self.robot_manager:
            self.robot_manager = {}
        self.robot_manager[robot_type] = managers
        self._num_robots = count_robots(self.robot_manager)
        self.reset_flag = {key: False for key in self.robot_manager.keys()}

    def reset(self, robot_type: str):
        assert robot_type in self.robot_manager, f"Unknown robot type: {robot_type},"
        f" robot has to be one of the following types: {self.robot_manager.keys()}"

        self.reset_flag[robot_type] = True
        if not all(self.reset_flag.values()):
            return

        with self._map_lock:
            max_fail_times = 5
            fail_times = 0
            while fail_times < max_fail_times:
                try:
                    starts, goals = [None] * self._num_robots, [None] * self._num_robots
                    robot_idx = 0
                    for _, robot_managers in self.robot_manager.items():
                        for manager in robot_managers:
                            start_pos, goal_pos = manager.set_start_pos_goal_pos(
                                forbidden_zones=starts
                            )
                            print(goal_pos)
                            starts[robot_idx] = (
                                start_pos.x,
                                start_pos.y,
                                manager.ROBOT_RADIUS * 2.25,
                            )
                            goals[robot_idx] = (
                                goal_pos.x,
                                goal_pos.y,
                                manager.ROBOT_RADIUS * 2.25,
                            )
                            robot_idx += 1
                    self.obstacles_manager.reset_pos_obstacles_random(
                        forbidden_zones=starts + goals
                    )
                    break
                except rospy.ServiceException as e:
                    rospy.logwarn(repr(e))
                    fail_times += 1
            if fail_times == max_fail_times:
                raise Exception("reset error!")

        self.reset_flag = dict.fromkeys(self.reset_flag, False)


class StagedMARLRandomTask(RandomMARLTask):
    """
    Enforces the paradigm of curriculum learning.
    The training stages are defined in 'training_curriculum.yaml'
    """

    def __init__(
        self,
        ns: str,
        obstacles_manager: ObstaclesManager = None,
        robot_manager: Dict[str, List[RobotManager]] = None,
        start_stage: int = 1,
        curriculum_file_path: str = None,
    ) -> None:
        super().__init__(obstacles_manager, robot_manager)
        self.ns = ns
        self.ns_prefix = f"/{ns}/" if ns else ""
        print("using staged MARL-task")
        self._curr_stage = start_stage
        self._stages = {}
        # self._PATHS = PATHS
        training_path  = rospkg.RosPack().get_path("training")
        self.curriculum_file = os.path.join(
            training_path, "configs", "training_curriculums", curriculum_file_path
        )
        self._read_stages_from_yaml()

        # check start stage format
        if not isinstance(start_stage, int):
            raise ValueError("Given start_stage not an Integer!")
        if self._curr_stage < 1 or self._curr_stage > len(self._stages):
            raise IndexError(
                "Start stage given for training curriculum out of bounds! Has to be between {1 to %d}!"
                % len(self._stages)
            )
        rospy.set_param("/curr_stage", self._curr_stage)

        # hyperparamters.json location
        # self.json_file = os.path.join(self._PATHS.get("model"), "hyperparameters.json")
        # if not rospy.get_param("debug_mode"):
        #     assert os.path.isfile(self.json_file), (
        #         "Found no 'hyperparameters.json' at %s" % self.json_file
        #     )

        # self._lock_json = FileLock(f"{self.json_file}.lock")

        # subs for triggers
        self._sub_next = rospy.Subscriber(
            f"{self.ns_prefix}next_stage", Bool, self.next_stage
        )
        self._sub_previous = rospy.Subscriber(
            f"{self.ns_prefix}previous_stage", Bool, self.previous_stage
        )

        self._initiate_stage()

    def next_stage(self, *args, **kwargs):
        if self._curr_stage < len(self._stages):
            self._curr_stage = self._curr_stage + 1
            self._initiate_stage()

            if self.ns == "eval_sim":
                rospy.set_param("/curr_stage", self._curr_stage)
                # if not rospy.get_param("debug_mode"):
                #     with self._lock_json:
                #         self._update_curr_stage_json()

                if self._curr_stage == len(self._stages):
                    rospy.set_param("/last_stage_reached", True)
        else:
            print(
                f"({self.ns}) INFO: Tried to trigger next stage but already reached last one"
            )

    def previous_stage(self, *args, **kwargs):
        if self._curr_stage > 1:
            rospy.set_param("/last_stage_reached", False)

            self._curr_stage = self._curr_stage - 1
            self._initiate_stage()

            if self.ns == "eval_sim":
                rospy.set_param("/curr_stage", self._curr_stage)
                with self._lock_json:
                    self._update_curr_stage_json()
        else:
            print(
                f"({self.ns}) INFO: Tried to trigger previous stage but already reached first one"
            )

    def _initiate_stage(self):
        #!/usr/bin/env python

    # import rospy
    # from std_srvs.srv import Empty
    # from nav_msgs.srv import SetMap

    # rospy.init_node('map_server_reinitializer')

    # def reinitialize_map_server(new_map_path):
    #     # Call the "reset" service of the "map_server" node
    #     rospy.wait_for_service('/map_server/reset')
    #     reset_service = rospy.ServiceProxy('/map_server/reset', Empty)
    #     reset_service()

    #     # Reinitialize the "map_server" node with the new map file
    #     rospy.wait_for_service('/map_server/set_map')
    #     set_map_service = rospy.ServiceProxy('/map_server/set_map', SetMap)
    #     new_map = open(new_map_path, 'r').read()
    #     set_map_service(new_map)

    # # Call the reinitialize_map_server function with the path to the new map file
    # new_map_path = '/path/to/new/map.yaml'
    # reinitialize_map_server(new_map_path)

        """

        rospy.init_node('map_server_param_modifier')

        new_map_path = '/path/to/new/map.yaml'

        rospy.set_param('/map_server/yaml_filename', new_map_path)

        """
        if self.obstacles_manager is None:
            return
        self._remove_obstacles()

        static_obstacles = self._stages[self._curr_stage]["static"]
        dynamic_obstacles = self._stages[self._curr_stage]["dynamic"]

        self.obstacles_manager.register_random_static_obstacles(
            self._stages[self._curr_stage]["static"]
        )
        self.obstacles_manager.register_random_dynamic_obstacles(
            self._stages[self._curr_stage]["dynamic"]
        )

        print(
            f"({self.ns}) Stage {self._curr_stage}:"
            f"Spawning {static_obstacles} static and {dynamic_obstacles} dynamic obstacles!"
        )

    def _read_stages_from_yaml(self):
        file_location = self.curriculum_file
        if os.path.isfile(file_location):
            with open(file_location, "r") as file:
                self._stages = yaml.load(file, Loader=yaml.FullLoader)
            assert isinstance(
                self._stages, dict
            ), "'training_curriculum.yaml' has wrong fromat! Has to encode dictionary!"
        else:
            raise FileNotFoundError(
                "Couldn't find 'training_curriculum.yaml' in %s "
                % self._PATHS.get("curriculum")
            )

    def _update_curr_stage_json(self):
        # with open(self.json_file, "r") as file:
        #     hyperparams = json.load(file)
        # try:
        #     hyperparams["curr_stage"] = self._curr_stage
        # except Exception as e:
        #     raise Warning(
        #         f" {e} \n Parameter 'curr_stage' not found in 'hyperparameters.json'!"
        #     )
        # else:
        #     with open(self.json_file, "w", encoding="utf-8") as target:
        #         json.dump(hyperparams, target, ensure_ascii=False, indent=4)
        pass

    def _remove_obstacles(self):
        self.obstacles_manager.remove_obstacles()

from ..case_task_generator.scripts.task_gen1 import CaseTaskManager as ctm
from task_generator.msg import robot_goal, crate_action, robot_goal_list
import numpy as np

class CasesMARLTask(ABSMARLTask):

    
    """Sets a randomly drawn start and goal position for each robot episodically."""

    def __init__(
        self,
        obstacles_manager: ObstaclesManager,
        robot_manager: Dict[str, List[RobotManager]],
        ns: str,
        start_stage: int = 1,
        map_path: str = None,
        curriculum_file_path: str = None,
    ):
        self.ns = ns
        map_path = rospy.get_param("/world_path") if map_path == None else map_path
        self.env = map_path.split('/')[-2]
        self.map_path = '/'.join(map_path.split('/')[:-1])
        self.max_dist = 1.2
        print(map_path)
        
        with open(f"{self.map_path}/map.yaml", "r") as stream:
            try:
                self.configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self._freq = 1/float(rospy.get_param('step_size'))

        super().__init__(obstacles_manager, robot_manager)
        self._num_robots = (
            count_robots(self.robot_manager) if type(self.robot_manager) is dict else 0
        )
        self.reset_flag = (
            {key: False for key in self.robot_manager.keys()}
            if type(self.robot_manager) is dict
            else {}
        )
        self.extended_setup = rospy.get_param("choose_goal", default = True)
        self.ns_prefix = f"/{ns}/" if ns else ""
        self._curr_stage = start_stage
        self._curr_stage = rospy.get_param("/curr_stage", default=6)
        self._stages = {}
        self.shelves = [[4+i*0.5,j+0.5] for j in range(3,8) for i in range(-1,2,2)]

        training_path  = rospkg.RosPack().get_path("training")
        self.curriculum_file = os.path.join(
            training_path, "configs", "training_curriculums", curriculum_file_path
        )
        self._read_stages_from_yaml()

        # check start stage format
        if not isinstance(start_stage, int):
            raise ValueError("Given start_stage not an Integer!")
        if self._curr_stage < 1 or self._curr_stage > len(self._stages):
            raise IndexError(
                "Start stage given for training curriculum out of bounds! Has to be between {1 to %d}!"
                % len(self._stages)
            )
        rospy.set_param("/curr_stage", self._curr_stage)

        self.open_tasks = []
        self.goal_reached_subscriber = rospy.Subscriber(f"{self.ns}/goals", robot_goal, self.subscriber_goal_status)
        self.goal_pos_publisher = rospy.Publisher(f"{self.ns}/open_tasks", robot_goal_list)
        self.stage_publisher = rospy.Publisher("/stage_info", Int8, queue_size=1)

        self._sub_next = rospy.Subscriber(
            f"{self.ns_prefix}next_stage", Bool, self.next_stage
        )
        self._sub_previous = rospy.Subscriber(
            f"{self.ns_prefix}previous_stage", Bool, self.previous_stage
        )

        self._initiate_stage()
        # path = '/'.join(self.map_path.split('/')[:-1]) + self._stages[6]["map"]
        # #self.reinit_map(f"{path}/map.yaml")
        # with open(f"{path}/map.yaml", "r") as stream:
        #     try:
        #         self.configs = yaml.safe_load(stream)
        #     except yaml.YAMLError as exc:
        #         print(exc)
        self.obs_goals = int(rospy.get_param("/observable_task_goals"))
        if "gridworld" in self.env:
            self.ctm = ctm(f'{self.map_path}/grid.npy', num_active_tasks=self.obs_goals)
        else:
            rospy.set_param("crossroad", True)

    def next_stage(self, *args, **kwargs):
        if self._curr_stage < len(self._stages) and self._curr_stage != 6:
            self._curr_stage = self._curr_stage + 1
            self._initiate_stage()

            if self.ns == "eval_sim":
                rospy.set_param("/curr_stage", self._curr_stage)
                # if not rospy.get_param("debug_mode"):
                #     with self._lock_json:
                #         self._update_curr_stage_json()

                if self._curr_stage == len(self._stages):
                    rospy.set_param("/last_stage_reached", True)
        else:
            print(
                f"({self.ns}) INFO: Tried to trigger next stage but already reached last one"
            )

    def previous_stage(self, *args, **kwargs):
        
        if self._curr_stage > 1:
            rospy.set_param("/last_stage_reached", False)
            if self._curr_stage < 8:
                self._curr_stage = self._curr_stage - 1
            #self.max_dist = min(self._curr_stage*3 , 8.0)
            self._initiate_stage()

            if self.ns == "eval_sim":
                rospy.set_param("/curr_stage", self._curr_stage)
                # with self._lock_json:
                #     self._update_curr_stage_json()
        else:
            print(
                f"({self.ns}) INFO: Tried to trigger previous stage but already reached first one"
            )

    def _initiate_stage(self):
        self.max_dist = self._curr_stage*2
        if 2 < self._curr_stage < 6:
            self.max_dist = 2.3 if self._curr_stage != 5 else 4
        if self.obstacles_manager is None:
            print("no obstacle manager")
            return
        self._remove_obstacles()
        self.extended_setup = False if self._curr_stage < 6 else True
        self.goal_publisher = []
        if len(self.goal_publisher) == 0:
            for _, robot_managers in self.robot_manager.items():
                for manager in robot_managers:
                    self.goal_publisher.append(rospy.Publisher(f"{self.ns_prefix}{manager.robot_id}/open_tasks", robot_goal_list, queue_size=1))
                    #print(f"{self.ns_prefix}{manager.robot_id}/open_tasks")
            print("initiated publishers")
        self.stage_publisher.publish(self._curr_stage)
        rospy.set_param("/curr_stage", self._curr_stage)


    def _read_stages_from_yaml(self):
        file_location = self.curriculum_file
        if os.path.isfile(file_location):
            with open(file_location, "r") as file:
                self._stages = yaml.load(file, Loader=yaml.FullLoader)
            assert isinstance(
                self._stages, dict
            ), "'training_curriculum.yaml' has wrong fromat! Has to encode dictionary!"
        else:
            raise FileNotFoundError(
                "Couldn't find 'training_curriculum.yaml' in %s "
                % self._PATHS.get("curriculum")
            )
    
    def _remove_obstacles(self):
        self.obstacles_manager.remove_obstacles()

    def subscriber_goal_status(self, msg: robot_goal):
        action = msg.crate_action.action
        if msg.robot_type == "refresh":
            print("refreshing goal publisher ", self.ns_prefix)
            if self._curr_stage < 7:
                self.publish_close_goals()
                return
            
            self.publisher_task_status(gen = False)
            return
        robots = self.robot_manager[msg.robot_type]
        robots: List[RobotManager] = robots
        for robot in robots:
            if robot.robot_id == msg.robot_id.split("/")[-1]:
                break
        else:
            print([r.robot_id for r in robots])
            raise ValueError(f'Robot: type: {msg.robot_type} id: "{msg.robot_id}" is not recognized.')

        if action == crate_action.PICKUP:
            if 3 < self._curr_stage < 6:
                new_goal = robot.get_random_pos(msg.robot_goal, 2.5)
                #new_goal = self._generate_goal(msg.robot_goal, robot)
                robot.publish_goal(new_goal.x, new_goal.y, 0)
                return
            try:
                ids, robot_goals, crate_goals = self.ctm.get_open_tasks(resolution= 1,generate= False)
                crate = self.ctm.pickup_crate(msg.robot_goal, robot) # crate at msg.goal is picked up by robot
                new_goal = crate.get_goal()
                new_goal.x += 0.5
                new_goal.y += 0.5

                robot.publish_goal(new_goal.x + self.configs['origin'][0], new_goal.y + self.configs['origin'][1], new_goal.theta)
                self.publisher_task_status(gen = (random.randint(0,4) == 0))
            except:
                print("something went wrong")
            
        elif action == crate_action.DROPOFF:
            robot.publish_goal(0,0,0)
            self.ctm.drop_crate(msg.crate_id)
            self.publisher_task_status()

        elif action == crate_action.BLOCK:
            robot.publish_goal(msg.robot_goal.x, msg.robot_goal.y, 0)
            if self._curr_stage > 6:
                self.ctm.block_goal(msg.crate_id)
                self.publisher_task_status(gen = False)

    def publisher_task_status(self, gen = True):
        ids, robot_goals, crate_goals = self.ctm.get_open_tasks(resolution= 1,generate= gen) # TODO check resolution parameter
        if len(robot_goals) == 0:
            return self.publisher_task_status(gen = True)
        self.open_tasks = []
        for id, r_goal, crate_goal in zip(ids, robot_goals, crate_goals):
            task = robot_goal(
                crate_action(crate_action.ASSIGN,"pub"),
                id, self.ns,
                '', '',
                r_goal,
                crate_goal
                )
            self.open_tasks.append(task)
        open_tasks_list = robot_goal_list(self.open_tasks)
        self.goal_pos_publisher.publish(open_tasks_list)
        
    def publish_close_goals(self, starts = None,dont_care = False):
        idx = 0
        if starts is None:
            starts = [None]*self._num_robots
            for i in range(self._num_robots):
                starts[i] = (random.randint(1,7),random.randint(1,10),0)
        
        if self.obs_goals > 1:
            for _, robot_managers in self.robot_manager.items():
                for manager in robot_managers:
                    valid = False
                    open_tasks = []
                    start = Pose2D(starts[idx][0], starts[idx][1], 0)
                    num_goals = random.randint(1,self.obs_goals)
                    for i in range(num_goals):
                        if random.randint(0, num_goals+1) > i+1:
                            r_goal = manager.get_random_pos(start, self.max_dist,dont_care=dont_care) if i == num_goals-1 and not valid else Pose2D()
                        else:
                            valid = True
                            if start.x == 3.5 or start.x == 4.5:
                                _, r_goal, _ = self._generate_goal(start, manager, starts, swap = 0)
                            else:
                                r_goal = manager.get_random_pos(start, self.max_dist,dont_care=dont_care)
                        crate_goal = manager.get_random_pos(start, 4*self.max_dist, dont_care=dont_care)
                        task = robot_goal(
                            crate_action(crate_action.ASSIGN,"pub"),
                            i, self.ns,
                            '', '',
                            r_goal,
                            crate_goal
                            )
                        open_tasks.append(task)
                    open_tasks_list = robot_goal_list(open_tasks)
                    try:
                        self.goal_publisher[idx].publish(open_tasks_list)
                    except:
                        print("no publishers")
                        self._initiate_stage()
                        self.goal_publisher[idx].publish(open_tasks_list)

                    idx += 1

    def add_robot_manager(self, robot_type: str, managers: List[RobotManager]):
        assert type(managers) is list
        if not self.robot_manager:
            self.robot_manager = {}
        self.robot_manager[robot_type] = managers
        self._num_robots = count_robots(self.robot_manager)
        self.reset_flag = {key: False for key in self.robot_manager.keys()}

    def reset(self, robot_type: str):
        self.stage_publisher.publish(self._curr_stage)
        # if self._curr_stage > 5:
        #     vertices = np.array([[3,3],[5,3],[3,1],[5,1]])
        #     self.obstacles_manager.register_static_obstacle_polygon(vertices)
        #     self.obstacles_manager.update_map()
        if "gridworld" in self.env:
            if self._curr_stage < 7: #or self._curr_stage == 7
                if self._curr_stage < 3:
                    seed = 1 if random.randint(self._curr_stage,2) == 2 else 0
                else:# self._curr_stage == 3:
                    seed = 3 if random.randint(1,3) != 2 else random.randint(1,3)%2+1
                # else:
                #     seed = 3 if random.randint(1,3) != 2 else random.randint(1,2)
                self.reset_single_target(robot_type, seed)
            else:
                self.reset_multi_target(robot_type)
        elif "crossroad" in self.env:
                self.reset_corssroad(robot_type)
        else:
            self.reset_single_target(robot_type, seed=0)

    def _generate_goal(self, start_pos, manager, starts = [None], goals = [None], swap = 0, max_dist = None):
        swaped = False
        forbidden = [None] * 2
        no_zone = [None] * 3
        max_dist = max_dist if max_dist else self.max_dist
        theta = random.uniform(-math.pi, math.pi)
        x = 6.5 if start_pos.x < 4 else 1.5
        forbidden[0] = (start_pos.x, start_pos.y, self._curr_stage*0.3)
        no_zone[0] = (x,5.5,6.5-self._curr_stage,14.7-2*self._curr_stage)
        if self._curr_stage > 4:
            #no_zone[0] = (x, start_pos.y, 3, 8-self._curr_stage)
            if random.randint(1,4) == 5:
                no_zone[1] = (abs(x-8),5.5,self._curr_stage-2.5,3+5*(self._curr_stage-4))
            else:
                no_zone[1] = (abs(x-8), start_pos.y, 3, 0.15*self._curr_stage)
        else:
            #no_zone[0] = (x,5.5,5-self._curr_stage/2,6)
            no_zone[1] = (abs(x-8), start_pos.y, 3, 0.15*self._curr_stage)
        if self._curr_stage == 6:
            no_zone = [None] * 3
        if random.randint(1,5) < swap:
            goal_pos = start_pos
            start_pos = manager.get_random_pos(start_pos, max_dist, forbidden_zones=starts+forbidden, no_zones=no_zone)
            swaped = True
        else:
            if self._curr_stage == 3:
                theta = random.uniform(-math.pi/4, math.pi/4) if start_pos.x > 4 else random.choice([-1,1])*(math.pi - random.uniform(0, math.pi/4))
            else:
                if random.randint(1,2)==2:
                    theta = random.uniform(-math.pi/4, math.pi/4) if start_pos.x > 4 else random.choice([-1,1])*(math.pi - random.uniform(0, math.pi/4))
                else:
                    theta = random.uniform(-math.pi/4, math.pi/4) if start_pos.x < 4 else random.choice([-1,1])*(math.pi - random.uniform(0, math.pi/4))
            goal_pos = manager.get_random_pos(start_pos, min(3.5,max_dist), forbidden_zones=forbidden+goals, no_zones=no_zone)
        start_pos.theta = theta
        return start_pos, goal_pos, swaped
    
    
    def generate_combinations(tuples):
        combinations = []
        for i in range(len(tuples)):
            for j in range(len(tuples)):
                if i != j:
                    combinations.append((tuples[i][0], tuples[j][0]))
                    combinations.append((tuples[i][0], tuples[j][1]))
                    combinations.append((tuples[i][1], tuples[j][1]))
                    combinations.append((tuples[i][1], tuples[j][0]))
        return combinations
    def reset_corssroad(self, robot_type: str):
        assert robot_type in self.robot_manager, f"Unknown robot type: {robot_type},"
        f" robot has to be one of the following types: {self.robot_manager.keys()}"
        c = self.configs["corridors"] 
        ver = c // 2 + 1
        hor = c//2 + c%2 + 1
        self.reset_flag[robot_type] = True
        swaped = True
        all_combinations = None
        if not all(self.reset_flag.values()):
            return
        
        pos = [None]*c
        done = False
        with self._map_lock:
            max_fail_times = 10
            fail_times = 0
            while fail_times < max_fail_times:
                try:
                    starts, goals = [None] * self._num_robots, [None] * self._num_robots
                    starts_h, goals_h = [None]*self._num_robots, [None]*self._num_robots
                    robot_idx = 0
                    for _, robot_managers in self.robot_manager.items():
                        for manager in robot_managers:
                            map_ = manager.map 
                            x, y = map_.info.origin.position.x, map_.info.origin.position.y
                            for i in range(ver-1):
                                pos[i] = ((int(x + (i+1)/ver*map_.info.width), y+100), 
                                (int(x + (i+1)/ver*map_.info.width), y + map_.info.height-100))
                            for i in range(hor-1):
                                pos[i+ver-1] = ((x + 100, int(y + (i+1)/hor*map_.info.height )),
                            (x + map_.info.width - 100, int(y +  (i+1)/hor*map_.info.height)))
                            break
                        if self._curr_stage > 3 and not all_combinations:
                            all_combinations = []
                            for i in range(len(pos)):
                                for j in range(len(pos)):
                                    if i != j:
                                        all_combinations.append((pos[i][0], pos[j][0]))
                                        all_combinations.append((pos[i][0], pos[j][1]))
                                        all_combinations.append((pos[i][1], pos[j][1]))
                                        all_combinations.append((pos[i][1], pos[j][0]))
                                    else:
                                        all_combinations.append(pos[i])
                                        all_combinations.append((pos[i][1], pos[i][0]))
                        for manager in robot_managers:
                            t = 0
                            if self._curr_stage < 4:
                                while t < 30:
                                    i = random.randint(0, c-1)
                                    start, end = pos[i]
                                    if start in starts_h:
                                        if end not in starts_h:
                                            h = start
                                            start = end
                                            end = h
                                            break
                                        t += 1
                                    else:
                                        break
                            else:
                                if not all_combinations:
                                    raise rospy.ServiceException
                                start, end = random.choice(all_combinations)
                                all_combinations.remove((start,end)) 
                                #print("new")
                                combinations = all_combinations.copy()
                                for combs in combinations:
                                    if start == combs[0] or end == combs[1]:
                                        # print("match")
                                        # print(start,end,combs)
                                        all_combinations.remove(combs) 
                            x_in_cells, y_in_cells = start
                            y_in_meters = y_in_cells * map_.info.resolution + map_.info.origin.position.y
                            x_in_meters = x_in_cells * map_.info.resolution + map_.info.origin.position.x
                            start_pos, goal_pos = Pose2D(), Pose2D()
                            start_pos.x, start_pos.y = x_in_meters, y_in_meters
                            if start_pos.x < 2:
                                start_pos.theta = 0
                                goal_pos.x, goal_pos.y = start_pos.x + map_.info.width*map_.info.resolution*0.2 + map_.info.origin.position.x, start_pos.y + random.choice([-0.5, 0.5])
                            elif start_pos.y < 2:
                                goal_pos.x, goal_pos.y = start_pos.x + random.choice([-0.5, 0.5]), start_pos.y + map_.info.width*map_.info.resolution*0.2 + map_.info.origin.position.y
                                start_pos.theta = math.pi/2
                            elif start_pos.y > 2 and start_pos.x < map_.info.width*map_.info.resolution + map_.info.origin.position.x - 2:
                                goal_pos.x, goal_pos.y = start_pos.x + random.choice([-0.5, 0.5]), start_pos.y - map_.info.width*map_.info.resolution*0.2 + map_.info.origin.position.y
                                start_pos.theta = -math.pi/2
                            else:
                                start_pos.theta = math.pi
                                goal_pos.x, goal_pos.y = start_pos.x - map_.info.width*map_.info.resolution*0.2 + map_.info.origin.position.x, start_pos.y + random.choice([-0.5, 0.5])
                            x_in_cells, y_in_cells = end
                            y_in_meters = y_in_cells * map_.info.resolution + map_.info.origin.position.y
                            x_in_meters = x_in_cells * map_.info.resolution + map_.info.origin.position.x
                            if done:
                                start_pos.theta = random.uniform(-math.pi, math.pi)
                            else:
                                goal_pos.x, goal_pos.y = x_in_meters, y_in_meters
                                if (starts_h[1] is not None and 1 < self._curr_stage < 5) or (starts_h[0] is not None and self._curr_stage == 1):
                                    done = True

                            manager.move_robot(start_pos)
                            if swaped:
                                manager.publish_goal(goal_pos.x, goal_pos.y, goal_pos.theta)
                            else:
                                manager.publish_goal(0,0,0)
                            
                            starts_h[robot_idx] = start
                            goals_h[robot_idx] = end
                            starts[robot_idx] = (
                                start_pos.x,
                                start_pos.y,
                                manager.ROBOT_RADIUS * 2.25,
                            )
                            goals[robot_idx] = (
                                goal_pos.x,
                                goal_pos.y,
                                manager.ROBOT_RADIUS * 4.0,
                            )
                            robot_idx += 1
                    self.obstacles_manager.reset_pos_obstacles_random(
                        forbidden_zones=starts + goals
                    )
                    break
                except rospy.ServiceException or IndexError as e:
                    rospy.logwarn(repr(e))
                    fail_times += 1
            if fail_times == max_fail_times:
                raise Exception("reset error!")
        self.publish_close_goals(starts,dont_care = True)

        self.reset_flag = dict.fromkeys(self.reset_flag, False)
        #self.publish_close_goals(starts)
    def reset_single_target(self, robot_type: str, seed = 0):
        self.simple = True
        choices = self.shelves.copy()
        choices1 = choices[2:-2]
        choices2 = choices[:2] + choices[-2:]
        assert robot_type in self.robot_manager, f"Unknown robot type: {robot_type},"
        f" robot has to be one of the following types: {self.robot_manager.keys()}"
        swaped = False
        swap = 3
        self.reset_flag[robot_type] = True
        if not all(self.reset_flag.values()):
            return
        if "gridworld" in self.env:
            self.ctm.generate_scenareo(nr_tasks=self._num_robots, type= 'random')
            _, _robot_goals, _crate_goals = self.ctm.get_open_tasks(resolution= 1)
        with self._map_lock:
            max_fail_times = 5
            fail_times = 0
            while fail_times < max_fail_times:
                try:
                    starts, goals = [None] * self._num_robots, [None] * self._num_robots
                    robot_idx = 0
                    for _, robot_managers in self.robot_manager.items():
                        for manager in robot_managers:
                            rospy.set_param("resolution", manager.map.info.resolution)
                            offx= manager.map.info.origin.position.x
                            offy = manager.map.info.origin.position.y
                            rospy.set_param("offx", offx)
                            rospy.set_param("offy", offy)
                            forbidden = [None] * 2
                            if self._curr_stage > 5 and seed != 0 and "gridworld" in self.env:
                                start_pos = Pose2D()
                                (start_pos.x, start_pos.y, start_pos.theta,) = get_random_pos_on_map(
                                    manager._free_space_indices,
                                    manager.map,
                                    manager.ROBOT_RADIUS * 2,
                                    forbidden_zones = starts #+ crate_goals,
                                )
                                goal_pos = _robot_goals[robot_idx]
                            elif seed == 3:
                                try:
                                    swaped = (swap != 6 and swaped)
                                    swap = 4 #if self._curr_stage < 5 else 5
                                    easy = (random.randint(1,4) == 2 and self._curr_stage !=3)
                                    start = random.choice(choices1) if easy else random.choice(choices2)
                                    if easy:
                                        choices1.remove(start)
                                    else:
                                        choices2.remove(start)
                                    start_pos = Pose2D()
                                    start_pos.x, start_pos.y = start[0], start[1]
                                    # if 3 < self._curr_stage < 6 and swaped and random.randint(1,2) == 2:
                                    #     start_pos.x, start_pos.y = goals[robot_idx-1][0], goals[robot_idx-1][1]
                                    #     swap = 6
                                    start_pos, goal_pos, swaped = self._generate_goal(start_pos, manager, starts, goals, swap)
                                    #manager.publish_goal(goal_pos.x, goal_pos.y, goal_pos.theta)
                                except:
                                    print("something went wrong")
                                    start_pos, goal_pos = manager.set_start_pos_goal_pos(
                                            forbidden_zones=starts+goals, max_dist = 1.5*self.max_dist
                                           )
                            elif seed == 1:
                                forbidden[0] = (4, 4, 4, 4.3)
                                start_pos = manager.get_random_pos(start_pos = None, max_dist=None, forbidden_zones=starts, no_zones=forbidden)
                                goal_pos = manager.get_random_pos(start_pos, max_dist=self._curr_stage*2, forbidden_zones=[(start_pos.x,start_pos.y,min(self._curr_stage*0.6,2.5))]+goals, no_zones = forbidden)
                                swaped = False
                            elif seed == 0:
                                start_pos = manager.get_random_pos(start_pos = None, max_dist=None, forbidden_zones=starts)
                                goal_pos = manager.get_random_pos(start_pos, max_dist=self._curr_stage*2, forbidden_zones=[(start_pos.x,start_pos.y,min(self._curr_stage-1, 5))]+goals)
                                # if "crossroad" not in self.env and "gridworld" not in self.env:
                                #     goal_pos = Pose2D()
                                #     c_map = Pose2D()
                                #     map_ = manager.map
                                #     c_map.x = map_.info.width/2* map_.info.resolution + map_.info.origin.position.y
                                #     c_map.y = map_.info.height/2* map_.info.resolution + map_.info.origin.position.y
                                #     goal_pos.x = c_map.x + c_map.x - start_pos.x
                                #     goal_pos.y = c_map.y + c_map.y - start_pos.y
                                
                            elif seed == 2:
                                forbidden[0] = (2.5, 5, 2.5, 6)
                                start_pos = manager.get_random_pos(start_pos = None, max_dist=None, forbidden_zones=starts, no_zones=forbidden)
                                goal_pos = manager.get_random_pos(start_pos, max_dist=self._curr_stage+1.5, forbidden_zones=[(start_pos.x,start_pos.y,min(self._curr_stage-1.5, 5))]+goals, no_zones = forbidden)
                                # start_pos, goal_pos = manager.set_start_pos_goal_pos(
                                #     forbidden_zones=starts+goals, max_dist = self.max_dist
                                #     )
                            try:
                                manager.move_robot(start_pos)
                            except:
                                print("reset error")
                                raise Exception("reset error!")
                            if swaped or self.simple:
                                manager.publish_goal(goal_pos.x, goal_pos.y, goal_pos.theta)
                            else:
                                manager.publish_goal(0,0,0)
                            
                            starts[robot_idx] = (
                                start_pos.x,
                                start_pos.y,
                                manager.ROBOT_RADIUS * 2.25,
                            )
                            goals[robot_idx] = (
                                goal_pos.x,
                                goal_pos.y,
                                manager.ROBOT_RADIUS * 4.0,
                            )
                            robot_idx += 1
                    self.obstacles_manager.reset_pos_obstacles_random(
                        forbidden_zones=starts + goals
                    )
                    break
                except rospy.ServiceException as e:
                    rospy.logwarn(repr(e))
                    fail_times += 1
            if fail_times == max_fail_times:
                raise Exception("reset error!")

        self.reset_flag = dict.fromkeys(self.reset_flag, False)
        self.publish_close_goals(starts)

    def reset_multi_target(self, robot_type: str):
        print(f'{robot_type=}')
        assert robot_type in self.robot_manager, f"Unknown robot type: {robot_type},"
        f" robot has to be one of the following types: {self.robot_manager.keys()}"

        self.reset_flag[robot_type] = True
        if not all(self.reset_flag.values()):
            return
        if self._curr_stage > 7:
            num = random.randint(self._num_robots-1,5)
        else:
            num = rospy.get_param("observable_task_goals", default = 5)
        self.ctm.generate_scenareo(nr_tasks=num, type= 'random')
        _, _robot_goals, _crate_goals = self.ctm.get_open_tasks(resolution= 1) # TODO check resolution parameter
        robot_goals_iter, crate_goals_iter = iter(_robot_goals), iter(_crate_goals)
        
        with self._map_lock:
            max_fail_times = 5
            fail_times = 0
            while fail_times < max_fail_times:
                starts, goals, crate_goals = [None] * self._num_robots, [None] * self._num_robots, [None] * self._num_robots
                goal_idx = 0
                rad = 0
                for robot_type, robot_managers in self.robot_manager.items():
                    for manager in robot_managers:
                        rad = manager.ROBOT_RADIUS if rad < manager.ROBOT_RADIUS else rad
                try:
                    robot_idx = 0
                    for robot_type, robot_managers in self.robot_manager.items():
                        for manager in robot_managers:
                            manager: RobotManager = manager
                            start = Pose2D()
                            (start.x, start.y, start.theta,) = get_random_pos_on_map(
                                manager._free_space_indices,
                                manager.map,
                                manager.ROBOT_RADIUS * 2,
                                forbidden_zones = starts #+ crate_goals,
                            )
                            manager.move_robot(start)
                            manager.publish_goal(0,0,0)
                            starts[robot_idx] = (
                                start.x,
                                start.y,
                                manager.ROBOT_RADIUS * 2.25,
                            )
                            robot_idx += 1
                    break
                except:
                    print("failed to reset: " +str(fail_times))
                    fail_times += 1
            if fail_times == max_fail_times:
                raise Exception("reset error!")
        if self.simple or self._curr_stage == 9 or not self.extended_setup:
            chosen_goals = self.find_best_matches(starts, _robot_goals)
            idx = 0
            for _, robot_managers in self.robot_manager.items():
                    for manager in robot_managers:
                        try:
                            chosen_goal = chosen_goals[idx]
                            manager.publish_goal(chosen_goal.x, chosen_goal.y, chosen_goal.theta)
                        except:
                            print("not enough goals")
                            start_pos, goal_pos = manager.set_start_pos_goal_pos(
                                forbidden_zones=starts, max_dist = self.max_dist
                            )
                            starts[idx] = (
                                start_pos.x,
                                start_pos.y,
                                manager.ROBOT_RADIUS * 2.25,
                            )
                        idx += 1
            self.reset_flag = dict.fromkeys(self.reset_flag, False)
            self.publish_close_goals(starts)
        else:
            self.reset_flag = dict.fromkeys(self.reset_flag, False)
            self.publisher_task_status()

    def distance(self, pose1, pose2):
        return math.sqrt((pose1.x - pose2.x)**2 + (pose1.y - pose2.y)**2)

    def find_best_matches(self, start_positions, goal_positions):
        matches = []
        goal_positions = [p for p in goal_positions if not (p.x == 0 and p.y == 0)]
        goal_positions_copy = goal_positions.copy() # make a copy of goal_positions to preserve original order
        for start_pose in start_positions:
            best_goal_pose = None
            min_distance = float('inf')
            for goal_pose in goal_positions_copy:
                d = self.distance(Pose2D(start_pose[0], start_pose[1], 0), goal_pose)
                if d < min_distance:
                    best_goal_pose = goal_pose
                    min_distance = d
            if best_goal_pose is not None:
                matches.append(best_goal_pose)
                goal_positions_copy.remove(best_goal_pose)
        return matches
    
    def find_best_matches2(self, start_positions, goal_positions):
        start_n = len(start_positions)
        goal_n = len(goal_positions)
        if goal_n == 0:
            return []
        goal_positions = [p for p in goal_positions if not (p.x == 0 and p.y == 0)] # remove goal positions with x=0 and y=0
        n = min(start_n, len(goal_positions))
        m = len(goal_positions)
        dist_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                start_pose = start_positions[i]
                dist_matrix[i, j] = self.distance(Pose2D(start_pose[0], start_pose[1], 0), goal_positions[j])
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        sorted_goal_positions = [None] * n
        for i in range(n):
            sorted_goal_positions[i] = goal_positions[col_ind[i]]
        return sorted_goal_positions
@unique
class ARENA_TASKS(Enum):
    MANUAL = "manual"
    RANDOM = "random"
    STAGED = "staged"
    SCENARIO = "scenario"
    CASES = "cases"


def get_mode(mode: str) -> ARENA_TASKS:
    return ARENA_TASKS(mode)


def get_MARL_task(
    ns: str,
    mode: str,
    robot_ids: List[str],
    PATHS: dict,
    start_stage: int = 1,
    cases_grid_map: np.ndarray = None
) -> ABSMARLTask:
    """Function to return desired navigation task manager.

    Args:
        ns (str): Environments' ROS namespace. There should only be one env per ns.
        mode (str): avigation task mode for the agents. Modes to chose from: ['random', 'staged']. \
            Defaults to "random".
        robot_ids (List[str]): List containing all robots' names in order to address the right namespaces.
        start_stage (int, optional): Starting difficulty level for the learning curriculum. Defaults to 1.
        PATHS (dict, optional): Dictionary containing program related paths. Defaults to None.

    Raises:
        NotImplementedError: The manual task mode is currently not implemented.
        NotImplementedError: The scenario task mode is currently not implemented.

    Returns:
        ABSMARLTask: A task manager instance.
    """
    assert type(robot_ids) is list

    task_mode = get_mode(mode)

    # get the map
    service_client_get_map = rospy.ServiceProxy("/static_map", GetMap)
    map_response = service_client_get_map()

    robot_manager = [
        RobotManager(
            ns=ns,
            map_=map_response.map,
            robot_type="jackal",
            robot_id=robot_ns,
        )
        for robot_ns in robot_ids
    ]

    obstacles_manager = ObstaclesManager(ns, map_response.map)

    task = None
    if task_mode == ARENA_TASKS.MANUAL:
        raise NotImplementedError
    if task_mode == ARENA_TASKS.RANDOM:
        rospy.set_param("/task_mode", "random")
        obstacles_manager.register_random_obstacles(10, 1.0)
        task = RandomMARLTask(obstacles_manager, robot_manager)
    if task_mode == ARENA_TASKS.STAGED:
        rospy.set_param("/task_mode", "staged")
        task = StagedMARLRandomTask(
            ns, obstacles_manager, robot_manager, start_stage, PATHS
        )
    if task_mode == ARENA_TASKS.SCENARIO:
        raise NotImplementedError
    if task_mode == ARENA_TASKS.CASES:
        rospy.set_param("/task_mode", "cases")
        task = CasesMARLTask(None, None, ns)
    return task


def get_task_manager(
    ns: str,
    mode: str,
    curriculum_path: dict,
    start_stage: int = 1,
    cases_grid_map: np.ndarray = None,
) -> ABSMARLTask:
    """Function to return desired navigation task manager.

    Args:
        ns (str): Environments' ROS namespace. There should only be one env per ns.
        mode (str): avigation task mode for the agents. Modes to chose from: ['random', 'staged', 'cases']. \
            Defaults to "random".
        robot_ids (List[str]): List containing all robots' names in order to address the right namespaces.
        start_stage (int, optional): Starting difficulty level for the learning curriculum. Defaults to 1.
        PATHS (dict, optional): Dictionary containing program related paths. Defaults to None.

    Raises:
        NotImplementedError: The manual task mode is currently not implemented.
        NotImplementedError: The scenario task mode is currently not implemented.

    Returns:
        ABSMARLTask: A task manager instance.
    """
    task_mode = get_mode(mode)
    task = None
    if task_mode == ARENA_TASKS.CASES:
        rospy.set_param("/task_mode", "cases")
        task = CasesMARLTask(None, None, ns, curriculum_file_path=curriculum_path )
    if task_mode == ARENA_TASKS.MANUAL:
        raise NotImplementedError
    if task_mode == ARENA_TASKS.RANDOM:
        rospy.set_param("/task_mode", "random")
        # obstacles_manager.register_random_obstacles(10, 1.0)
        task = RandomMARLTask(None, None)
    if task_mode == ARENA_TASKS.STAGED:
        rospy.set_param("/task_mode", "staged")
        task = StagedMARLRandomTask(ns, None, None, start_stage, curriculum_path)
    if task_mode == ARENA_TASKS.SCENARIO:
        raise NotImplementedError
    
    return task


def init_obstacle_manager(n_envs, mode: str = "train"):
    service_client_get_map = rospy.ServiceProxy("/static_map", GetMap)
    map_response = service_client_get_map()
    if mode == "train":
        return {
            f"sim_{i}": ObstaclesManager(f"sim_{i}", map_response.map)
            for i in range(1, n_envs + 1)
        }
    elif mode == "eval":
        return ObstaclesManager("eval_sim", map_response.map)
    else:
        raise ValueError("mode must be either 'train' or 'eval'")


