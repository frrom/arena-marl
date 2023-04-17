#%%
from time import sleep
from typing import Dict, Type, Union, List

import json
import os
import rospkg
import rospy
import yaml
import warnings

from abc import ABC, abstractmethod
from enum import Enum, unique
from threading import Lock
from filelock import FileLock

from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap, SetMap
from std_msgs.msg import Bool, Int8
from geometry_msgs.msg import Pose2D
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

from case_task_generator.scripts.task_gen1 import CaseTaskManager as ctm
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
        self._curr_stage = 4
        self._stages = {}

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
        path = '/'.join(self.map_path.split('/')[:-1]) + self._stages[6]["map"]
        #self.reinit_map(f"{path}/map.yaml")
        with open(f"{path}/map.yaml", "r") as stream:
            try:
                self.configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.obs_goals = int(rospy.get_param("/observable_task_goals"))
        self.ctm = ctm(f'{path}/grid.npy', num_active_tasks=self.obs_goals)

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
            self.max_dist = min(self._curr_stage*3 , 8.0)
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
        stage_idx = 0
        
        if self._curr_stage < 7:
            self.max_dist = min(self._curr_stage*2, 8.0)
            if self._curr_stage == 4:
                self.max_dist = 4.5
            if self.obstacles_manager is None:
                print("no obstacle manager")
                return
            self._remove_obstacles()
            self.extended_setup = False
            self.goal_publisher = []
            if len(self.goal_publisher) == 0:
                for _, robot_managers in self.robot_manager.items():
                    for manager in robot_managers:
                        self.goal_publisher.append(rospy.Publisher(f"{self.ns_prefix}{manager.robot_id}open_tasks", robot_goal_list))
                print("initiated publishers")
            # stage == 3 heuristic goal distribution

            if self._curr_stage > 5:
                stage_idx = 1 #choose a closely generated goal

        else:
            if self.obstacles_manager is None:
                return
            if rospy.get_param("choose_goal", default = True):
                self.extended_setup = True
            path = '/'.join(self.map_path.split('/')[:-1]) + self._stages[self._curr_stage]["map"]
            #self.reinit_map(f"{path}/map.yaml")
            with open(f"{path}/map.yaml", "r") as stream:
                try:
                    self.configs = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            if self.obstacles_manager is not None:
                self._remove_obstacles()
            
            stage_idx = 2
        self.stage_publisher.publish(stage_idx)
        rospy.set_param("/communication", stage_idx)
        rospy.set_param("/task_distribution", stage_idx)


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
        robots = self.robot_manager[msg.robot_type]
        robots: List[RobotManager] = robots
        for robot in robots:
            if robot.robot_id == msg.robot_id.split("/")[-1]:
                break
        else:
            print([r.robot_id for r in robots])
            raise ValueError(f'Robot: type: {msg.robot_type} id: "{msg.robot_id}" is not recognized.')

        if action == crate_action.PICKUP:
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
            self.ctm.drop_crate(msg.crate_id)
            self.publisher_task_status()

        elif action == crate_action.BLOCK:
            self.ctm.block_goal(msg.crate_id)
            self.publisher_task_status(gen = False)

    def publisher_task_status(self, gen = True):
        ids, robot_goals, crate_goals = self.ctm.get_open_tasks(resolution= 1,generate= gen) # TODO check resolution parameter
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
        
    def publish_close_goals(self, starts):
        idx = 0
        if self.obs_goals > 1:
            for _, robot_managers in self.robot_manager.items():
                for manager in robot_managers:
                    open_tasks = []
                    start = Pose2D(starts[idx][0], starts[idx][1], 0)
                    for i in range(random.randint(2,self.obs_goals)):
                        r_goal = manager.get_random_pos(start, self.max_dist)
                        crate_goal = manager.get_random_pos(start, 4*self.max_dist)
                        task = robot_goal(
                            crate_action(crate_action.ASSIGN,"pub"),
                            i, self.ns,
                            '', '',
                            r_goal,
                            crate_goal
                            )
                        open_tasks.append(task)
                    open_tasks_list = robot_goal_list(self.open_tasks)
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
        if self._curr_stage < 5 or self._curr_stage == 6:
            self.reset_single_target(robot_type)
        else:
            self.reset_multi_target(robot_type)

    def reset_single_target(self, robot_type: str):
        assert robot_type in self.robot_manager, f"Unknown robot type: {robot_type},"
        f" robot has to be one of the following types: {self.robot_manager.keys()}"

        self.reset_flag[robot_type] = True
        if not all(self.reset_flag.values()):
            return

        with self._map_lock:
            max_fail_times = 5
            fail_times = 0
            if self._curr_stage == 4:
                self.ctm.generate_scenareo(nr_tasks=self._num_robots, type= 'random')
                _, _robot_goals, _crate_goals = self.ctm.get_open_tasks(resolution= 1)
            while fail_times < max_fail_times:
                try:
                    starts, goals = [None] * self._num_robots, [None] * self._num_robots
                    robot_idx = 0
                    for _, robot_managers in self.robot_manager.items():
                        for manager in robot_managers:
                            if self._curr_stage == 4:
                                try:
                                    start_pos = _robot_goals[robot_idx]
                                    #start_pos.theta = random.uniform(-math.pi, math.pi)
                                except:
                                    start_pos = manager.get_random_pos(start_pos, self.max_dist, forbidden_zones=starts)
                                goal_pos = manager.get_random_pos(start_pos, self.max_dist, forbidden_zones=starts+goals)
                                manager.move_robot(start_pos)
                                manager.publish_goal(goal_pos.x, goal_pos.y, goal_pos.theta)
                            else:
                                start_pos, goal_pos = manager.set_start_pos_goal_pos(
                                    forbidden_zones=starts+goals, max_dist = self.max_dist
                                )
                            #print(goal_pos)
                            starts[robot_idx] = (
                                start_pos.x,
                                start_pos.y,
                                manager.ROBOT_RADIUS * 2.25,
                            )
                            goals[robot_idx] = (
                                goal_pos.x,
                                goal_pos.y,
                                manager.ROBOT_RADIUS * 3.25,
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

        self.ctm.generate_scenareo(nr_tasks=self._num_robots, type= 'random')
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
                for goal in _robot_goals:
                    goals[goal_idx] = (
                            goal.x,
                            goal.y,
                            rad * 3.25,
                        )
                    goal_idx += 1
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
                                forbidden_zones= goals #+ crate_goals,
                            )
                            manager.move_robot(start)
                            starts[robot_idx] = (
                                start.x,
                                start.y,
                                manager.ROBOT_RADIUS * 2.25,
                            )
                            robot_idx += 1
                    self.obstacles_manager.reset_pos_obstacles_random(
                        forbidden_zones=starts + goals# + crate_goals 
                    )
                    break
                except:
                    print("failed to reset: " +str(fail_times))
                    fail_times += 1
            if fail_times == max_fail_times:
                raise Exception("reset error!")
        if self._curr_stage == 5 or not self.extended_setup:
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


