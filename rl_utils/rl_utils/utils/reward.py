from typing import Tuple

import numpy as np
import scipy.spatial
from geometry_msgs.msg import Pose2D
from numpy.lib.utils import safe_eval
import math
import rospy
from std_msgs.msg import Int8


def polar_displacement(pos1, pos2, delta_t):
    delta_r = math.sqrt(pos1[0]**2 + pos2[0]**2 - 2*pos1[0]*pos2[0]*math.cos(pos2[1] - pos1[1]))
    delta_theta = math.atan2(pos2[0]*math.sin(pos2[1] - pos1[1]), pos1[0] - pos2[0]*math.cos(pos2[1] - pos1[1]))
    if abs(delta_theta) < np.pi/2:
        delta_r = -delta_r
    v_r = delta_r/delta_t

    v_theta = (pos2[1] - pos1[1])/delta_t
    return v_r, v_theta

class RewardCalculator:
    def __init__(
        self,
        holonomic: bool,
        robot_radius: float,
        safe_dist: float,
        goal_radius: float,
        rule: str = "rule_00",
        extended_eval: bool = False,
        max_steps = 400,
    ):
        """
        A class for calculating reward based various rules.


        :param safe_dist (float): The minimum distance to obstacles or wall that robot is in safe status.
                                  if the robot get too close to them it will be punished. Unit[ m ]
        :param goal_radius (float): The minimum distance to goal that goal position is considered to be reached.
        """
        self.curr_reward = 0
        # additional info will be stored here and be returned alonge with reward.
        self.info = {}
        self.holonomic = holonomic
        self.robot_radius = 1.1*robot_radius
        self.goal_radius = goal_radius
        self.last_goal_dist = None
        self.last_dist_to_path = None
        self.last_action = None
        self._curr_dist_to_path = None
        self.safe_dist = safe_dist
        self._extended_eval = extended_eval
        self.crate = 0
        self.steps = 0
        self.max_steps = max_steps
        self.min_dist = 1000
        self.curr_vel = 0
        self.curr_avel = 0
        self.counter = 0
        self.last_goal_in_robot_frame = None
        self.extended_setup = rospy.get_param("choose_goal", default = True)
        self.crossroad = rospy.get_param("crossroad", default=False)
        self._stage_info = rospy.Subscriber("stage_info", Int8, self.callback_stage_info)
        self.stage_info = rospy.get_param("/curr_stage", default = 1)
        self.subreward = 0
        self.counter2 = 0
        self.factor = 1
        self.angle = 0
        self.intermediate = 0

        self.kdtree = None

        self._cal_funcs = {
            "rule_00": RewardCalculator._cal_reward_rule_00,
            "rule_01": RewardCalculator._cal_reward_rule_01,
            "rule_02": RewardCalculator._cal_reward_rule_02,
            "rule_03": RewardCalculator._cal_reward_rule_03,
            "rule_04": RewardCalculator._cal_reward_rule_04,
            "rule_05": RewardCalculator._cal_reward_rule_05,
            "rule_06": RewardCalculator._cal_reward_rule_06,
            "rule_07": RewardCalculator._cal_reward_rule_07,
            "barn": RewardCalculator._cal_reward_rule_barn,
        }
        self.cal_func = self._cal_funcs[rule]

    def reset(self):
        """
        reset variables related to the episode
        """
        self.last_goal_dist = None
        self.last_dist_to_path = None
        self.last_action = None
        self.kdtree = None
        self._curr_dist_to_path = None
        self.crate = 0
        self.steps = 0
        self.min_dist = 1000
        self.curr_vel = 0
        self.counter = 0
        self.curr_avel= 0
        self.last_goal_in_robot_frame = None
        self.subreward = 0
        self.factor = 1
        self.counter2 = 0
        self.angle = 0
        self.intermediate = 0

    def _reset(self):
        """
        reset variables related to current step
        """
        self.curr_reward = 0
        self.info = {}
        self.info["is_success"] = 0
        self.info["crate"] = self.crate
        self.info["crash"] = False

    def get_reward(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        goals_in_robot_frame: np.ndarray = np.array([]),
        package: bool = False,
        vel = [0,0],
        curr_goal = Pose2D(),
        ids: np.ndarray = np.array([]),
        *args,
        **kwargs
    ):
        """
        Returns reward and info to the gym environment.

        :param laser_scan (np.ndarray): laser scan data
        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        """
        self._reset()
        if self.extended_setup:
            self.cal_func(self, laser_scan, goal_in_robot_frame, goals_in_robot_frame=goals_in_robot_frame, vel=vel, goal= curr_goal, *args, **kwargs)
        else:
            self.cal_func(self, laser_scan, goal_in_robot_frame, *args, **kwargs)
        return self.curr_reward, self.info

    def _cal_reward_rule_00(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._reward_goal_reached(goal_in_robot_frame)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4
        )

    def _cal_reward_rule_01(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._reward_distance_traveled(kwargs["action"], consumption_factor=0.0075)
        self._reward_goal_reached(goal_in_robot_frame, reward=15)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan, punishment=10)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4
        )

    def _cal_reward_rule_02(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._set_current_dist_to_globalplan(
            kwargs["global_plan"], kwargs["robot_pose"]
        )
        self._reward_distance_traveled(kwargs["action"], consumption_factor=0.0075)
        self._reward_following_global_plan(reward_factor=0.2, penalty_factor=0.3)
        self._reward_goal_reached(goal_in_robot_frame, reward=15)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan, punishment=10)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4
        )

    def _cal_reward_rule_03(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._set_current_dist_to_globalplan(
            kwargs["global_plan"], kwargs["robot_pose"]
        )
        self._reward_following_global_plan(kwargs["action"])
        if laser_scan.min() > self.safe_dist:
            self._reward_distance_global_plan(
                reward_factor=0.2,
                penalty_factor=0.3,
            )
        else:
            self.last_dist_to_path = None
        self._reward_goal_reached(goal_in_robot_frame, reward=15)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan, punishment=10)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4
        )

    def _cal_reward_rule_04(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._set_current_dist_to_globalplan(
            kwargs["global_plan"], kwargs["robot_pose"]
        )
        self._reward_following_global_plan(kwargs["action"])
        if laser_scan.min() > self.safe_dist + 0.35:
            self._reward_distance_global_plan(
                reward_factor=0.2,
                penalty_factor=0.3,
            )
            self._reward_abrupt_direction_change(kwargs["action"])
            self._reward_reverse_drive(kwargs["action"])
        else:
            self.last_dist_to_path = None
        self._reward_goal_reached(goal_in_robot_frame, reward=25)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan, punishment=10)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4
        )

    def _cal_reward_rule_05(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._curr_action = kwargs["action"]
        self._set_current_dist_to_globalplan(
            kwargs["global_plan"], kwargs["robot_pose"]
        )
        # self._reward_following_global_plan(self._curr_action)
        if laser_scan.min() > self.safe_dist:
            self._reward_distance_global_plan(
                reward_factor=0.2,
                penalty_factor=0.3,
            )
            self._reward_abrupt_vel_change(vel_idx=0, factor=0.2)
            self._reward_abrupt_vel_change(vel_idx=-1, factor=0.05)
            if self.holonomic:
                self._reward_abrupt_vel_change(vel_idx=1, factor=0.05)
            self._reward_reverse_drive(self._curr_action, 0.0001)
        else:
            self.last_dist_to_path = None
        self._reward_goal_reached(goal_in_robot_frame, reward=17.5)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan, punishment=10)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.4, penalty_factor=0.6
        )
        self.last_action = self._curr_action

    def _cal_reward_rule_06(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._curr_action = kwargs["action"]
        # self._reward_following_global_plan(self._curr_action)
        if laser_scan.min() > self.safe_dist:
            self._reward_abrupt_vel_change(vel_idx=0, factor=0.05)
            self._reward_abrupt_vel_change(vel_idx=-1, factor=0.01)
            if self.holonomic:
                self._reward_abrupt_vel_change(vel_idx=1, factor=0.01)
            self._reward_reverse_drive(self._curr_action, 0.00005)
        else:
            self.last_dist_to_path = None
        self._reward_goal_reached(goal_in_robot_frame, reward=17.5)
        self._reward_safe_dist(laser_scan, punishment=0.2)
        self._reward_collision(laser_scan, punishment=10)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.5, penalty_factor=0.6
        )
        self.last_action = self._curr_action
    def _cal_reward_rule_07(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        goals_in_robot_frame: np.ndarray = np.array([]),
        ids: np.ndarray = np.array([]),
        package: bool = False,
        vel= [0,0],
        goal=Pose2D(),
        *args,
        **kwargs
    ):
        self._curr_action = kwargs["action"]
        self.curr_pos = kwargs["robot_pose"]
        # self._set_current_dist_to_globalplan(
        #     kwargs["global_plan"], kwargs["robot_pose"]
        # )

        self.info["is_done"] = False
        self.curr_vel, self.curr_avel = vel[0], vel[1]
        mode = 1 if self.stage_info > 1 else 0
        if laser_scan.min() > 1.3*self.safe_dist:
            self._reward_abrupt_vel_change(vel_idx=0, factor=0.1)
            self._reward_abrupt_vel_change(vel_idx=1, factor=0.03)
            # if self.holonomic:
            #     self._reward_abrupt_vel_change(vel_idx=1, factor=0.03)
            self._reward_reverse_drive()
        if goal_in_robot_frame[0] != 0:
            self._reward_goal_approached(goal_in_robot_frame, goal, reward_factor=0.4, penalty_factor=0.5, mode=mode)
            self._reward_not_moving(punishment=0.008)
            self._reward_goal_reached(goal_in_robot_frame, reward=40.0)
        else:
            self.curr_reward -= 0.04

        #self._reward_max_velocity(max_vel = 0.6)
        self._reward_safe_dist(laser_scan, punishment=0.04)
        if self.stage_info > 1:
            self._reward_high_conc_force()
            self._reward_steering()
        self._reward_distance_traveled(mode = 1, penalty = 0.005)
        self._reward_collision(laser_scan, punishment=10.0)
    
        self.last_goal_in_robot_frame = goal_in_robot_frame
        self.last_action = self._curr_action
    def _cal_reward_rule_barn(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._curr_action = kwargs["action"]
        self._set_current_dist_to_globalplan(
            kwargs["global_plan"], kwargs["robot_pose"]
        )
        # self._reward_following_global_plan(self._curr_action)
        # if laser_scan.min() > self.safe_dist:
        # self._reward_distance_global_plan(
        #     reward_factor=0.2,
        #     penalty_factor=0.3,
        # )
        # else:
        #     self.last_dist_to_path = None
        self._reward_abrupt_vel_change(vel_idx=0, factor=1.1)
        self._reward_abrupt_vel_change(vel_idx=-1, factor=0.55)
        if self.holonomic:
            self._reward_abrupt_vel_change(vel_idx=1, factor=0.55)
        self._reward_reverse_drive(self._curr_action, 0.0001)
        self._reward_goal_reached(goal_in_robot_frame, reward=15)
        self._reward_safe_dist(laser_scan, punishment=0.005)
        self._reward_collision(laser_scan, punishment=15)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.5, penalty_factor=0.7
        )
        self.last_action = self._curr_action

    def _set_current_dist_to_globalplan(
        self, global_plan: np.ndarray, robot_pose: Pose2D
    ):
        if global_plan is not None and len(global_plan) != 0:
            self._curr_dist_to_path, idx = self.get_min_dist2global_kdtree(
                global_plan, robot_pose
            )
    def _reward_max_velocity(self, max_vel, penalty_factor=0.5):
        if self.curr_vel > max_vel:
            self.curr_reward -= penalty_factor*(self.curr_vel-max_vel)**2

    def _reward_goal_reached(
        self, goal_in_robot_frame=Tuple[float, float], reward: float = 15
    ):
        """
        Reward for reaching the goal.

        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        :param reward (float, optional): reward amount for reaching. defaults to 15
        """
        if goal_in_robot_frame[0] < self.goal_radius:
            self.curr_reward = reward
            self.info["is_done"] = True
            self.info["done_reason"] = 2
            self.info["is_success"] = 1
            self.steps = 0
            self.subreward = 0
        
    def _reward_distance_to_goals(self,
        goals_in_robot_frame: np.ndarray,
        reward_factor: float = 0.6,
        distance_factor: float = 0.3,):
        
        for dist in goals_in_robot_frame[:,0]:
            self.curr_reward += reward_factor*np.exp(-(dist-self.goal_radius)*distance_factor)/self.steps
        self.steps += 0.12
        #print(self.curr_reward)
    def _reward_goal_approached(
        self,
        goal_in_robot_frame=Tuple[float, float],
        goal = Pose2D(),
        reward_factor: float = 0.3,
        penalty_factor: float = 0.5,
        mode = 0
    ):
        """
        Reward for approaching the goal.

        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        :param reward_factor (float, optional): positive factor for approaching goal. defaults to 0.3
        :param penalty_factor (float, optional): negative factor for withdrawing from goal. defaults to 0.5
        """
        if self.last_goal_dist is not None:
            # goal_in_robot_frame : [rho, theta]

            # higher negative weight when moving away from goal
            # (to avoid driving unnecessary circles when train in contin. action space)
            step_reward = 0.5
            if self.crossroad:
                self.subreward = 2
            if mode == 1:
                x = goal.x
                if not self.crossroad and 3-self.robot_radius < self.curr_pos.x < 5+self.robot_radius and 3 < self.curr_pos.y < 8 and goal_in_robot_frame[0] > 0.5 + self.robot_radius:
                    self.curr_reward -= 0.01
                if (self.subreward == 2 or ((x > 4 and self.curr_pos.x >= 4.5) or (x < 4 and self.curr_pos.x <= 3.5) or (goal.y > 8 and self.curr_pos.y > 8) or (goal.y < 3 and self.curr_pos.y < 3))) and goal.x != 0:
                    if self.subreward < 2:
                        self.min_dist = goal_in_robot_frame[0]
                        self.intermediate = goal_in_robot_frame[0] if self.intermediate == 0 else min(abs(8-goal.y),abs(goal.y-3))+0.75
                        self.subreward = 2
                        #self.curr_reward += 0.02
                    if self.min_dist > goal_in_robot_frame[0]:
                        if (self.intermediate - goal_in_robot_frame[0]) > step_reward:
                            self.curr_reward += (self.intermediate - goal_in_robot_frame[0])//step_reward
                            self.intermediate = self.intermediate - (self.intermediate - goal_in_robot_frame[0])//step_reward*step_reward
                        self.min_dist = goal_in_robot_frame[0]
                    # else:
                    #     self.curr_reward -= 0.02
                    if (self.last_goal_dist - goal_in_robot_frame[0]) > 0:
                        w = reward_factor
                    else:
                        w = penalty_factor
                        self.curr_reward -= 0.02
                    self.curr_reward += w * (self.last_goal_dist - goal_in_robot_frame[0])
                    # if (self.last_goal_dist - goal_in_robot_frame[0]) <= 0:
                    #     self.curr_reward -= 0.02
                    self.last_goal_dist = goal_in_robot_frame[0]
                elif self.subreward < 2:
                    if self.subreward == 0:
                        dist = min(abs(8 - self.curr_pos.y), abs(self.curr_pos.y - 3))
                        if self.min_dist > dist:
                            if self.min_dist > 100:
                                self.intermediate = dist
                                self.last_goal_dist = dist
                            self.min_dist = dist
                            
                            if (self.intermediate - dist) > step_reward:
                                self.curr_reward += (self.intermediate - dist)//step_reward
                                self.intermediate = self.intermediate - (self.intermediate - dist)//step_reward*step_reward
                        # else:
                        #     self.curr_reward -= 0.03
                        if self.curr_pos.y > 8.0 or self.curr_pos.y < 3.0:
                            self.subreward = 1
                            self.min_dist = 1000
                    if self.subreward == 1:
                        dist = abs(x - self.curr_pos.x)
                        if self.min_dist > dist:
                            if self.min_dist > 100:
                                self.intermediate = 1.5
                                self.last_goal_dist = dist
                            self.min_dist = dist
                            if (self.intermediate - dist) > step_reward:
                                self.curr_reward += (self.intermediate - dist)//step_reward
                                self.intermediate = self.intermediate - (self.intermediate - dist)//step_reward*step_reward
                        # else:
                        #     self.curr_reward -= 0.03
                    # if (self.last_goal_dist - dist) <= 0:
                    #     self.curr_reward -= 0.02
                    if (self.last_goal_dist - dist) > 0:
                        w = reward_factor
                    else:
                        w = penalty_factor
                        self.curr_reward -= 0.02
                    self.curr_reward += w * (self.last_goal_dist - dist)
                    self.last_goal_dist = dist
                else:
                    self.curr_reward -= 0.03
            else:
                if (self.last_goal_dist - goal_in_robot_frame[0]) > 0:
                    w = reward_factor
                else:
                    w = penalty_factor
                reward = w * (self.last_goal_dist - goal_in_robot_frame[0])
                self.curr_reward += reward
                self.last_goal_dist = goal_in_robot_frame[0]
        else:
            self.last_goal_dist = goal_in_robot_frame[0]
                

    def _reward_collision(self, laser_scan: np.ndarray, punishment: float = 10):
        """
        Reward for colliding with an obstacle.

        :param laser_scan (np.ndarray): laser scan data
        :param punishment (float, optional): punishment for collision. defaults to 10
        """
        if laser_scan.min() <= self.robot_radius:
            # punish = 0
            # for i in range(int(self.steps/0.1), self.max_steps):
            #     punish += min(i*0.1/100, 0.1)
            # self.curr_reward -= punish
            self.info["crash"] = True
            if self.stage_info > 6:
                self.curr_reward -= punishment
            
                if not self._extended_eval:
                    self.info["is_done"] = True
                    self.info["done_reason"] = 1
                    self.info["is_success"] = 0
            else:
                self.curr_reward -= (0.5 + 0.3*self.stage_info)*max(0.7,4*self._curr_action[0]**2)
                #self.curr_reward -= 0.6*np.exp(0.5*(self.stage_info-3))*max(0.7,2*abs(self.curr_vel))

    def _reward_safe_dist(self, laser_scan: np.ndarray, punishment: float = 0.15):
        """
        Reward for undercutting safe distance.

        :param laser_scan (np.ndarray): laser scan data
        :param punishment (float, optional): punishment for undercutting. defaults to 0.15
        """
        if laser_scan.min() < 1.15*self.safe_dist:
            self.curr_reward -= self.stage_info*punishment*np.exp(-laser_scan.min()+self.robot_radius)
            #print(2*punishment*np.exp(2*(-laser_scan.min()+self.robot_radius)))
            if self._extended_eval:
                self.info["safe_dist"] = True
        
    def _reward_not_moving(self, action: np.ndarray = None, punishment: float = 0.02):
        """
        Reward for not moving. Only applies half of the punishment amount
        when angular velocity is larger than zero.

        :param action (np.ndarray (,2)): [0] - linear velocity, [1] - angular velocity
        :param punishment (float, optional): punishment for not moving. defaults to 0.01
        """
        if action is not None and action[0] == 0.0:
            self.curr_reward -= punishment if action[1] == 0.0 else punishment / 2
        else:
            if abs(self.curr_vel) <= 0.01:
                self.curr_reward -= punishment
                self.counter += 1
            else:
                self.counter = 0
            # if abs(self.curr_vel - self._curr_action[0]) > 0.1:
            #     self.counter2 = min(14, self.counter2 + 1) 
            #     if self.counter2 > 10:
            #         self.curr_reward -= 0.1*np.exp(0.5*(self.stage_info-3))*(self.counter2-10)
            # else:
            #     self.counter2 = max(0,self.counter2 - 1)
            if self.counter > 20:
                if self.stage_info == 8:
                    self.info["is_done"] = True
                    self.info["done_reason"] = 1
                    self.info["is_success"] = 0
                    self.curr_reward -= 10.0
                else:
                    self.curr_reward -= 0.2

    def _reward_distance_traveled(
        self,
        action: np.array = None,
        punishment: float = 0.01,
        consumption_factor: float = 0.005,
        mode = 0,
        penalty = 1
    ):
        """
        Reward for driving a certain distance. Supposed to represent "fuel consumption".

        :param action (np.ndarray (,2)): [0] - linear velocity, [1] - angular velocity
        :param punishment (float, optional): punishment when action can't be retrieved. defaults to 0.01
        :param consumption_factor (float, optional): weighted velocity punishment. defaults to 0.01
        """
        if mode == 1:
            self.curr_reward -= 0.005
            # print(self.max_steps*0.1, self.steps)
            # if (self.max_steps-2)*0.1 <= self.steps:
            #     print("failed to find goal")
            #     self.curr_reward -= 10.0
        else:
            if action is None:
                self.curr_reward -= punishment
            else:
                lin_vel = action[0]
                ang_vel = action[1]
                reward = (lin_vel + (ang_vel * 0.001)) * consumption_factor
        
            self.curr_reward -= reward

    def _reward_distance_global_plan(
        self,
        reward_factor: float = 0.1,
        penalty_factor: float = 0.15,
    ):
        """
        Reward for approaching/veering away the global plan. (Weighted difference between
        prior distance to global plan and current distance to global plan)

        :param global_plan: (np.ndarray): vector containing poses on global plan
        :param robot_pose (Pose2D): robot position
        :param reward_factor (float, optional): positive factor when approaching global plan. defaults to 0.1
        :param penalty_factor (float, optional): negative factor when veering away from global plan. defaults to 0.15
        """
        if self._curr_dist_to_path:
            if self.last_dist_to_path is not None:
                if self._curr_dist_to_path < self.last_dist_to_path:
                    w = reward_factor
                else:
                    w = penalty_factor

                self.curr_reward += w * (
                    self.last_dist_to_path - self._curr_dist_to_path
                )
            self.last_dist_to_path = self._curr_dist_to_path

    def _reward_following_global_plan(
        self,
        action: np.array = None,
        dist_to_path: float = 0.5,
    ):
        """
        Reward for travelling on the global plan.

        :param global_plan: (np.ndarray): vector containing poses on global plan
        :param robot_pose (Pose2D): robot position
        :param action (np.ndarray (,2)): [0] = linear velocity, [1] = angular velocity
        :param dist_to_path (float, optional): applies reward within this distance
        """
        if (
            self._curr_dist_to_path
            and action is not None
            and self._curr_dist_to_path <= dist_to_path
        ):
            self.curr_reward += 0.1 * action[0]

    def get_min_dist2global_kdtree(self, global_plan: np.array, robot_pose: Pose2D):
        """
        Calculates minimal distance to global plan using kd tree search.

        :param global_plan: (np.ndarray): vector containing poses on global plan
        :param robot_pose (Pose2D): robot position
        """
        if self.kdtree is None:
            self.kdtree = scipy.spatial.cKDTree(global_plan)

        dist, index = self.kdtree.query([robot_pose.x, robot_pose.y])
        return dist, index

    def _reward_abrupt_direction_change(self, action: np.ndarray = None):
        """
        Applies a penalty when an abrupt change of direction occured.

        :param action: (np.ndarray (,2)): [0] = linear velocity, [1] = angular velocity
        """
        if self.last_action is not None:
            curr_ang_vel = action[1]
            last_ang_vel = self.last_action[1]

            vel_diff = abs(curr_ang_vel - last_ang_vel)
            self.curr_reward -= (vel_diff**4) / 50
        self.last_action = action

    def _reward_reverse_drive(self, action: np.array = None, penalty: float = 0.01):
        """
        Applies a penalty when an abrupt change of direction occured.

        :param action: (np.ndarray (,2)): [0] = linear velocity, [1] = angular velocity
        """
        if action is not None and action[0] < 0:
            self.curr_reward -= penalty
        elif self.curr_vel < 0:
            self.curr_reward -= 0.2*self.curr_vel**2

    def _reward_high_conc_force(self, penalty: float = 0.03):
        if self.last_action is not None:
            # if self.last_action[1]*self._curr_action[1] > 0.04:
            #     self.angle += self.curr_avel
            #     self.factor = max(min(self.factor*(1 + np.sign(self.angle)*self.curr_avel/(5.5+2*abs(self.curr_vel))),20),0.5)
            #     fac = self.factor if np.sign(self.angle)*self.curr_avel > 0 else 0
            # else:
            #     self.angle *= 0.75
            #     self.factor = max(0.5,self.factor*0.75)
            #     fac = 0
            #fac = 1
            self.curr_reward -= penalty * (self.curr_vel*self.curr_avel)**2 if abs(self.curr_vel*self.curr_avel) > 1 else 0
            #self.curr_reward -= penalty*abs((abs(self.curr_vel)+0.4)**2*self._curr_action[1]/2 + self._curr_action[1]*fac/2)
    
    def _reward_steering(self, penalty=0.01):
        if self.last_action is not None:
            self.angle += self.curr_avel/10
            angle = 4*self.angle/(3*np.pi)
            factor = min((0.6*angle)**6,25)
            fac = 0.07*factor*abs(self._curr_action[1])/(0.8+abs(self.curr_vel)) if np.sign(self.angle)*self.curr_avel > 0 else 0
            self.curr_reward -= fac  
            if abs(self.last_action[1]*self._curr_action[1]) < 0.01:
                self.angle *= (1-min(abs(self.curr_vel)/2,0.7))

    def _reward_abrupt_vel_change(self, vel_idx: int, factor: float = 1):
        """
        Applies a penalty when an abrupt change of direction occured.

        :param action: (np.ndarray (,2)): [0] = linear velocity, [1] = angular velocity
        """
        if self.last_action is not None:
            curr_vel = self._curr_action[vel_idx]
            last_vel = self.last_action[vel_idx]

            vel_diff = abs(curr_vel - last_vel)
            self.curr_reward -= ((vel_diff**4) / 100) * factor

    def callback_stage_info(self,msg_stage_info):
        # if msg_stage_info.data <= self.stage_info:
        #     self.stage_info = 1
        # else:
        self.stage_info = msg_stage_info.data
