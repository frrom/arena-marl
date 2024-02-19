#! /usr/bin/env python3
import threading
from typing import Tuple

from numpy.core.numeric import normalize_axis_tuple
import rospy
import random
import numpy as np
from collections import deque

import time  # for debuging
import threading

# observation msgs
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry

# services
from flatland_msgs.srv import StepWorld, StepWorldRequest

# message filter
import message_filters

# for transformations
from tf.transformations import *

from gym import spaces
import numpy as np

from std_msgs.msg import Bool
from task_generator.msg import robot_goal, crate_action, robot_goal_list


class ObservationCollector:
    def __init__(
        self,
        ns: str,
        num_lidar_beams: int,
        lidar_range: float,
        external_time_sync: bool = False,
    ):
        """a class to collect and merge observations

        Args:
            num_lidar_beams (int): [description]
            lidar_range (float): [description]
        """
        self.obs_goals = int(rospy.get_param("/observable_task_goals"))
        self.ns = ns
        if ns is None or ns == "":
            self.ns_prefix = ""
        else:
            self.ns_prefix = "/" + ns + "/"
        self._sim = ns.split('/')[0]
        self.ports = int(rospy.get_param("num_ports"))
        self.num_robots = int(rospy.get_param("num_robots"))
        self.extend = rospy.get_param("warehouse", default = True)
        self._action_in_obs = rospy.get_param("actions_in_obs", default=False)
        self.window = rospy.get_param("slinding_window", default=True)
        self.window_length = rospy.get_param("window_length", default=1)

        # define observation_space
        if not self._action_in_obs:
            self.observation_space = ObservationCollector._stack_spaces(
                (
                    spaces.Box(
                        low=0,
                        high=lidar_range,
                        shape=(num_lidar_beams,),
                        dtype=np.float32,
                    ),
                    spaces.Box(low=0, high=15, shape=(1,), dtype=np.float32),
                    spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                )
            )
        else:
            self.observation_space = ObservationCollector._stack_spaces(
                (
                    spaces.Box(
                        low=0,
                        high=lidar_range,
                        shape=(num_lidar_beams,),
                        dtype=np.float32,
                    ),
                    spaces.Box(
                        low=-2.0,
                        high=2.0,
                        shape=(2,),
                        dtype=np.float32,  # acceleration/linear vel
                    ),
                    spaces.Box(
                        low=-4.0,
                        high=4.0,
                        shape=(1,),
                        dtype=np.float32,  # angular vel
                    ),
                    spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(1,),
                        dtype=np.float32,  # (real) linear vel
                    ),
                    spaces.Box(low=0, high=15, shape=(1,), dtype=np.float32),  # rho_goal
                    spaces.Box(
                        low=-np.pi,
                        high=np.pi,
                        shape=(1,),
                        dtype=np.float32,  # theta_goal
                    ),
                    spaces.Box(
                        low=0.0,
                        high=40.0,
                        shape=(2,),
                        dtype=np.float32,  # current position -- enable for old model
                    ),
                    spaces.Box(
                        low=0,
                        high=1,
                        shape=(1,),
                        dtype=np.uint8,  # package_boolean
                    ),
                )
            )
        self.obs_size = self.observation_space.shape[0]
        self.msg_size = 0
        #print(self.obs_size)
        if self.extend:
            self.observation_space = ObservationCollector._stack_spaces((
                spaces.Box(low=0, high=15, shape=(self.obs_goals,), dtype=np.float32),  # rho
                spaces.Box(
                    low=-np.pi,
                    high=np.pi,
                    shape=(self.obs_goals,),
                    dtype=np.float32,  # theta
                ),
                spaces.Box(low=0, high=15, shape=(self.obs_goals,), dtype=np.float32), #target distance
                self.observation_space))
        self.window_size = 0
        if self.window:
            for _ in range(self.window_length):
                self.observation_space = ObservationCollector._stack_spaces((self.observation_space,
                    spaces.Box(
                        low=0,
                        high=lidar_range,
                        shape=(num_lidar_beams,),
                        dtype=np.float32,
                    ),
                    spaces.Box(
                        low=-2.0,
                        high=2.0,
                        shape=(2,),
                        dtype=np.float32,  # acceleration/linear vel
                    ),
                    spaces.Box(
                        low=-4.0,
                        high=4.0,
                        shape=(1,),
                        dtype=np.float32,  # angular vel
                    ),
                    spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(1,),
                        dtype=np.float32,  # (real) linear vel
                    ),
                    spaces.Box(low=0, high=15, shape=(1,), dtype=np.float32),  # rho_goal
                    spaces.Box(
                        low=-np.pi,
                        high=np.pi,
                        shape=(1,),
                        dtype=np.float32,  # theta_goal
                    ),
                ))
            self.window_size = self.window_length * (self.obs_size-3)

        if self.ports > 0:
            self.msg_size = self.obs_size*self.ports + self.ports
            for _ in range(self.ports):
                self.observation_space = ObservationCollector._stack_spaces((self.observation_space,
                    spaces.Box(
                        low=0,
                        high=lidar_range,
                        shape=(num_lidar_beams,),
                        dtype=np.float32,
                    ),
                    spaces.Box(
                        low=-2.0,
                        high=2.0,
                        shape=(2,),
                        dtype=np.float32,  # linear vel
                    ),
                    spaces.Box(
                        low=-4.0,
                        high=4.0,
                        shape=(1,),
                        dtype=np.float32,  # angular vel
                    ),
                    spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(1,),
                        dtype=np.float32,  # (real) linear vel
                    ),
                    spaces.Box(low=0, high=15, shape=(1,), dtype=np.float32),  # rho_goal
                    spaces.Box(
                        low=-np.pi,
                        high=np.pi,
                        shape=(1,),
                        dtype=np.float32,  # theta_goal
                    ),
                    spaces.Box(
                        low=0.0,
                        high=40.0,
                        shape=(2,),
                        dtype=np.float32,  # current position of agent j
                    ),
                    spaces.Box(
                        low=0,
                        high=1,
                        shape=(1,),
                        dtype=np.uint8,  # package_boolean
                    ),
                ))
            
            self.observation_space = ObservationCollector._stack_spaces((self.observation_space,
                spaces.Box(
                        low=0,
                        high=4,
                        shape=(self.ports,),
                        dtype=np.uint8,  #robots to choose from
                    )))
        self._laser_num_beams = num_lidar_beams
        print(self.observation_space.shape)
        # for frequency controlling
        self._action_frequency = 1 / rospy.get_param("/robot_action_rate")

        self._clock = Clock()
        self._scan = LaserScan()
        self._robot_pose = Pose2D()
        self._robot_vel = Twist()
        self._subgoal = Pose2D()
        self._globalplan = np.array([])
        self._subgoals = np.array([])
        self._crate_list = np.array([])
        self._id_list = np.array([])

        # train mode?
        self._is_train_mode = rospy.get_param("/train_mode")

        # synchronization parameters
        self._ext_time_sync = external_time_sync
        self._first_sync_obs = True  # whether to return first sync'd obs or most recent
        self.max_deque_size = 10
        self._sync_slop = 0.05

        self._laser_deque = deque()
        self._rs_deque = deque()

        # subscriptions
        # ApproximateTimeSynchronizer appears to be slow for training, but with real robot, own sync method doesn't accept almost any messages as synced
        # need to evaulate each possibility
        if self._ext_time_sync:
            self._scan_sub = message_filters.Subscriber(
                f"{self.ns_prefix}scan", LaserScan
            )
            self._robot_state_sub = message_filters.Subscriber(
                f"{self.ns_prefix}odom", Odometry
            )

            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self._scan_sub, self._robot_state_sub],
                self.max_deque_size,
                slop=self._sync_slop,
            )
            # self.ts = message_filters.TimeSynchronizer([self._scan_sub, self._robot_state_sub], 10)
            self.ts.registerCallback(self.callback_odom_scan)
        else:
            self._scan_sub = rospy.Subscriber(
                f"{self.ns_prefix}scan",
                LaserScan,
                self.callback_scan,
                tcp_nodelay=True,
            )

            self._robot_state_sub = rospy.Subscriber(
                f"{self.ns_prefix}odom",
                Odometry,
                self.callback_robot_state,
                tcp_nodelay=True,
            )

        # self._clock_sub = rospy.Subscriber(
        #     f'{self.ns_prefix}clock', Clock, self.callback_clock, tcp_nodelay=True)
        goal_topic = (
            f"{self.ns_prefix}subgoal"
            if rospy.get_param("num_robots", default=1) == 1
            else f"{self.ns_prefix}goal"
        )
        self._subgoal_sub = rospy.Subscriber(
            goal_topic, PoseStamped, self.callback_subgoal
        )
        self._subgoals_sub = rospy.Subscriber(
            f"{self.ns_prefix}open_tasks", robot_goal_list, self.callback_subgoals #first training stage
        )
        #print("listener ", f"{self.ns_prefix}open_tasks")
        self._subgoals2_sub = rospy.Subscriber(
            f"{self._sim}/open_tasks", robot_goal_list, self.callback_subgoals #last training stage (full warehouse)
        )

        self._globalplan_sub = rospy.Subscriber(
            f"{self.ns_prefix}globalPlan", Path, self.callback_global_plan
        )

        # service clients
        if self._is_train_mode:
            _sim_namespace = self.ns_prefix.split("/")[1]
            self._service_name_step = f"/{_sim_namespace}/step_world"
            self._sim_step_client = rospy.ServiceProxy(
                self._service_name_step, StepWorld
            )

    def get_observation_space(self):
        return self.observation_space

    def get_observations(self, *args, **kwargs):
        # apply action time horizon
        # if self._is_train_mode:
        # self.call_service_takeSimStep(self._action_frequency)
        # else:
        #     try:
        #         rospy.wait_for_message(f"{self.ns_prefix}next_cycle", Bool)
        #     except Exception:
        #         pass

        if not self._ext_time_sync:
            # try to retrieve sync'ed obs
            laser_scan, robot_pose = self.get_sync_obs()
            if laser_scan is not None and robot_pose is not None:
                # print("Synced successfully")
                self._scan = laser_scan
                self._robot_pose = robot_pose
            # else:
            #     print("Not synced")

        if len(self._scan.ranges) > 0:
            scan = self._scan.ranges.astype(np.float32)
        else:
            scan = np.zeros(self._laser_num_beams, dtype=float)

        goal_dists = np.zeros((self.obs_goals))
        pub_goals = np.zeros((self.obs_goals,2))
        pub_crates = np.zeros((self.obs_goals,2))
        
        for i, goal in enumerate(self._subgoals):
            if goal.x == 0:
                rho = 0
                theta = 1
            else:
                rho, theta = ObservationCollector._get_goal_pose_in_robot_frame(
                    goal, self._robot_pose
                )
            if goal_dists[-1] == 0:
                pub_goals[i,:] = [rho, theta]
                rho, theta = ObservationCollector._get_goal_pose_in_robot_frame(
                self._crate_list[i], self._robot_pose)
                pub_crates[i,:] = [rho, theta]
                goal_dists[i], _  = ObservationCollector._get_goal_pose_in_robot_frame(
                self._crate_list[i], goal)
            else:
                break
        rho, theta = ObservationCollector._get_goal_pose_in_robot_frame(self._subgoal, self._robot_pose) if self._subgoal.x != 0 else (0,0)
        merged_obs = (
            np.hstack([scan, np.array([rho, theta])])
            if not self._action_in_obs
            else np.hstack(
                [
                    scan,
                    kwargs.get("last_action", np.array([0, 0, 0])),
                    np.array([0]),
                    np.array([rho,theta]),
                    np.array([self._robot_pose.x, self._robot_pose.y]),
                    np.array([0])

                ]
            )
        )
        if self.window:
            window = np.zeros(self.window_size)
            merged_obs = np.hstack(
                [
                merged_obs,
                window
                ])
        if self.ports > 0:
            msgs = np.zeros(self.msg_size)
            merged_obs = np.hstack(
                [
                merged_obs,
                msgs
                ])
        if self.extend:
            merged_obs = np.hstack(
                [
                pub_goals[:,0],
                pub_goals[:,1],
                goal_dists,
                merged_obs
                ])
        
        obs_dict = {
            "laser_scan": scan,
            "goal_in_robot_frame": [rho, theta],
            "goals_in_robot_frame": pub_goals,
            "crates_in_robot_frame": pub_crates,
            "global_plan": self._globalplan,
            "robot_pose": self._robot_pose,
            "last_action": kwargs.get("last_action", np.array([0, 0, 0])),
            "current_pos": self._robot_pose,
            "ids": self._id_list,
            "obs_goals": self._subgoals,
            "obs_crates": self._crate_list,
            "curr_goal": self._subgoal,
        }

        self._laser_deque.clear()
        self._rs_deque.clear()
        return merged_obs, obs_dict

    @staticmethod
    def _get_goal_pose_in_robot_frame(goal_pos: Pose2D, robot_pos: Pose2D):
        y_relative = goal_pos.y - robot_pos.y
        x_relative = goal_pos.x - robot_pos.x
        rho = (x_relative**2 + y_relative**2) ** 0.5
        theta = (np.arctan2(y_relative, x_relative) - robot_pos.theta + 4 * np.pi) % (
            2 * np.pi
        ) - np.pi
        # angle =  np.arctan2(y_relative, x_relative) - robot_pos.theta
        # theta = angle - math.trunc(angle/np.pi)*2*np.pi
        return rho, theta

    def get_sync_obs(self):
        laser_scan = None
        robot_pose = None

        # print(f"laser deque: {len(self._laser_deque)}, robot state deque: {len(self._rs_deque)}")
        while len(self._rs_deque) > 0 and len(self._laser_deque) > 0:
            laser_scan_msg = self._laser_deque.popleft()
            robot_pose_msg = self._rs_deque.popleft()

            laser_stamp = laser_scan_msg.header.stamp.to_sec()
            robot_stamp = robot_pose_msg.header.stamp.to_sec()

            while abs(laser_stamp - robot_stamp) > self._sync_slop:
                if laser_stamp > robot_stamp:
                    if len(self._rs_deque) == 0:
                        return laser_scan, robot_pose
                    robot_pose_msg = self._rs_deque.popleft()
                    robot_stamp = robot_pose_msg.header.stamp.to_sec()
                else:
                    if len(self._laser_deque) == 0:
                        return laser_scan, robot_pose
                    laser_scan_msg = self._laser_deque.popleft()
                    laser_stamp = laser_scan_msg.header.stamp.to_sec()

            laser_scan = self.process_scan_msg(laser_scan_msg)
            robot_pose, _ = self.process_robot_state_msg(robot_pose_msg)

            if self._first_sync_obs:
                break

        # print(f"Laser_stamp: {laser_stamp}, Robot_stamp: {robot_stamp}")
        return laser_scan, robot_pose

    def call_service_takeSimStep(self, t=None):
        request = StepWorldRequest() if t is None else StepWorldRequest(t)
        timeout = 12
        try:
            for i in range(timeout):
                response = self._sim_step_client(request)
                rospy.logdebug("step service=", response)
                # print('took step')
                if response.success:
                    break
                if i == timeout - 1:
                    raise TimeoutError(
                        f"Timeout while trying to call '{self.ns_prefix}step_world'"
                    )
                # print("took step")
                time.sleep(0.33)

        except rospy.ServiceException as e:
            rospy.logdebug("step Service call failed: %s" % e)

    def callback_odom_scan(self, scan, odom):
        self._scan = self.process_scan_msg(scan)
        self._robot_pose, self._robot_vel = self.process_robot_state_msg(odom)

    def callback_clock(self, msg_Clock):
        self._clock = msg_Clock.clock.to_sec()
        return

    def callback_subgoal(self, msg_Subgoal):
        self._subgoal = self.process_subgoal_msg(msg_Subgoal)
        return
    def callback_close_subgoals(self, msg:robot_goal_list):
        self.callback_subgoals(msg)
        return
    
    def callback_subgoals(self, msg: robot_goal_list):
        goal_list = []
        crate_list = []
        ids = []
        #print(msg)
        for r_goal in msg.open_tasks:
            goal_list.append(r_goal.robot_goal)
            crate_list.append(r_goal.crate_goal)
            ids.append(r_goal.crate_id)
            # goal_list.append(self.pose3D_to_pose2D(r_goal.robot_goal))
            # crate_list.append(self.pose3D_to_pose2D(r_goal.crate_goal))
        self._subgoals = goal_list
        self._crate_list = crate_list
        self._id_list = ids
        return

    def callback_global_plan(self, msg_global_plan):
        self._globalplan = ObservationCollector.process_global_plan_msg(msg_global_plan)
        return

    def callback_scan(self, msg_laserscan):
        if len(self._laser_deque) == self.max_deque_size:
            self._laser_deque.popleft()
        self._laser_deque.append(msg_laserscan)

    def callback_robot_state(self, msg_robotstate):
        if len(self._rs_deque) == self.max_deque_size:
            self._rs_deque.popleft()
        self._rs_deque.append(msg_robotstate)

    def callback_observation_received(self, msg_LaserScan, msg_RobotStateStamped):
        # process sensor msg
        self._scan = self.process_scan_msg(msg_LaserScan)
        self._robot_pose, self._robot_vel = self.process_robot_state_msg(
            msg_RobotStateStamped
        )
        self.obs_received = True
        return

    def process_scan_msg(self, msg_LaserScan: LaserScan):
        # remove_nans_from_scan
        self._scan_stamp = msg_LaserScan.header.stamp.to_sec()
        scan = np.array(msg_LaserScan.ranges)
        scan[np.isnan(scan)] = msg_LaserScan.range_max
        msg_LaserScan.ranges = scan
        return msg_LaserScan

    def process_robot_state_msg(self, msg_Odometry):
        pose3d = msg_Odometry.pose.pose
        twist = msg_Odometry.twist.twist
        return self.pose3D_to_pose2D(pose3d), twist

    def process_pose_msg(self, msg_PoseWithCovarianceStamped):
        # remove Covariance
        pose_with_cov = msg_PoseWithCovarianceStamped.pose
        pose = pose_with_cov.pose
        return self.pose3D_to_pose2D(pose)

    def process_subgoal_msg(self, msg_Subgoal):
        return self.pose3D_to_pose2D(msg_Subgoal.pose)

    @staticmethod
    def process_global_plan_msg(globalplan):
        global_plan_2d = list(
            map(
                lambda p: ObservationCollector.pose3D_to_pose2D(p.pose),
                globalplan.poses,
            )
        )
        return np.array(list(map(lambda p2d: [p2d.x, p2d.y], global_plan_2d)))

    @staticmethod
    def pose3D_to_pose2D(pose3d):
        pose2d = Pose2D()
        pose2d.x = pose3d.position.x
        pose2d.y = pose3d.position.y
        quaternion = (
            pose3d.orientation.x,
            pose3d.orientation.y,
            pose3d.orientation.z,
            pose3d.orientation.w,
        )
        euler = euler_from_quaternion(quaternion)
        yaw = euler[2]
        pose2d.theta = yaw
        return pose2d

    @staticmethod
    def _stack_spaces(ss: Tuple[spaces.Box]):
        low = []
        high = []
        for space in ss:
            low.extend(space.low.tolist())
            high.extend(space.high.tolist())
        return spaces.Box(np.array(low).flatten(), np.array(high).flatten())


if __name__ == "__main__":

    rospy.init_node("states", anonymous=True)
    print("start")

    state_collector = ObservationCollector("sim1/", 360, 10)
    i = 0
    r = rospy.Rate(100)
    while i <= 1000:
        i = i + 1
        obs = state_collector.get_observations()

        time.sleep(0.001)
