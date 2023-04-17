"""PettingZoo Environment for Single-/Multi Agent Reinforcement Learning"""
from typing import Any, Callable, Dict, List, Tuple, Union
from warnings import warn

import numpy as np
import rospy


from flatland_msgs.srv import StepWorld, StepWorldRequest
from gym import spaces
from pettingzoo import *
from pettingzoo.utils.conversions import from_parallel, to_parallel
from rl_utils.rl_utils.training_agent_wrapper import TrainingDRLAgent
from rl_utils.rl_utils.utils.supersuit_utils import MarkovVectorEnv_patched
from task_generator.tasks import get_MARL_task
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Int8
from ..utils.communication_channel import Channel

# from marl_agent.utils.supersuit_utils import *
# from rl_agent.utils.supersuit_utils import MarkovVectorEnv_patched
import supersuit as ss
from task_generator.msg import robot_goal, crate_action, robot_goal_list


def env_fn(**kwargs: Dict[str, Any]):  # -> VecEnv:
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = FlatlandPettingZooEnv(**kwargs)
    env = from_parallel(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pad_observations_v0(env)
    env = to_parallel(env)
    env = MarkovVectorEnv_patched(env, black_death=True)
    return env


class FlatlandPettingZooEnv(ParallelEnv):
    """The SuperSuit Parallel environment steps every live agent at once."""

    def __init__(
        self,
        # agent_list_fn: Callable[[int, str, str, str, str], List[TrainingDRLAgent]],
        agent_list,
        ns: str = None,
        task_manager_reset: Callable[[str], None] = None,
        max_num_moves_per_eps: int = 1000,
    ) -> None:
        """Initialization method for the Arena-Rosnav Pettingzoo Environment.

        Args:
            num_agents (int): Number of possible agents.
            agent_list_fn (Callable[ [int, str, str, str, str], List[TrainingDRLAgent] ]): Initialization function for the agents. \
                Returns a list of agent instances.
            ns (str, optional): Environments' ROS namespace. There should only be one env per ns. Defaults to None.
            task_mode (str, optional): Navigation task mode for the agents. Modes to chose from: ['random', 'staged']. \
                Defaults to "random".
            max_num_moves_per_eps (int, optional): Maximum number of moves per episode. Defaults to 1000.
            
        Note:
            These attributes should not be changed after initialization:
            - possible_agents
            - action_spaces
            - observation_spaces
        """
        self._ns = "" if ns is None or not ns else f"{ns}/"
        self._is_train_mode = rospy.get_param("/train_mode")
        self.ports = int(rospy.get_param("num_ports"))
        self.extended_setup = rospy.get_param("choose_goal", default = True)
        self.obs_goals = int(rospy.get_param("/observable_task_goals"))

        self.metadata = {}

        self.agent_list: List[TrainingDRLAgent] = agent_list

        # self.agent_list: List[TrainingDRLAgent] = agent_list_fn(
        #     num_agents, ns=ns, **(agent_list_kwargs or {})
        # )

        self.robot_model, self.agents = agent_list[0].robot_model, []
        # list containing the unique robot namespaces
        # used as identifier
        self.possible_agents = [a._robot_sim_ns for a in self.agent_list]
        self.dones = {agent: False for agent in self.possible_agents[:]}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agent_object_mapping = dict(zip(self.possible_agents, self.agent_list))


        if self.ports > 0:
            self.comm_channel = Channel(ns, self.agent_object_mapping, self.ports)

        self.terminal_observation = {}

        self._validate_agent_list()
        
        # task manager
        self.task_manager_reset = task_manager_reset

        # service clients
        if self._is_train_mode:
            self._service_name_step = f"{self._ns}step_world"
            self._sim_step_client = rospy.ServiceProxy(
                self._service_name_step, StepWorld
            )

        self._max_num_moves = max_num_moves_per_eps
        self.action_provided, self.curr_actions = False, {}

        rospy.set_param(
            f"{self._ns}training/{self.robot_model}/step_mode", "apply_actions"
        )  # "apply_actions" or "get_states"
        rospy.set_param(
            f"{self._ns}training/{self.robot_model}/reset_mode", "reset_states"
        )  # "reset_states" or "get_obs"
        print(f'{self._ns}goals')
        self.goal_publisher = rospy.Publisher(f'{self._ns}goals', robot_goal)
        self.stage_info = 0
        self._stage_info = rospy.Subscriber("stage_info", Int8, self.callback_stage_info)

    def observation_space(self, agent: str) -> spaces.Box:
        """Returns specific agents' observation space.

        Args:
            agent (str): Agent name as given in ``self.possible_agents``.

        Returns:
            spaces.Box: Observation space of type _gym.spaces_.
        """
        return self.agent_object_mapping[agent].observation_space

    def action_space(self, agent: str) -> spaces.Box:
        """Returns specific agents' action space.

        Args:
            agent (str): Agent name as given in ``self.possible_agents``.

        Returns:
            spaces.Box: Action space of type _gym.spaces_.
        """
        return self.agent_object_mapping[agent].action_space

    def _validate_agent_list(self) -> None:
        """Validates the agent list.

        Description:
            Checks if all agents are named differently. That means each robot adresses its own namespace.
        """
        assert len(self.possible_agents) == len(
            set(self.possible_agents)
        ), "Robot names and thus their namespaces, have to be unique!"

    def reset(self) -> Dict[str, np.ndarray]:
        """Resets the environment and returns the new set of observations (keyed by the agent name)

        Description:
            This method is called when all agents reach an end criterion. End criterions are: exceeding the \
            max number of steps per episode, crash or reaching the end criterion.
            The scene is then reseted.
            
        Returns:
            Dict[str, np.ndarray]: Observations dictionary in {_agent name_: _respective observations_}.
        """

        mode = rospy.get_param(f"{self._ns}training/{self.robot_model}/reset_mode")

        assert (
            mode == "reset_states" or mode == "get_obs"
        ), "Reset mode has to be either 'reset_states' or 'get_obs'"

        if mode == "reset_states":
            print("reset environment " + self._ns)
            self.agents, self.num_moves, self.terminal_observation = (
                self.possible_agents[:],
                0,
                {},
            )
            self.dones = {agent: False for agent in self.agents}
            # reset the reward calculator
            for agent in self.agents:
                self.agent_object_mapping[agent].reward_calculator.reset()
                self.agent_object_mapping[agent].set_package_boolean(False)
                self.agent_object_mapping[agent].set_reserved_status(False)
                self.agent_object_mapping[agent].set_agent_goal([Pose2D()])
            
            for agent in self.agents:
                noop = np.zeros(shape=self.action_space(agent).shape)
                self.agent_object_mapping[agent].publish_action(noop)
            # reset the task manager
            self.task_manager_reset(self.robot_model)
            # for agent in self.agents:
            #     merged, _dict = self.agent_object_mapping[agent].get_observations()
            #     print(_dict["current_pos"])
            if self.ports > 0:
                self.comm_channel.reset()
            self.action_provided, self.curr_actions = False, {}

            # After returning, we will manually take a step in the simulation
            # prepare next step to return the first observations after reset
            rospy.set_param(
                f"{self._ns}training/{self.robot_model}/reset_mode", "get_obs"
            )

            fake_obss = {
                agent: np.zeros(train_agent.observation_space.shape)
                for agent, train_agent in zip(self.agents, self.agent_list)
            }

            return fake_obss
        elif mode == "get_obs":
            # get first observations for the next episode
            observations = {
                agent: self.agent_object_mapping[agent].get_observations()[0]
                for agent in self.agents
            }
            # We have now stepped the simulation and can return the obs and prepare for next reset
            rospy.set_param(
                f"{self._ns}training/{self.robot_model}/reset_mode", "reset_states"
            )
            return observations

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Simulates one timestep and returns the most recent environment information.

        Description:
            This function takes in velocity commands and applies those to the simulation.
            Afterwards, agents' observations are retrieved from the current timestep and \
            the reward is calculated. \
            Proceeding with the ``RewardCalculator`` processing the observations and detecting certain events like \
            if a crash occured, a goal was reached. Those informations are returned in the '*reward\_info*' \
            which itself is a dictionary. \
            Eventually, dictionaries containing every agents' observations, rewards, done flags and \
            episode information is returned.

        Args:
            actions (Dict[str, np.ndarray]): Actions dictionary in {_agent name_: _respective observations_}.
            mode (str): Mode of step. Steps have to be devided into a apply action step, that simply publishes the action, \
            so that one can then manally step the simulation, and then again call the step function but this time to \
            retrieves the states.
            Modes to chose from: ['apply_actions', 'get_states'].

        Returns:
            Tuple[ Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Dict[str, Any]], ]: Observations, \
                rewards, done flags and episode informations dictionary.
        
        Note:
            Done reasons are mapped as follows: __0__ - episode length exceeded, __1__ - agent crashed, \
                __2__ - agent reached its goal.
        """
        ### NEW IDEA

        mode = rospy.get_param(f"{self._ns}training/{self.robot_model}/step_mode")
        assert mode in [
            "apply_actions",
            "get_states",
        ], "Step mode has to be either 'apply_action' or 'get_states'"

        if mode == "apply_actions":
            # First step is to apply the actions to each agent
            self.apply_action(actions)
            dones = {agent: False for agent in self.agents}
            rewards = {agent: 0 for agent in self.agents}
            infos = {agent: 0 for agent in self.agents}
            obss = {
                agent: np.zeros(train_agent.observation_space.shape)
                for agent, train_agent in zip(self.agents, self.agent_list)
            }

            # After returning, we will manually take a step in the simulation
            # prepare next step to return the states
            rospy.set_param(
                f"{self._ns}training/{self.robot_model}/step_mode", "get_states"
            )
            return obss, rewards, dones, infos
        elif mode == "get_states":

            # We have now stepped the simulation and can get the states
            rospy.set_param(
                f"{self._ns}training/{self.robot_model}/step_mode", "apply_actions"
            )
            return self.get_states()

        ### OLD IDEA
        # If a user passes in actions with no agents, then just return empty observations, etc.
        # if not actions:
        #     self.agents = []
        #     return {}, {}, {}, {}

        # # actions
        # for agent in self.possible_agents:
        #     if agent in actions:
        #         self.agent_object_mapping[agent].publish_action(actions[agent])
        #     else:
        #         noop = np.zeros(shape=self.action_space(agent).shape)
        #         self.agent_object_mapping[agent].publish_action(noop)

        # # todo: remove this function call from pettingZoo env step.
        # # fast-forward simulation
        # self.call_service_takeSimStep()
        # self.num_moves += 1

        # merged_obs, rewards, reward_infos = {}, {}, {}

        # for agent in actions:
        #     # observations
        #     merged, _dict = self.agent_object_mapping[agent].get_observations()
        #     merged_obs[agent] = merged

        #     # rewards and infos
        #     reward, reward_info = self.agent_object_mapping[agent].get_reward(
        #         action=actions[agent], obs_dict=_dict
        #     )
        #     rewards[agent], reward_infos[agent] = reward, reward_info

        # # dones & infos
        # dones, infos = self._get_dones(reward_infos), self._get_infos(reward_infos)

        # # remove done agents from the active agents list
        # self.agents = [agent for agent in self.agents if not dones[agent]]

        # for agent in self.possible_agents:
        #     # agent is done in this episode
        #     if agent in dones and dones[agent]:
        #         self.terminal_observation[agent] = merged_obs[agent]
        #         infos[agent]["terminal_observation"] = merged_obs[agent]
        #     # agent is done since atleast last episode
        #     elif agent not in self.agents:
        #         if agent not in infos:
        #             infos[agent] = {}
        #         infos[agent]["terminal_observation"] = self.terminal_observation[agent]

        # return merged_obs, rewards, dones, infos

    def apply_action(self, actions: dict) -> None:
        """_summary_

        Args:
            action (dict): _description_

        Returns:
            _type_: _description_
        """
        if self.action_provided:
            warn(
                "Disobeyed method order. Called 'apply_action' multiple times without retrieving states."
            )

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        # actions
        for agent in self.possible_agents:
            if agent in actions and not self.dones[agent]:
                self.agent_object_mapping[agent].publish_action(actions[agent])
            else:
                noop = np.zeros(shape=self.action_space(agent).shape)
                self.agent_object_mapping[agent].publish_action(noop)

        self.num_moves += 1
        self.action_provided, self.curr_actions = True, actions

    @staticmethod
    def _get_goal_pose_in_robot_frame(goal_pos: Pose2D, robot_pos: Pose2D):
        y_relative = goal_pos.y - robot_pos.y
        x_relative = goal_pos.x - robot_pos.x
        rho = (x_relative**2 + y_relative**2) ** 0.5
        theta = (np.arctan2(y_relative, x_relative) - robot_pos.theta + 4 * np.pi) % (
            2 * np.pi
        ) - np.pi
        return rho, theta

    def get_states(
        self,
    ) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]:
        assert self.action_provided, "No actions provided"
        merged_obs, rewards, reward_infos = {}, {}, {}
        calc_reward = True
        reserved_backup = []
        for agent in self.curr_actions:
            # observations
            merged, _dict = self.agent_object_mapping[agent].get_observations()
            
            _dict["package"] = self.agent_object_mapping[agent].get_package_boolean()
            # rewards and infos
            msg_size = _dict["msg_size"]
            if self.extended_setup and self.stage_info == 1:
                calc_reward = False
                choice = self.curr_actions[agent][2]
                if int(choice) != 0 and not self.agent_object_mapping[agent].get_reserved_status():
                    try:
                        goal = _dict["obs_goals"][int(choice)-1]
                        crate = _dict["obs_crates"][int(choice)-1]
                        crate_id = _dict["ids"][int(choice)-1]
                    except:
                        #print("wrong goal setting")
                        goal = Pose2D()
                    if not(goal.x == 0 and goal.y == 0) and goal not in reserved_backup:
                        reserved_backup.append(goal)
                        self.agent_object_mapping[agent].set_agent_goal([goal, crate, crate_id])
                        self.agent_publisher(crate_action(crate_action.BLOCK,"pub"), idx= crate_id,
                        r_type=self.agent_object_mapping[agent].robot_model, r_id=self.agent_object_mapping[agent]._ns_robot)
                        self.agent_object_mapping[agent].set_reserved_status(True)
                    
                if self.agent_object_mapping[agent].get_reserved_status():
                    _dict["goals_in_robot_frame"] = np.array([])
                    calc_reward = True
                if  self.agent_object_mapping[agent].get_agent_goal()[0] == Pose2D():
                    rho, theta = 0, 0
                    calc_reward = False
                else:
                    rho, theta = FlatlandPettingZooEnv._get_goal_pose_in_robot_frame(self.agent_object_mapping[agent].get_agent_goal()[0], _dict["robot_pose"])
                _dict["goal_in_robot_frame"] = [rho, theta]
            if calc_reward:
                reward, reward_info = self.agent_object_mapping[agent].get_reward(
                    action=self.curr_actions[agent], obs_dict=_dict
                )
            else:
                reward = -0.4
                reward_info = {"is_success": 0,
                                "is_done": False}
                #print("no reward at this stage")
            
            merged[-msg_size-5:-msg_size-3] = _dict["goal_in_robot_frame"]
            if reward_info["is_success"] == 1 and self.stage_info == 1:
                #set the package boolean True and save goal point, if robot reaches goal and doesnt have a package
                #set package boolean False and release goal, if robot reaches goal and has a package
                if self.agent_object_mapping[agent].get_package_boolean():
                    print(agent + " dropoff")
                    self.agent_object_mapping[agent].set_package_boolean(False)
                    crate_id = self.agent_object_mapping[agent].get_agent_goal()[-1] if self.extended_setup else reward_info["crate_id"]
                    self.agent_publisher(crate_action(crate_action.DROPOFF,"pub"), idx= crate_id,
                    r_type=self.agent_object_mapping[agent].robot_model,r_id=self.agent_object_mapping[agent]._ns_robot)
                    self.agent_object_mapping[agent].set_agent_goal([Pose2D()])
                    self.agent_object_mapping[agent].set_reserved_status(False)
                    reward += 30
                    if "eval" in self._ns:
                        self.dones[agent] = True

                else:
                    print(self.agent_object_mapping[agent]._ns_robot + " pickup at:")
                    print(_dict["current_pos"], self.agent_object_mapping[agent].get_agent_goal())
                    reward_info["is_done"] = False
                    self.agent_object_mapping[agent].set_package_boolean(True)
                    if self.extended_setup:
                        crate_id = self.agent_object_mapping[agent].get_agent_goal()[-1]
                        pub_goal = self.agent_object_mapping[agent].get_agent_goal()[0]
                        self.agent_object_mapping[agent].set_agent_goal([self.agent_object_mapping[agent].get_agent_goal()[1], crate_id])
                    else:
                        #merged[-3:-1] = _dict["crates_in_robot_frame"][int(reward_info["crate"]),:]
                        self.agent_object_mapping[agent].set_agent_goal(_dict["obs_crates"][int(reward_info["crate"])])
                        crate_id = reward_info["crate_id"]
                        pub_goal = _dict["obs_goals"][int(reward_info["crate"])]
                    self.agent_publisher(crate_action(crate_action.PICKUP,"pub"), idx = crate_id, r_type=self.agent_object_mapping[agent].robot_model,
                    r_id=self.agent_object_mapping[agent]._ns_robot, goal = pub_goal)
            if reward_info["is_done"] and self.stage_info < 1:
                if self.curr_actions[agent][2] == 0:
                    reward -= 0.4
                if not self.dones[agent]:
                    if reward_info["is_success"] == 1:
                        print(agent + " success")
                    else: print(agent + " collision")
                    self.dones[agent] = True

            pack_trafo = 1 if self.agent_object_mapping[agent].get_package_boolean() else 0
            merged[-msg_size-1] = pack_trafo
            if self.agent_object_mapping[agent]._agent_params["normalize"]:
                merged = self.agent_object_mapping[agent].normalize_observations(merged)
            merged_obs[agent] = merged
            
            #print(agent, reward, pack_trafo)
            rewards[agent], reward_infos[agent] = reward, reward_info

        # dones & infos
        dones, infos = self._get_dones(), self._get_infos(reward_infos)
        
        
        if self.ports > 0:
            msgs = {}
            for agent in self.curr_actions:
                try:
                    msg = self.comm_channel.retrieve_messages(agent)
                    msgs[agent] = np.hstack([msg[key][:-msg_size] for key in msg])
                except:
                    msgs[agent] = np.zeros(merged_obs[agent][3*self.obs_goals:-msg_size].size * self.ports)
                    print("no msg data")
            self.comm_channel.step(merged_obs, self.curr_actions)
            for agent in self.curr_actions:
                merged_obs[agent][-msg_size:] = np.hstack([msgs[agent], self.comm_channel.get_robots_in_range(agent)])
                #print(merged_obs[agent].shape)
                if int(self.curr_actions[agent][-1]) > 0:
                    rewards[agent] -= 0.8

        # remove done agents from the active agents list
        self.agents = [agent for agent in self.agents if not dones[agent]]

        for agent in self.possible_agents:
            # agent is done in this episode
            if agent in dones and dones[agent]:
                self.terminal_observation[agent] = merged_obs[agent]
                infos[agent]["terminal_observation"] = merged_obs[agent]
            # agent is done since atleast last episode
            elif agent not in self.agents:
                if agent not in infos:
                    infos[agent] = {}
                infos[agent]["terminal_observation"] = self.terminal_observation[agent]

        self.action_provided, self.curr_actions = False, {}

        return merged_obs, rewards, dones, infos
    def agent_publisher(self, action, r_type = 'jackal', goal = None, crate = None, idx=0, r_id = ''):
        task = robot_goal(
                    action,
                    idx, self._ns,
                    r_id, r_type,
                    goal,
                    crate
                    )
        self.goal_publisher.publish(task)
    @property
    def max_num_agents(self):
        return len(self.agents)

    # def call_service_takeSimStep(self, t: float = None):
    #     """Fast-forwards the simulation.

    #     Description:
    #         Simulates the Flatland simulation for a certain amount of seconds.

    #     Args:
    #         t (float, optional): Time in seconds. When ``t`` is None, time is forwarded by ``step_size`` s \
    #             (ROS parameter). Defaults to None.
    #     """
    #     request = StepWorldRequest() if t is None else StepWorldRequest(t)

    #     try:
    #         response = self._sim_step_client(request)
    #         rospy.logdebug("step service=", response)
    #     except rospy.ServiceException as e:
    #         rospy.logdebug(f"step Service call failed: {e}")

    def _get_dones(self, reward_infos: Dict[str, Dict[str, Any]] = {}) -> Dict[str, bool]:
        """Extracts end flags from the reward information dictionary.

        Args:
            reward_infos (Dict[str, Dict[str, Any]]): Episode information from the ``RewardCalculator`` in \
                {_agent name_: _reward infos_}.

        Returns:
            Dict[str, bool]: Dones dictionary in {_agent name_: _done flag_}
            
        Note:
            Relevant dictionary keys are: "is_done", "is_success", "done_reason"
        """
        if reward_infos == {}:
            return (
                {agent: self.dones[agent] for agent in self.agents}
                if self.num_moves < self._max_num_moves
                else {agent: True for agent in self.agents}
            )
        else:
            return (
                {agent: reward_infos[agent]["is_done"] for agent in self.agents}
                if self.num_moves < self._max_num_moves
                else {agent: True for agent in self.agents}
            )

    def callback_stage_info(self,msg_stage_info):
        # if msg_stage_info.data <= self.stage_info:
        #     self.stage_info = 1
        # else:
        self.stage_info = msg_stage_info.data
    def _get_infos(
        self, reward_infos: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Extracts the current episode information from the reward information dictionary.

        Args:
            reward_infos (Dict[str, Dict[str, Any]]): Episode information from the ``RewardCalculator`` in \
                {_agent name_: _reward infos_}.

        Returns:
            Dict[str, Dict[str, Any]]: Info dictionary in {_agent name_: _done flag_}
            
        Note:
            Relevant dictionary keys are: "is_done", "is_success", "done_reason"
        """
        infos = {agent: {} for agent in self.agents}
        for agent in self.agents:
            if reward_infos[agent]["is_done"]:
                infos[agent] = reward_infos[agent]
            elif self.num_moves >= self._max_num_moves:
                infos[agent] = {
                    "done_reason": 0,
                    "is_success": 0,
                }
        return infos
