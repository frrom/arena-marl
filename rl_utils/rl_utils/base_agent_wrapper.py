from abc import ABC, abstractmethod
from typing import Tuple

import json
import numpy as np
import os
import rospy
import rospkg
import yaml

from gym import spaces

from geometry_msgs.msg import Twist

from .utils.observation_collector import ObservationCollector
from .utils.reward import RewardCalculator
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped

# robot_model = rospy.get_param("model")
ROOT_ROBOT_PATH = os.path.join(
    rospkg.RosPack().get_path("arena-simulation-setup"), "robot"
)
DEFAULT_HYPERPARAMETER = os.path.join(
    rospkg.RosPack().get_path("training"),
    "configs",
    "hyperparameters",
    "default.json",
)
DEFAULT_NUM_LASER_BEAMS, DEFAULT_LASER_RANGE = 360, 3.5
GOAL_RADIUS = 0.33


class BaseDRLAgent(ABC):
    def __init__(
        self,
        ns: str = None,
        robot_model: str = "burger",
        robot_ns: str = None,
        hyperparameter_path: str = DEFAULT_HYPERPARAMETER,
    ) -> None:
        """[summary]

        Args:
            ns (str, optional):
                Agent name (directory has to be of the same name). Defaults to None.
            robot_model (str, optional):
                Robot model name. Defaults to "burger".
            robot_ns (str, optional):
                Robot specific ROS namespace extension. Defaults to None.
            hyperparameter_path (str, optional):
                Path to json file containing defined hyperparameters.
                Defaults to DEFAULT_HYPERPARAMETER.
        """
        self._is_train_mode = rospy.get_param("/train_mode")

        self._ns = "" if ns is None or not ns else f"{ns}/"
        self._ns_robot = self._ns if robot_ns is None else self._ns + robot_ns
        self._robot_sim_ns = robot_ns

        self.robot_model = robot_model
        self.package_bool = False
        self.reserved_goal = False
        self.agent_goal = [Pose2D()]
        self.position = Pose2D()
        self.extend = rospy.get_param("choose_goal", default = False)
        self.ports = int(rospy.get_param("num_ports"))
        self.max_steps = int(rospy.get_param("n_moves"))
        self.last_pos = None
        self.reserving = 0
        #print(self.extend)

        robot_setting_path = os.path.join(
            ROOT_ROBOT_PATH, f"{self.robot_model}", f"{self.robot_model}.model.yaml"
        )

        action_space_path = os.path.join(
            rospkg.RosPack().get_path("arena-simulation-setup"),
            "robot",
            f"{self.robot_model}",
            "model_params.yaml",
        )

        self.read_setting_files(robot_setting_path, action_space_path)
        self.load_hyperparameters(path=hyperparameter_path)
        # self._check_robot_type_from_params()

        self.setup_action_space(extend=False)
        self.setup_reward_calculator()

        self.observation_collector = ObservationCollector(
            self._ns_robot, self._num_laser_beams, self._laser_range
        )
        self.obs_size = self.observation_collector.obs_size
        self.msg_size = self.observation_collector.msg_size
        self.sliding_window = np.zeros((self.observation_collector.window_length, self.observation_collector.obs_size-3))
        # for time controlling in train mode
        self._action_frequency = 1 / rospy.get_param("/robot_action_rate")

        if self._is_train_mode:
            # w/o action publisher node
            self._action_pub = rospy.Publisher(
                f"{self._ns_robot}/cmd_vel", Twist, queue_size=1
            )
        else:
            # w/ action publisher node
            # (controls action rate being published on '../cmd_vel')
            self._action_pub = rospy.Publisher(
                f"{self._ns_robot}/cmd_vel_pub", Twist, queue_size=1
            )

    @abstractmethod
    def setup_agent(self) -> None:
        """Sets up the new agent / loads a pretrained one.

        Raises:
            NotImplementedError: Abstract method.
        """
        raise NotImplementedError

    def load_hyperparameters(self, path: str) -> None:
        """Loads the hyperparameters from a json file.

        Args:
            path (str): Path to the json file.
        """
        assert os.path.isfile(path), f"Hyperparameters file cannot be found at {path}!"

        with open(path, "r") as file:
            hyperparams = json.load(file)

        self._agent_params = hyperparams
        self.robot_config_name = self.robot_model
        rospy.set_param(
            "actions_in_obs",
            self._agent_params.get("actions_in_observationspace", False),
        )

        # todo: What are the imports for?
        # import rosnav.model.custom_policy
        import rosnav.model.custom_sb3_policy
        import rosnav.model.custom_lstm_policy
        import rosnav.model.custom_policy
    def get_package_boolean(self):
        return self.package_bool
    def set_package_boolean(self, pack):
        self.package_bool = pack
    def get_agent_goal(self):
        return self.agent_goal
    def set_agent_goal(self, goal):
        self.agent_goal = goal
    def get_reserved_status(self):
        return self.reserved_goal
    def set_reserved_status(self,a):
        self.reserved_goal = a
    def read_setting_files(
        self, robot_setting_yaml: str, action_space_yaml: str
    ) -> None:
        """Retrieves the robot radius (in 'self._robot_radius'), \
            laser scan range (in 'self._laser_range') and \
            the action space from respective yaml file.

        Args:
            robot_setting_yaml (str): 
                Yaml file containing the robot specific settings. 
            action_space_yaml (str): 
                Yaml file containing the action space configuration. 
        """
        self._num_laser_beams = None
        self._laser_range = None

        with open(action_space_yaml, "r", encoding="utf-8") as target:
            config = yaml.load(target, Loader=yaml.FullLoader)

        self._robot_radius = config["robot_radius"] * 1.05

        with open(robot_setting_yaml, "r") as fd:
            robot_data = yaml.safe_load(fd)

            # get laser related information
            for plugin in robot_data["plugins"]:
                if plugin["type"] == "Laser":
                    laser_angle_min = plugin["angle"]["min"]
                    laser_angle_max = plugin["angle"]["max"]
                    laser_angle_increment = plugin["angle"]["increment"]
                    self._num_laser_beams = int(
                        round(
                            (laser_angle_max - laser_angle_min) / laser_angle_increment
                        )
                    )
                    self._laser_range = plugin["range"]

        if self._num_laser_beams is None:
            self._num_laser_beams = DEFAULT_NUM_LASER_BEAMS
            print(
                f"{self._robot_sim_ns}:"
                "Wasn't able to read the number of laser beams."
                "Set to default: {DEFAULT_NUM_LASER_BEAMS}"
            )
        if self._laser_range is None:
            self._laser_range = DEFAULT_LASER_RANGE
            print(
                f"{self._robot_sim_ns}:"
                "Wasn't able to read the laser range."
                "Set to default: {DEFAULT_LASER_RANGE}"
            )

        with open(action_space_yaml, "r") as fd:
            setting_data = yaml.safe_load(fd)

            self._holonomic = setting_data["is_holonomic"]
            self._discrete_actions = setting_data["actions"]["discrete"]
            self._cont_actions = {
                "linear_range": setting_data["actions"]["continuous"]["linear_range"],
                "angular_range": setting_data["actions"]["continuous"]["angular_range"],
            }

        rospy.set_param("laser/num_beams", self._num_laser_beams)

    def _check_robot_type_from_params(self):
        """Retrives the agent-specific robot name from the dictionary loaded\
            from respective 'hyperparameter.json' and compares it to the provided
            robot model from initialization.    
        """
        assert self._agent_params and self._agent_params["robot"]
        assert self.robot_model == self._agent_params["robot"], (
            "Robot model in hyperparameter.json is not the same as the parsed model!"
            f"({self.robot_model} != {self._agent_params['robot']})"
        )

    def setup_action_space(self, extend=False) -> None:
        """Sets up the action space. (spaces.Box)"""
        assert self._discrete_actions or self._cont_actions
        assert self._agent_params and "discrete_action_space" in self._agent_params

        if self._agent_params["discrete_action_space"]:
            # self._discrete_actions is a list, each element is a dict with the keys ["name", 'linear','angular']
            assert (
                not self._holonomic
            ), "Discrete action space currently not supported for holonomic robots"

            self.action_space = spaces.Discrete(len(self._discrete_actions))
        else:
            linear_range = self._cont_actions["linear_range"].copy()
            angular_range = self._cont_actions["angular_range"].copy()

            if not self._holonomic:
                self._action_space = spaces.Box(
                    low=np.array([linear_range[0], angular_range[0]]),
                    high=np.array([linear_range[1], angular_range[1]]),
                    dtype=np.float32,
                )
                #print("holonomic agent")
            else:
                #print("non-holonomic agent")
                linear_range_x, linear_range_y = (
                    linear_range["x"],
                    linear_range["y"],
                )
                self._action_space = spaces.Box(
                    low=np.array(
                        [
                            linear_range_x[0],
                            linear_range_y[0],
                            angular_range[0],
                        ]
                    ),
                    high=np.array(
                        [
                            linear_range_x[1],
                            linear_range_y[1],
                            angular_range[1],
                        ]
                    ),
                    dtype=float,
                )
        if self.extend:
            #print("using extended action space")
            self._action_space = BaseDRLAgent._stack_spaces((self._action_space, spaces.Box(low=0,high=5,shape=(1,),dtype=float,))) # 6 for new model, 5 for old model
        self._action_space = BaseDRLAgent._stack_spaces((self._action_space, spaces.Box(low=0,high=2,shape=(self.ports,),dtype=float,)))

    def setup_reward_calculator(self) -> None:
        """Sets up the reward calculator."""
        assert self._agent_params and "reward_fnc" in self._agent_params

        rule = self._agent_params["reward_fnc"] if self.extend else "rule_07"
        self.reward_calculator = RewardCalculator(
            holonomic=self._holonomic,
            robot_radius=self._robot_radius,
            safe_dist=1.6 * self._robot_radius,
            goal_radius=GOAL_RADIUS,
            rule = rule,
            #rule=self._agent_params["reward_fnc"],
            extended_eval=False,
            max_steps = self.max_steps
        )

    @property
    def action_space(self) -> spaces.Box:
        """Returns the DRL agent's action space.

        Returns:
            spaces.Box: Agent's action space
        """
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        """Returns the DRL agent's observation space.

        Returns:
            spaces.Box: Agent's observation space
        """
        return self.observation_collector.observation_space

    def get_observations(self) -> Tuple[np.ndarray, dict]:
        """Retrieves the latest synchronized observation.

        Returns:
            Tuple[np.ndarray, dict]: 
                Tuple, where first entry depicts the observation data concatenated \
                into one array. Second entry represents the observation dictionary.
        """
        self.merged_obs, self.obs_dict = self.observation_collector.get_observations()
        # pack_trafo = 1 if self.package_bool else 0
        # merged_obs[-1] = pack_trafo
        # if self._agent_params["normalize"]:
        #     merged_obs = self.normalize_observations(merged_obs)
        return self.merged_obs, self.obs_dict

    def normalize_observations(self, merged_obs: np.ndarray) -> np.ndarray:
        """Normalizes the observations with the loaded VecNormalize object.

        Note:
            VecNormalize object from Stable-Baselines3 is agent specific\
            and integral part in order to map right actions.\

        Args:
            merged_obs (np.ndarray):
                observation data concatenated into one array.

        Returns:
            np.ndarray: Normalized observations array.
        """
        assert self._agent_params["normalize"] and hasattr(self, "_obs_norm_func")
        return self._obs_norm_func(merged_obs)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Infers an action based on the given observation.

        Args:
            obs (np.ndarray): Merged observation array.

        Returns:
            np.ndarray:
                Action in [linear velocity, angular velocity]
        """
        assert self._agent, "Agent model not initialized!"
        action = self._agent.predict(obs, deterministic=True)[0]
        if self._agent_params["discrete_action_space"]:
            action = self._get_disc_action(action)
        else:
            # clip action
            action = np.maximum(
                np.minimum(self._action_space.high, action),
                self._action_space.low,
            )
        return action

    def get_reward(self, action: np.ndarray, obs_dict: dict) -> float:
        """Calculates the reward based on the parsed observation

        Args:
            action (np.ndarray):
                Velocity commands of the agent\
                in [linear velocity, angular velocity].
            obs_dict (dict):
                Observation dictionary where each key makes up a different \
                kind of information about the environment.
        Returns:
            float: Reward amount
        """
        
        return self.reward_calculator.get_reward(action=action, **obs_dict)

    def publish_action(self, action: np.ndarray) -> None:
        """Publishes an action on 'self._action_pub' (ROS topic).

        Args:
            action (np.ndarray):
                Action in [linear velocity, angular velocity]
        """
        action_msg = (
            self._get_hol_action_msg(action)
            if self._holonomic
            else self._get_nonhol_action_msg(action)
        )
        self._action_pub.publish(action_msg)

    def _get_disc_action(self, action: int) -> np.ndarray:
        """Returns defined velocity commands for parsed action index.\
            (Discrete action space)

        Args:
            action (int): Index of the desired action.

        Returns:
            np.ndarray: Velocity commands corresponding to the index.
        """
        return np.array(
            [
                self._discrete_actions[action]["linear"],
                self._discrete_actions[action]["angular"],
            ]
        )

    def _get_hol_action_msg(self, action: np.ndarray):
        assert (
            len(action) == 3
        ), "Holonomic robots require action arrays to have 3 entries."
        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.linear.y = action[1]
        action_msg.angular.z = action[2]
        return action_msg

    def _get_nonhol_action_msg(self, action: np.ndarray):
        assert (
            len(action) == 2
        ), "Non-holonomic robots require action arrays to have 2 entries."
        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.angular.z = action[1]
        return action_msg

    @staticmethod
    def _stack_spaces(ss: Tuple[spaces.Box]):
        low = []
        high = []
        for space in ss:
            low.extend(space.low.tolist())
            high.extend(space.high.tolist())
        return spaces.Box(np.array(low).flatten(), np.array(high).flatten())
