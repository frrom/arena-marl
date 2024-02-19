import os
from typing import Tuple

import gym
import rospkg
import rospy
import torch as th
import yaml
from stable_baselines3.common.policies import BaseFeaturesExtractor
from torch import nn

from ..utils.utils import get_observation_space_from_file, get_extended_observation_space_from_file
#from rl_utils.rl_utils.utils.utils import get_observation_space_from_file

class EXTRACTOR_1(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.
    Architecture was taken as reference from: https://arxiv.org/abs/1808.03841

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.

    Note:
        self._rs: Robot state size - placeholder for robot related inputs to the NN
        self._l: Number of laser beams - placeholder for the laser beam data
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 128,
    ):
        self._l, self._rs = get_observation_space_from_file(robot_model)
        super(EXTRACTOR_1, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc_1(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_2(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.
    Architecture was taken as reference from: https://arxiv.org/abs/1808.03841
    (DRLself._lOCAL_PLANNER)

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 128,
    ):
        self._l, self._rs = get_observation_space_from_file(robot_model)
        super(EXTRACTOR_2, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc_1(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_3(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(EXTRACTOR_3, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
        )

        self.fc_2 = nn.Sequential(nn.Linear(256, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc_2(self.fc_1(self.cnn(laser_scan)))
        # return self.fc_2(features)
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_4(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.
    Architecture was taken as reference from: https://github.com/ethz-asl/navrep
    (CNN_NAVREP)

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
    ):
        self._l, self._rs = get_observation_space_from_file(robot_model)
        super(EXTRACTOR_4, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 9, 4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 6, 4),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_5(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
    ):
        self._l, self._rs = get_observation_space_from_file(robot_model)
        super(EXTRACTOR_5, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_6(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
    ):
        self._l, self._rs = get_observation_space_from_file(robot_model)
        super(EXTRACTOR_6, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)

class EXTRACTOR_7(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
        obs_space = None
    ):
        self._l, self._rs, self._msg = get_extended_observation_space_from_file(robot_model)
        self.ports = int(rospy.get_param("num_ports"))
        self.obs_goals = 3*int(rospy.get_param("observable_task_goals"))
        spaces = features_dim + self._rs - self.obs_goals + self.ports + int(min(features_dim/10, self.obs_goals))

        if self.ports > 0:
            spaces += features_dim 
            
        super(EXTRACTOR_7, self).__init__(observation_space, spaces)
        
        if self.obs_goals > 0:
            self.dsc_cnn = nn.Sequential(
                nn.Conv1d(1, 32, 4, 2),
                nn.ReLU(),
                nn.Conv1d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Compute shape by doing one forward pass
            with th.no_grad():
                tensor_forward = th.randn(1, 1, self.obs_goals)
                n_flatten = self.dsc_cnn(tensor_forward).shape[1]
            
            self.dsc_fc = nn.Sequential(
                nn.Linear(n_flatten, int(min(features_dim/10, self.obs_goals))),
                nn.ReLU(),
            )

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        if self.ports > 0:
            self.msg_cnn = nn.Sequential(
                nn.Conv1d(1, 128, 8, 4),
                nn.ReLU(),
                nn.Conv1d(128, 48, 4, 2),
                nn.ReLU(),
                nn.Conv1d(48, 86, 4, 2),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Compute shape by doing one forward pass
            with th.no_grad():
                tensor_forward = th.randn(1, 1, self._msg)
                n_flatten = self.msg_cnn(tensor_forward).shape[1]

            self.msg_fc = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )
        self.spaces = spaces
        print("feature size: ", spaces)
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, self.obs_goals: self.obs_goals+self._l], 1)

        if self.obs_goals > 0:
            pub_goals = th.unsqueeze(observations[:, :self.obs_goals], 1)
            goal_features = self.dsc_fc(self.dsc_cnn(pub_goals))
        else:
            pub_goals = []
            goal_features = laser_scan.size(0)
        
        if self.ports > 0:
            robot_state = observations[:, self.obs_goals+self._l :-self._msg-self.ports]
            msgs = th.unsqueeze(observations[:, self._rs+self._l:-self.ports], 1)
            msg_features = self.msg_fc(self.msg_cnn(msgs))
            ports = observations[:, -self.ports:]
        else:
            robot_state = observations[:, self.obs_goals+self._l :]
            msgs = []
            num_rows = laser_scan.size(0)
            # Create empty tensor with the same number of rows as scan
            msg_features = th.empty((num_rows, 0))
            ports = th.empty((num_rows, 0))

        extracted_features = self.fc(self.cnn(laser_scan))
        
        
        return th.cat((goal_features, extracted_features, robot_state, msg_features, ports), 1)

    def _return_output_size(self):
        return self.spaces
    
class PSEUDO_RNN_EXTRACTOR(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
        obs_space = None
    ):
        self._l, self._rs, self._msg = get_extended_observation_space_from_file(robot_model)
        print(self._l + self._rs + self._msg)
        self.ports = int(rospy.get_param("num_ports"))
        self.obs_goals = 3*int(rospy.get_param("observable_task_goals"))
        spaces = features_dim + self.ports + int(min(features_dim/10, self.obs_goals))

        if self.ports > 0:
            spaces += features_dim 
        
        super(PSEUDO_RNN_EXTRACTOR, self).__init__(observation_space, spaces)

        if self.obs_goals > 0:
            self.dsc_cnn = nn.Sequential(
                nn.Conv1d(1, 32, 4, 2),
                nn.ReLU(),
                nn.Conv1d(32, 16, 4, 1),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Compute shape by doing one forward pass
            with th.no_grad():
                tensor_forward = th.randn(1, 1, self.obs_goals)
                n_flatten = self.dsc_cnn(tensor_forward).shape[1]
            
            self.dsc_fc = nn.Sequential(
                nn.Linear(n_flatten, int(min(features_dim/10, self.obs_goals))),
                nn.ReLU(),
            )

        s = int(self._l/2)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, s, 16, 4),
            nn.ReLU(),
            nn.Conv1d(s, 128, 8, 2),
            nn.ReLU(),
            nn.Conv1d(128, 64, 4, 1),
            nn.ReLU(),
            nn.Conv1d(64, 16, 4, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.body_net = nn.LSTM(input_size=n_flatten + self._rs - self.obs_goals,
                                hidden_size=128,
                                num_layers=1,
                                batch_first=True)
        
        self.linear_layer = nn.Linear(128, features_dim)
        self.activation_fn = nn.ReLU()


        if self.ports > 0:
            s = int(self._msg/2)
            self.msg_cnn = nn.Sequential(
                nn.Conv1d(1, s, 16, 4),
                nn.ReLU(),
                nn.Conv1d(s, 128, 8, 2),
                nn.ReLU(),
                nn.Conv1d(128, 64, 4, 1),
                nn.ReLU(),
                nn.Conv1d(64, 16, 4, 1),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Compute shape by doing one forward pass
            with th.no_grad():
                tensor_forward = th.randn(1, 1, self._msg-self.ports)
                n_flatten = self.msg_cnn(tensor_forward).shape[1]

            self.msg_fc = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )
        self.spaces = spaces
        print("feature size: ", spaces)
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, self.obs_goals: self.obs_goals+self._l], 1)

        if self.obs_goals > 0:
            pub_goals = th.unsqueeze(observations[:, :self.obs_goals], 1)
            goal_features = self.dsc_fc(self.dsc_cnn(pub_goals))
        else:
            pub_goals = []
            goal_features = laser_scan.size(0)
        
        if self.ports > 0:
            robot_state = observations[:, self.obs_goals+self._l :-self._msg]
            msgs = th.unsqueeze(observations[:, self._rs+self._l:-self.ports], 1)
            msg_features = self.msg_fc(self.msg_cnn(msgs))
            ports = observations[:, -self.ports:]
        else:
            robot_state = observations[:, self.obs_goals+self._l :]
            msgs = []
            num_rows = laser_scan.size(0)
            # Create empty tensor with the same number of rows as scan
            msg_features = th.empty((num_rows, 0))
            ports = th.empty((num_rows, 0))

        extracted_features, (h_n, c_n) = self.body_net(th.cat((self.cnn(laser_scan),robot_state),1))
        extracted_features = self.linear_layer(extracted_features)
        extracted_features = self.activation_fn(extracted_features)
        
        self.feat = th.cat((goal_features, extracted_features, msg_features, ports), 1)
    
        return self.feat
    
    # def backward(self):
    #     #Why is retain_variables True??
    #     self.loss.backward(retain_graph=True)
    #     return self.loss

class EXTRACTOR_8(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
        obs_space = None
    ):
        self._l, self._rs, self._msg = get_extended_observation_space_from_file(robot_model)
        
        self.ports = int(rospy.get_param("num_ports"))
        self.obs_goals = 3*int(rospy.get_param("observable_task_goals"))
        self.window_length = rospy.get_param("window_length", default=1)
        self.window = (self._l + 6)
        spaces = features_dim + self._rs - self.obs_goals + self.ports + int(min(features_dim/10, self.obs_goals)) + int(self.window_length*features_dim/25)
        print("input_size = " , self._l + self._rs + self._msg + self.window_length*self.window) 
        if self.ports > 0:
            spaces += features_dim 
        
        super(EXTRACTOR_8, self).__init__(observation_space, spaces)

        if self.obs_goals > 0:
            self.dsc_cnn = nn.Sequential(
                nn.Conv1d(1, 32, 4, 2),
                nn.ReLU(),
                nn.Conv1d(32, 16, 4, 1),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Compute shape by doing one forward pass
            with th.no_grad():
                tensor_forward = th.randn(1, 1, self.obs_goals)
                n_flatten = self.dsc_cnn(tensor_forward).shape[1]
            
            self.dsc_fc = nn.Sequential(
                nn.Linear(n_flatten, int(min(features_dim/10, self.obs_goals))),
                nn.ReLU(),
            )

        s = int(self._l/2)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, s, 16, 4),
            nn.ReLU(),
            nn.Conv1d(s, 128, 8, 2),
            nn.ReLU(),
            nn.Conv1d(128, 64, 4, 1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 4, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]
        
        self.fc = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )
        
        if self.window_length > 0:
            self.window_net = nn.Sequential(
                nn.Linear(self.window_length*(n_flatten+self.window-self._l), features_dim),
                nn.ReLU(),
                nn.Linear(features_dim, int(self.window_length*features_dim/25)),
                nn.ReLU(),
            )

        if self.ports > 0:
            s = int(self._msg/2)
            self.msg_cnn = nn.Sequential(
                nn.Conv1d(1, s, 16, 4),
                nn.ReLU(),
                nn.Conv1d(s, 128, 8, 2),
                nn.ReLU(),
                nn.Conv1d(128, 64, 4, 1),
                nn.ReLU(),
                nn.Conv1d(64, 16, 4, 1),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Compute shape by doing one forward pass
            with th.no_grad():
                tensor_forward = th.randn(1, 1, self._msg-self.ports)
                n_flatten = self.msg_cnn(tensor_forward).shape[1]

            self.msg_fc = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )
        self.spaces = spaces
        print("feature size: ", spaces)
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, self.obs_goals: self.obs_goals+self._l], 1)

        if self.obs_goals > 0:
            pub_goals = th.unsqueeze(observations[:, :self.obs_goals], 1)
            goal_features = self.dsc_fc(self.dsc_cnn(pub_goals))
        else:
            pub_goals = []
            goal_features = laser_scan.size(0)
        
        
        if self.window_length>0:
            for i in range(self.window_length):
                win = observations[:,self._rs+self._l+i*self.window:self._rs+self._l+(i+1)*self.window]
                if i == 0:
                    window = th.cat((self.cnn(th.unsqueeze(win[:self._l],1)),win[:,self._l:]),1)
                else:
                    window = th.cat((window, self.cnn(th.unsqueeze(win[:self._l],1)),win[:,self._l:]),1)
            window_features  = self.window_net(window)
        else:
            window = []
            num_rows = laser_scan.size(0)
            window_features = th.empty((num_rows, 0))

        if self.ports > 0:
            #robot_state = observations[:, self.obs_goals+self._l :-self._msg-self.window*self.window_length]
            msgs = th.unsqueeze(observations[:, self._rs+self._l+self.window_length*self.window:-self.ports], 1)
            msg_features = self.msg_fc(self.msg_cnn(msgs))
            ports = observations[:, -self.ports:]
        else:
            msgs = []
            num_rows = laser_scan.size(0)
            # Create empty tensor with the same number of rows as scan
            msg_features = th.empty((num_rows, 0))
            ports = th.empty((num_rows, 0))

        robot_state = observations[:, self.obs_goals+self._l:self._rs+self._l]
        lidar_features = self.fc(self.cnn(laser_scan))
        self.feat = th.cat((goal_features, lidar_features, robot_state, window_features, msg_features, ports), 1)

        return self.feat

class LSTM_EXTRACTOR(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
        obs_space = None
    ):
        self._l, self._rs, self._msg = get_extended_observation_space_from_file(robot_model)
        print(self._l + self._rs + self._msg)
        self.ports = int(rospy.get_param("num_ports"))
        self.obs_goals = 3*int(rospy.get_param("observable_task_goals"))
        spaces = features_dim + self.ports + int(min(features_dim/10, self.obs_goals))

        if self.ports > 0:
            spaces += features_dim 
        
        super(LSTM_EXTRACTOR, self).__init__(observation_space, features_dim)

        if self.obs_goals > 0:
            self.dsc_cnn = nn.Sequential(
                nn.Conv1d(1, 32, 4, 2),
                nn.ReLU(),
                nn.Conv1d(32, 16, 4, 1),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Compute shape by doing one forward pass
            with th.no_grad():
                tensor_forward = th.randn(1, 1, self.obs_goals)
                n_flatten = self.dsc_cnn(tensor_forward).shape[1]
            
            self.dsc_fc = nn.Sequential(
                nn.Linear(n_flatten, int(min(features_dim/10, self.obs_goals))),
                nn.ReLU(),
            )

        s = int(self._l/2)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, s, 16, 4),
            nn.ReLU(),
            nn.Conv1d(s, 64, 8, 2),
            nn.ReLU(),
            nn.Conv1d(64, 8, 4, 1),
            nn.ReLU(),
            # nn.Conv1d(64, 16, 4, 1),
            # nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.body_net = nn.LSTM(input_size=n_flatten + self._rs - self.obs_goals,
                                hidden_size=128,
                                num_layers=2,
                                batch_first=True)
        
        self.linear_layer = nn.Linear(128, features_dim)
        self.activation_fn = nn.ReLU()

        
        self.net = nn.Linear(spaces, features_dim)
        self.activation_fn = nn.ReLU()


        if self.ports > 0:
            s = int(self._msg/2)
            self.msg_cnn = nn.Sequential(
                nn.Conv1d(1, s, 16, 4),
                nn.ReLU(),
                nn.Conv1d(s, 128, 8, 2),
                nn.ReLU(),
                nn.Conv1d(128, 64, 4, 1),
                nn.ReLU(),
                nn.Conv1d(64, 16, 4, 1),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Compute shape by doing one forward pass
            with th.no_grad():
                tensor_forward = th.randn(1, 1, self._msg-self.ports)
                n_flatten = self.msg_cnn(tensor_forward).shape[1]

            self.msg_fc = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )
        self.spaces = spaces
        print("feature size: ", spaces)
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, self.obs_goals: self.obs_goals+self._l], 1)

        if self.obs_goals > 0:
            pub_goals = th.unsqueeze(observations[:, :self.obs_goals], 1)
            goal_features = self.dsc_fc(self.dsc_cnn(pub_goals))
        else:
            pub_goals = []
            goal_features = laser_scan.size(0)
        
        if self.ports > 0:
            robot_state = observations[:, self.obs_goals+self._l :-self._msg]
            msgs = th.unsqueeze(observations[:, self._rs+self._l:-self.ports], 1)
            msg_features = self.msg_fc(self.msg_cnn(msgs))
            ports = observations[:, -self.ports:]
        else:
            robot_state = observations[:, self.obs_goals+self._l :]
            msgs = []
            num_rows = laser_scan.size(0)
            # Create empty tensor with the same number of rows as scan
            msg_features = th.empty((num_rows, 0))
            ports = th.empty((num_rows, 0))

        extracted_features, (h_n, c_n) = self.body_net(th.cat((self.cnn(laser_scan),robot_state),1))
        extracted_features = self.linear_layer(extracted_features)
        extracted_features = self.activation_fn(extracted_features)
        
        self.feat = th.cat((goal_features, extracted_features, msg_features, ports), 1)
    
        return self.net(self.feat)


class UNIFIED_SPACE_EXTRACTOR(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
    ):
        self._l, self._rs = get_observation_space_from_file(robot_model)
        super().__init__(observation_space, features_dim)

        self.model = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        # obs = th.unsqueeze(observations, 0)

        return self.model(observations)
