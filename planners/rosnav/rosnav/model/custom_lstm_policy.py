import os
import rospy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import rospkg
import torch as th
import yaml

from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from .feature_extractors import *


from .agent_factory import AgentFactory
from ..utils.utils import get_observation_space


__all__ = ["MLP_ARENA2D_POLICY"]


""" 
_RS: Robot state size - placeholder for robot related inputs to the NN
_L: Number of laser beams - placeholder for the laser beam data 
"""
_L, _RS = get_observation_space()


class CustomLSTM(nn.Module):
    """
    Custom Multilayer Perceptron for policy and value function.
    Architecture was taken as reference from: https://github.com/ignc-research/arena2D/tree/master/arena2d-agents.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomLSTM, self).__init__()
    
         # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Body network
        # self.body_net = nn.Sequential(
        #     nn.LSTM(input_size=feature_dim,
        #             hidden_size=128,
        #             num_layers=2,
        #             batch_first=True).squeeze(0),
        #     #nn.Linear(_L + _RS, 64),
        #     nn.Linear(128, feature_dim),
        #     nn.ReLU(),
        # )

        # Body network
        self.body_net = nn.LSTM(input_size=feature_dim,
                                hidden_size=128,
                                num_layers=2,
                                batch_first=True)
        
        self.linear_layer = nn.Linear(128, feature_dim)
        self.activation_fn = nn.ReLU()

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )



    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        body_x, (h_n, c_n) = self.body_net(features)
        #linear_x = self.linear_layer(h_n[-1])
        linear_x = self.linear_layer(body_x) #output[:, -1, :]
        linear_x = self.activation_fn(linear_x)
        return self.policy_net(linear_x), self.value_net(linear_x)
        # body_x = self.body_net(features)
        # return self.policy_net(body_x), self.value_net(body_x)


@AgentFactory.register("LSTMAgent")
class LSTM_ARENA2D_POLICY(ActorCriticPolicy):
    """
    Policy using the custom Multilayer Perceptron.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        robot_model = None,
        *args,
        **kwargs,
    ):
        super(LSTM_ARENA2D_POLICY, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        # Enable orthogonal initialization
        self.ortho_init = True


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomLSTM(self.features_dim)

    # def get_kwargs(self):
    #     features_extractor_class = EXTRACTOR_7
    #     features_extractor_kwargs = dict(features_dim=300)
    #     features_extractor_kwargs["robot_model"] = self.robot_model
    #     return dict(features_extractor_class = EXTRACTOR_7,
    #     features_extractor_kwargs = features_extractor_kwargs)
