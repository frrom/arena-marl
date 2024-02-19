import os
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
import rospy
from std_msgs.msg import Int8
import torch as th
from rl_utils.rl_utils.envs.pettingzoo_env import FlatlandPettingZooEnv
from rl_utils.rl_utils.utils.utils import call_service_takeSimStep
from rl_utils.rl_utils.utils.wandb_helper import WandbLogger
from stable_baselines3.common import utils

# from stable_baselines3.common import logger
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.ppo.ppo import PPO
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
import random

class helper_buffer():
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        num_robots = 4,
    ):

        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = observation_space.shape
        self.action_dim = action_space.shape[0]
        self.pos = 0
        self.full = False
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.num_robots = num_robots
        self.reset()
    
    def _reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        #self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.infos = np.array([[None for _ in range(self.n_envs)] for _ in range(self.buffer_size)])
        #self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        
    def reset(self) -> None:
        self._reset()
        self.g_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.g_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.g_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.g_dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.g_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.g_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.g_infos = np.array([[None for _ in range(self.n_envs)] for _ in range(self.buffer_size)])
        self.pos2 = 0
        self.full = False
    def add(
        self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, value: th.Tensor, log_prob: th.Tensor, infos
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param done: End of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob #log_prob.clone().cpu().numpy()
        self.infos[self.pos] = infos
        self.pos += 1
        if self.pos == self.buffer_size and not self.full:
            indices = [i for i, element in enumerate(infos) if 'is_success' not in element.keys()]
            hook = -1
            for i in indices:
                if i//self.num_robots > hook:
                    hook = i//self.num_robots*self.num_robots
                    self.g_observations[:,self.pos2:self.pos2+self.num_robots,:] = self.observations[:,hook:hook+self.num_robots,:]
                    self.g_actions[:,self.pos2:self.pos2+self.num_robots,:] = self.actions[:,hook:hook+self.num_robots,:]
                    self.g_rewards[:,self.pos2:self.pos2+self.num_robots] = self.rewards[:,hook:hook+self.num_robots]
                    self.g_dones[:,self.pos2:self.pos2+self.num_robots] = self.dones[:,hook:hook+self.num_robots]
                    self.g_values[:,self.pos2:self.pos2+self.num_robots] = self.values[:,hook:hook+self.num_robots]
                    self.g_log_probs[:,self.pos2:self.pos2+self.num_robots] = self.log_probs[:,hook:hook+self.num_robots]
                    self.g_infos[:,self.pos2:self.pos2+self.num_robots] = self.infos[:,hook:hook+self.num_robots]
                    self.pos2 += self.num_robots
                if self.pos2 >= self.n_envs:
                    self.full = True
                    break
            if self.pos2 < self.n_envs:
                self._reset()
        return self.full 
    def get(self):
        return self.g_observations, self.g_actions, self.g_rewards, self.g_dones, self.g_values, self.g_log_probs, self.g_infos

class Heterogenous_PPO(object):
    def __init__(
        self,
        agent_ppo_dict: Dict[str, PPO],
        agent_env_dict: Dict[str, SB3VecEnvWrapper],
        agent_param_dict: Dict[str, Dict[str, Any]],
        n_envs: int,
        wandb_logger: WandbLogger,
        model_save_path_dict: Dict[str, str] = None,
        verbose: bool = True,
    ) -> None:
        self.agent_ppo_dict = agent_ppo_dict
        self.agent_env_dict = agent_env_dict
        self.agent_param_dict = agent_param_dict

        self.model_save_path_dict = model_save_path_dict or {
            agent: None for agent in agent_ppo_dict
        }

        self.wandb_logger = wandb_logger

        self.n_envs, self.ns_prefix = n_envs, "sim_"
        self.num_timesteps = 0
        self.verbose = verbose
        self.collect_good_runs = False
        

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[FlatlandPettingZooEnv],
        callback=None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self.start_time = time.time()
        self.stage_info = rospy.get_param("/curr_stage", default = 1)
        self._stage_info = rospy.Subscriber("stage_info", Int8, self.callback_stage_info)
        ### First reset all environment states
        
        for agent, ppo in self.agent_ppo_dict.items():
            # ppo.device = "cpu"
            # ppo.policy.cpu()
            
            ppo.n_envs = self.agent_env_dict[agent].num_envs
            
            factor = self.agent_param_dict[agent]["batch_size"] // ppo.n_envs // self.agent_param_dict[agent]["train_max_steps_per_episode"]
            ppo.n_steps = int(self.agent_param_dict[agent]["train_max_steps_per_episode"] * max(1,factor))
            if self.stage_info >= 6:
                ppo.n_steps = int(self.agent_param_dict[agent]["train_max_steps_per_episode"]*(self.stage_info-5))
            # reinitialize the rollout buffer when ppo was loaded and does not
            # have the appropriate shape
            #update_helper_buffer(int(self.agent_param_dict[agent]["train_max_steps_per_episode"])
            self._update_rollout_buffer(agent, ppo)

            if ppo.ep_info_buffer is None or reset_num_timesteps:
                ppo.ep_info_buffer = deque(maxlen=100)
                ppo.ep_success_buffer = deque(maxlen=100)

            if ppo.action_noise is not None:
                ppo.action_noise.reset()

            if reset_num_timesteps:
                ppo.num_timesteps = 0
                ppo._episode_num = 0
            else:
                # Make sure training timesteps are ahead of the internal counter
                total_timesteps += ppo.num_timesteps

            self._total_timesteps = total_timesteps

            # Avoid resetting the environment when calling ``.learn()`` consecutive times
            if reset_num_timesteps or ppo._last_obs is None:
                ### reset states
                ppo.env.reset()

        ### perform one step in each simulation to update the scene
        for i in range(1, self.n_envs + 1):
            call_service_takeSimStep(ns=self.ns_prefix + str(i))

        ### Now reset all environments to get respective last observations
        for _, ppo in self.agent_ppo_dict.items():
            ### get new observations
            ppo._last_obs = ppo.env.reset()
            ppo._last_dones = np.zeros((ppo.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if ppo._vec_normalize_env is not None:
                ppo._last_original_obs = ppo._vec_normalize_env.get_original_obs()

        if eval_env is not None and ppo.seed is not None:
            eval_env.seed(ppo.seed)

        # Configure logger's outputs
        # utils.configure_logger(
        #     self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps
        # )

        # Create directories for best models and logs
        self._init_callback(callback)

        return total_timesteps, None

    def callback_stage_info(self,msg_stage_info):
        # if msg_stage_info.data <= self.stage_info:
        #     self.stage_info = 1
        # else:
        
        if self.stage_info != msg_stage_info.data:
            self.wandb_logger.log_single(
                "time/training_stage", self.stage_info, step=self.num_timesteps-1
            )
            self.wandb_logger.log_single(
                "time/training_stage", msg_stage_info.data, step=self.num_timesteps
            )
            self.stage_info = msg_stage_info.data
            print("updating rollout-buffer size")
            for agent, ppo in self.agent_ppo_dict.items():
                if self.stage_info >= 6:
                    ppo.n_steps = int(self.agent_param_dict[agent]["train_max_steps_per_episode"]*(self.stage_info-5))
                else:
                    factor = self.agent_param_dict[agent]["batch_size"] // ppo.n_envs // self.agent_param_dict[agent]["train_max_steps_per_episode"]
                    ppo.n_steps = int(self.agent_param_dict[agent]["train_max_steps_per_episode"] * max(1,factor))
                self._update_rollout_buffer(agent, ppo)
                ppo.rollout_buffer.reset()
        self.collect_good_runs = False
        #self.collect_good_runs = True if self.stage_info < 2 else False

    def collect_rollouts(
        self,
        callback: BaseCallback,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert all(
            list(map(lambda x: x._last_obs is not None, self.agent_ppo_dict.values()))
        ), "No previous observation was provided"

        n_steps = 0
        rollout_buffers = {
            agent: ppo.rollout_buffer for agent, ppo in self.agent_ppo_dict.items()
        }

        Heterogenous_PPO._reset_all_rollout_buffers(rollout_buffers)

        for agent, ppo in self.agent_ppo_dict.items():
            if ppo.use_sde:
                ppo.policy.reset_noise(self.agent_env_dict[agent].num_envs)

        if callback:
            callback.on_rollout_start()

        complete_collection_dict = {
            agent: False for agent in self.agent_ppo_dict.keys()
        }
        timesteps = 0
        agent_actions_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_dones_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_values_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_log_probs_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_new_obs_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        helper_buffer_dict = {agent: helper_buffer(
            int(self.agent_param_dict[agent]["train_max_steps_per_episode"]/2),
            ppo.observation_space,
            ppo.action_space,
            ppo.device,
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            n_envs=self.agent_env_dict[agent].num_envs,
        ) for agent, ppo in self.agent_ppo_dict.items()}
        # only end loop when all replay buffers are filled
        while not all(complete_collection_dict.values()):
            for agent, ppo in self.agent_ppo_dict.items():
                st = ppo.n_steps/2 if self.stage_info < 6 else ppo.n_steps
                #ppo.ent_coef = 0.02 - 0.0199*timesteps/st
                if (
                    ppo.use_sde
                    and ppo.sde_sample_freq > 0
                    and n_steps % ppo.sde_sample_freq == 0
                ):
                    ppo.policy.reset_noise(self.agent_env_dict[agent].num_envs)

                actions = Heterogenous_PPO.infer_action(
                    agent,
                    ppo,
                    agent_actions_dict,
                    agent_values_dict,
                    agent_log_probs_dict,
                )

                # Env step for all robots
                self.agent_env_dict[agent].step(actions)

            for i in range(1, self.n_envs + 1):
                call_service_takeSimStep(ns=self.ns_prefix + str(i))

            
            for agent, ppo in self.agent_ppo_dict.items():
                (
                    agent_new_obs_dict[agent],
                    rewards,
                    agent_dones_dict[agent],
                    infos,
                ) = self.agent_env_dict[agent].step(
                    actions
                )  # Apply dummy action

                # only continue memorizing experiences if buffer is not full
                # and if at least one robot is still alive
                if not complete_collection_dict[agent] and not Heterogenous_PPO.check_robot_model_done(
                        agent_dones_dict[agent]):
                    
                    horizon = 1
                    if self.collect_good_runs:
                        horizon = 0
                        done = helper_buffer_dict[agent].add(ppo._last_obs,
                            agent_actions_dict[agent],
                            rewards,
                            ppo._last_dones,
                            agent_values_dict[agent],
                            agent_log_probs_dict[agent],infos)
                        if done:
                            obs, act, re, do, val, log, info = helper_buffer_dict[agent].get()
                            helper_buffer_dict[agent].reset()
                            horizon = int(self.agent_param_dict[agent]["train_max_steps_per_episode"]/2)
                    
                    for i in range(horizon):
                        if self.collect_good_runs:
                            o, a, r, d, v, l, infos = obs[i,:,:], act[i,:,:], re[i,:], do[i,:], th.tensor(val[i,:]), th.tensor(log[i,:]), info[i,:]
                            ld = do[i-1,:] if i > 0 else d
                        else:
                            o,a,r,d,v,l,ld = ppo._last_obs, agent_actions_dict[agent], rewards, ppo._last_dones, agent_values_dict[agent], agent_log_probs_dict[agent],agent_dones_dict[agent]
                        
                        ppo.num_timesteps += self.agent_env_dict[agent].num_envs
                        ppo._update_info_buffer(infos)

                        if isinstance(ppo.action_space, gym.spaces.Discrete):
                            # Reshape in case of discrete action
                            actions = actions.reshape(-1, 1)
                        try:
                            rollout_buffers[agent].add(o, a, r, d, v, l)
                            # rollout_buffers[agent].add(
                            #     ppo._last_obs,
                            #     agent_actions_dict[agent],
                            #     rewards,
                            #     ppo._last_dones,
                            #     agent_values_dict[agent],
                            #     agent_log_probs_dict[agent],)
                            n_steps += 1
                        except:
                            if self.collect_good_runs:
                                break
                            else:
                                print("rolloutbuffer overflow")
                                rollout_buffers = {
                                    agent: ppo.rollout_buffer for agent, ppo in self.agent_ppo_dict.items()
                                    }
                                Heterogenous_PPO._reset_all_rollout_buffers(rollout_buffers)
                                print(ppo.rollout_buffer.buffer_size)
                ppo._last_obs = agent_new_obs_dict[agent]
                ppo._last_dones = agent_dones_dict[agent]
                if timesteps == st//2:
                    ppo.ent_coef = self.agent_param_dict[agent]["ent_coef"] + random.choice([-1,0])*0.005
                    # # Re-compile the policy to apply the changes
                    # ppo.policy._build(lr_schedule=lambda x: 0.001)
            

            timesteps += 1
            # Give access to local variables
            if callback:
                callback.update_locals(locals())
                # Stop training if all model performances (mean rewards) are higher than threshold
                for i in range(horizon):
                    if callback.on_step() is False:
                        return False

            if Heterogenous_PPO.check_for_reset(agent_dones_dict, rollout_buffers):
                print("success rate: " + str(np.sum(rollout_buffers[agent].dones[rollout_buffers[agent].pos-1,:])/np.size(rollout_buffers[agent].dones[rollout_buffers[agent].pos-1,:])))
                self.reset_all_envs()
                timesteps = 0

            

            self.check_for_complete_collection(complete_collection_dict, n_steps)

        ### Print size of rollout buffer
        #   For debugging purposes - print the last size of the rollout buffer that is skipped before
        # if n_steps % 100 == 0:
        #     print(
        #         "Size of rollout buffer for agent {}: {}".format(
        #             agent, rollout_buffers[agent].pos
        #         )
        #     )

        for agent, ppo in self.agent_ppo_dict.items():
            with th.no_grad():
                # Compute value for the last timestep
                obs_tensor = th.as_tensor(agent_new_obs_dict[agent]).to(ppo.device)
                _, agent_values_dict[agent], _ = ppo.policy.forward(obs_tensor)

        for agent in self.agent_ppo_dict.keys():
            rollout_buffers[agent].compute_returns_and_advantage(
                last_values=agent_values_dict[agent], dones=agent_dones_dict[agent]
            )

        ### Eval and save
        #   Perform evaluation phase and save best model for each robot, if it has improved
        if callback:
            return callback.on_rollout_end(), n_steps

        return True, n_steps

    def train(self):
        for agent, ppo in self.agent_ppo_dict.items():
            print(f"[{agent}] Start Training Procedure")
            ppo.train(agent)

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
        eval_env=None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path=None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, _ = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        # if callback:
        # callback.on_training_start(locals(), globals())

        avg_n_robots = np.mean([envs.num_envs for envs in self.agent_env_dict.values()])

        while self.num_timesteps < total_timesteps:

            start_time = time.time()
            continue_training, n_steps = self.collect_rollouts(callback)

            if continue_training is False:
                break

            self.num_timesteps += n_steps * avg_n_robots

            iteration += 1

            # TODO: WandB LOGGING
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                duration = time.time() - start_time
                fps = int(n_steps * avg_n_robots / duration)
                if self.wandb_logger:
                    self.wandb_logger.log_single(
                        "time/fps", fps, step=self.num_timesteps
                    )
                    self.wandb_logger.log_single(
                        "time/total_timesteps", self.num_timesteps, step=self.num_timesteps
                    )
                    self.wandb_logger.log_single(
                        "time/training_stage", int(rospy.get_param("/curr_stage", default = 1)), step=self.num_timesteps
                    )
                    
                print("---------------------------------------")
                print(
                    "Iteration: {}\tTimesteps: {}\tFPS: {}".format(
                        iteration, self.num_timesteps, fps
                    )
                )
                print("---------------------------------------")
                # logger.record("time/iterations", iteration, exclude="tensorboard")
                # if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                #     logger.record(
                #         "rollout/ep_rew_mean",
                #         safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                #     )
                #     logger.record(
                #         "rollout/ep_len_mean",
                #         safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                #     )
                # logger.record("time/fps", fps)
                # logger.record(
                #     "time/time_elapsed",
                #     int(time.time() - self.start_time),
                #     exclude="tensorboard",
                # )
                # logger.record(
                #     "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
                # )
                # logger.dump(step=self.num_timesteps)

            self.train()

        self.save_models()

        if callback:
            callback.on_training_end()

        return self

    def _init_callback(self, callback):
        callback._init_callback()

    @staticmethod
    def infer_action(
        agent: str,
        ppo: PPO,
        agent_actions_dict: dict,
        agent_values_dict: dict,
        agent_log_probs_dict: dict,
    ) -> np.ndarray:
        with th.no_grad():
            # Convert to pytorch tensor
            obs_tensor = th.as_tensor(ppo._last_obs).to(ppo.device)
            (
                actions,
                agent_values_dict[agent],
                agent_log_probs_dict[agent],
            ) = ppo.policy.forward(obs_tensor)

        agent_actions_dict[agent] = actions.cpu().numpy()
        # Rescale and perform action
        clipped_actions = agent_actions_dict[agent]
        # Clip the actions to avoid out of bound error
        if isinstance(ppo.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                agent_actions_dict[agent],
                ppo.action_space.low,
                ppo.action_space.high,
            )
        return clipped_actions

    @staticmethod
    def check_for_reset(
        dones_dict: dict, rollout_buffer_dict: Dict[str, RolloutBuffer]
    ) -> bool:
        """Check for each robot type if either the agents are done or the buffer is full. If one is true for each robot type prepare for reset.

        Args:
            dones_dict (dict): dictionary with all done values.
            rollout_buffer_dict (Dict[str, RolloutBuffer]): dictionary with the respective rolloutbuffers.

        Returns:
            bool: _description_
        """
        buffers_full = [
            rollout_buffer.full for rollout_buffer in rollout_buffer_dict.values()
        ]
        dones = list(map(lambda x: np.all(x == 1), dones_dict.values()))
        check = [_a or _b for _a, _b in zip(buffers_full, dones)]
        return all(check)

    @staticmethod
    def check_robot_model_done(dones_array: np.ndarray) -> bool:
        return all(list(map(lambda x: np.all(x == 1), dones_array)))

    def check_for_complete_collection(
        self, completion_dict: dict, curr_steps_count: int
    ) -> bool:
        # in hyperparameter file: n_steps = n_envs / batch_size
        # RolloutBuffer size = n_steps * (n_envs * n_robots)
        for agent, ppo in self.agent_ppo_dict.items():
            # if ppo.n_steps <= curr_steps_count:
            if ppo.rollout_buffer.full and not completion_dict[agent]:
                completion_dict[agent] = True

    @staticmethod
    def _reset_all_rollout_buffers(
        rollout_buffer_dict: Dict[str, RolloutBuffer]
    ) -> None:
        for buffer in rollout_buffer_dict.values():
            buffer.reset()

    def reset_all_envs(self) -> None:
        st = 0
        while st < 5:
            #try:
            for agent, env in self.agent_env_dict.items():
                # reset states
                print("reset " + agent)
                env.reset()
                # retrieve new simulation state
                self.agent_ppo_dict[agent]._last_obs = env.reset()
            for agent, ppo in self.agent_ppo_dict.items():
                ppo.ent_coef = self.agent_param_dict[agent]["ent_coef"] + random.choice([-1,1,0])*0.005
            #     ppo.policy._build(lr_schedule=lambda x: 0.001)
            # perform one step in each simulation to update the scene
            for i in range(1, self.n_envs + 1):
                call_service_takeSimStep(ns=self.ns_prefix + str(i))
            break
            # except:
            #     print("reset failed")
            #     st += 1
        if st >= 5:
            raise RuntimeError

    def _update_current_progress_remaining(
        self, num_timesteps: int, total_timesteps: int
    ) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(
            total_timesteps
        )

    def save_models(self) -> None:
        if not rospy.get_param("debug_mode", False):
            for agent, ppo in self.agent_ppo_dict.items():
                if self.model_save_path_dict[agent]:
                    ppo.save(
                        os.path.join(self.model_save_path_dict[agent], "best_model")
                    )

    def _update_rollout_buffer(self, agent: str, ppo: PPO) -> None:
        ppo.rollout_buffer = RolloutBuffer(
            ppo.n_steps,
            ppo.observation_space,
            ppo.action_space,
            ppo.device,
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            n_envs=self.agent_env_dict[agent].num_envs,
        )
