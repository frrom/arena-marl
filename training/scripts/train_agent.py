import sys
import os, sys, rospy, time

from datetime import time
from typing import Callable, List

import rospy
import rospkg
from multiprocessing import cpu_count, set_start_method

from rosnav.model.agent_factory import (
    AgentFactory,
)
from rosnav.model.base_agent import (
    BaseAgent,
)
from training.tools.custom_mlp_utils import *
from rosnav.model.custom_policy import *
from rosnav.model.custom_sb3_policy import *

# from tools.argsparser import parse_training_args
from training.tools.staged_train_callback import InitiateNewTrainStage
from training.tools.argsparser import parse_training_args

from rl_utils.rl_utils.envs.pettingzoo_env import env_fn
from rl_utils.rl_utils.utils.supersuit_utils import (
    vec_env_create,
)

# from tools.argsparser import parse_marl_training_args
from training.tools.train_agent_utils import (
    get_MARL_agent_name_and_start_time,
    get_paths,
    choose_agent_model,
    load_config,
    initialize_hyperparameters,
)
from training.tools import train_agent_utils
from stable_baselines3.common.callbacks import (
    StopTrainingOnRewardThreshold,
    MarlEvalCallback,
)

from training.tools.train_agent_utils import *

from rl_utils.rl_utils.training_agent_wrapper import TrainingDRLAgent
from task_generator.tasks import get_MARL_task, init_obstacle_manager, get_task_manager
from task_generator.robot_manager import init_robot_managers


def instantiate_drl_agents(
    num_robots: int = 1,
    existing_robots: int = 0,
    robot_model: str = "burger",
    ns: str = None,
    robot_name_prefix: str = "robot",
    hyperparameter_path: str = os.path.join(
        rospkg.RosPack().get_path("training"),
        "configs",
        "hyperparameters",
        "default.json",
    ),
) -> List[TrainingDRLAgent]:
    """Function which generates a list agents which handle the ROS connection.

    Args:
        num_robots (int, optional): Number of robots in the environment. Defaults to 1.
        robot_model (str, optional): Model of the robot. Defaults to "burger".
        ns (str, optional): Name of the namespace (used for ROS topics). Defaults to None.
        robot_name_prefix (str, optional): Name with which to prefix robots in the ROS environment. Defaults to "robot".
        hyperparameter_path (str, optional): Path where to load hyperparameters from. Defaults to DEFAULT_HYPERPARAMETER.
        action_space_path (str, optional): Path where to load action spaces from. Defaults to DEFAULT_ACTION_SPACE.

    Returns:
        List[TrainingDRLAgent]: List containing _num\_robots_ agent classes.
    """
    return [
        TrainingDRLAgent(
            ns=ns,
            robot_model=robot_model,
            robot_ns=robot_name_prefix + str(i + 1),
            hyperparameter_path=hyperparameter_path,
        )
        for i in range(existing_robots, existing_robots + num_robots)
    ]


def main(args):
    # load configuration
    config = load_config(args.config)

    # set debug_mode
    rospy.set_param("debug_mode", config["debug_mode"])

    robots, existing_robots = {}, 0
    MARL_NAME, START_TIME = get_MARL_agent_name_and_start_time()

    task_managers = {
        f"sim_{i}": get_task_manager(
            ns=f"sim_{i}",
            mode=config["task_mode"],
            curriculum_path=config["training_curriculum"]["training_curriculum_file"],
        )
        for i in range(1, config["n_envs"] + 1)
    }
    obstacle_manager_dict = init_obstacle_manager(n_envs=config["n_envs"])

    # create seperate model instances for each robot
    for robot_name, robot_train_params in config["robots"].items():
        # generate agent name and model specific paths
        agent_name = (
            robot_train_params["resume"].split("/")[-1]
            if robot_train_params["resume"]
            else f"{robot_name}_{START_TIME}"
        )

        paths = train_agent_utils.get_paths(
            MARL_NAME,
            robot_name,
            agent_name,
            robot_train_params,
            config["training_curriculum"]["training_curriculum_file"],
            config["eval_log"],
            config["tb"],
        )

        # initialize hyperparameters (save to/ load from json)
        hyper_params = train_agent_utils.initialize_hyperparameters(
            PATHS=paths,
            config=robot_train_params,
            n_envs=config["n_envs"],
        )

        # create agent wrapper dict for specific robot
        # each entry contains list of agents for specfic namespace
        # e.g. agent_list["sim_1"] -> [TrainingDRLAgent]
        agent_dict = {
            f"sim_{i}": instantiate_drl_agents(
                num_robots=robot_train_params["num_robots"],
                existing_robots=existing_robots,
                robot_model=robot_name,
                hyperparameter_path=paths["hyperparams"],
                ns=f"sim_{i}",
            )
            for i in range(1, config["n_envs"] + 1)
        }

        robot_manager_dict = init_robot_managers(
            n_envs=config["n_envs"], robot_type=robot_name, agent_dict=agent_dict
        )

        for i in range(1, config["n_envs"] + 1):
            task_managers[f"sim_{i}"].set_obstacle_manager(
                obstacle_manager_dict[f"sim_{i}"]
            )
            task_managers[f"sim_{i}"].add_robot_manager(
                robot_type=robot_name, managers=robot_manager_dict[f"sim_{i}"]
            )

        env = vec_env_create(
            env_fn,
            agent_dict,
            task_managers=task_managers,
            num_cpus=cpu_count() - 1,
            num_vec_envs=config["n_envs"],
            max_num_moves_per_eps=config["max_num_moves_per_eps"],
        )

        existing_robots += robot_train_params["num_robots"]

        # env = VecNormalize(
        #     env,
        #     training=True,
        #     norm_obs=True,
        #     norm_reward=True,
        #     clip_reward=15,
        #     clip_obs=15,
        # )

        model = choose_agent_model(
            agent_name, paths, robot_train_params, env, hyper_params, config["n_envs"]
        )

        # add configuration for one robot to robots dictionary
        robots[robot_name] = {
            "model": model,
            "env": env,
            "n_envs": config["n_envs"],
            "robot_train_params": robot_train_params,
            "hyper_params": hyper_params,
            "paths": paths,
        }

    # set num of timesteps to be generated
    n_timesteps = 40000000 if config["n_timesteps"] is None else config["n_timesteps"]

    start = time.time()
    try:
        model = robots[robot_name]["model"]
        model.learn(
            total_timesteps=n_timesteps,
            reset_num_timesteps=True,
            # Übergib einfach das dict für den aktuellen roboter
            # callback=get_evalcallback(
            #     eval_config=config["periodic_eval"],
            #     curriculum_config=config["training_curriculum"],
            #     stop_training_config=config["stop_training"],
            #     train_env=robots[robot_name]["env"],
            #     num_robots=robots[robot_name]["robot_train_params"]["num_robots"],
            #     num_envs=config["n_envs"],
            #     task_mode=config["task_mode"],
            #     PATHS=robots[robot_name]["paths"],
            # ),
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt..")
    # finally:
    # update the timesteps the model has trained in total
    # update_total_timesteps_json(n_timesteps, PATHS)

    robots[robot_name][model].env.close()
    print(f"Time passed: {time.time() - start}s")
    print("Training script will be terminated")
    sys.exit()


def get_evalcallback(
    eval_config: dict,
    curriculum_config: dict,
    stop_training_config: dict,
    train_env: VecEnv,
    num_robots: int,
    num_envs: int,
    task_mode: str,
    PATHS: dict,
) -> MarlEvalCallback:
    """Function which generates an evaluation callback with an evaluation environment.

    Args:
        train_env (VecEnv): Vectorized training environment
        num_robots (int): Number of robots in the environment
        num_envs (int): Number of parallel spawned environments
        task_mode (str): Task mode for the current experiment
        PATHS (dict): Dictionary which holds hyperparameters for the experiment

    Returns:
        MarlEvalCallback: [description]
    """
    eval_env = env_fn(
        num_agents=num_robots,
        ns="eval_sim",
        agent_list_fn=instantiate_drl_agents,
        max_num_moves_per_eps=eval_config["max_num_moves_per_eps"],
        PATHS=PATHS,
    )

    # eval_env = VecNormalize(
    #     eval_env,
    #     training=False,
    #     norm_obs=True,
    #     norm_reward=False,
    #     clip_reward=15,
    #     clip_obs=3.5,
    # )

    return MarlEvalCallback(
        train_env=train_env,
        eval_env=eval_env,
        num_robots=num_robots,
        n_eval_episodes=eval_config["n_eval_episodes"],
        eval_freq=eval_config["eval_freq"],
        deterministic=True,
        log_path=PATHS["eval"],
        best_model_save_path=PATHS["model"],
        callback_on_eval_end=InitiateNewTrainStage(
            n_envs=num_envs,
            treshhold_type=curriculum_config["threshold_type"],
            upper_threshold=curriculum_config["upper_threshold"],
            lower_threshold=curriculum_config["lower_threshold"],
            task_mode=task_mode,
            verbose=1,
        ),
        callback_on_new_best=StopTrainingOnRewardThreshold(
            treshhold_type=stop_training_config["threshold_type"],
            threshold=stop_training_config["threshold"],
            verbose=1,
        ),
    )


if __name__ == "__main__":
    set_start_method("fork")
    # args, _ = parse_marl_training_args()
    args, _ = parse_training_args()
    # rospy.init_node("train_env", disable_signals=False, anonymous=True)
    main(args)
