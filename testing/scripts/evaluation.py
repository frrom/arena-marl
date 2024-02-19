import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from rl_utils.rl_utils.utils.utils import call_service_takeSimStep
from stable_baselines3.common.callbacks import (
    MarlEvalCallback,
    StopTrainingOnRewardThreshold,
)
from training.tools.staged_train_callback import InitiateNewTrainStage
from training.tools.train_agent_utils import create_evaluation_setup
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import random
from geometry_msgs.msg import Pose2D, PoseStamped


def evaluate_policy(
    robots: Dict[str, Dict[str, Any]],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.
    """
    obs = {robot: {} for robot in robots}

    not_reseted = True
    # Avoid double reset, as VecEnv are reset automatically.
    if not_reseted:
        # Reset the env states for all robots
        for robot in robots:
            rospy.set_param(f"eval_sim/training/{robot}/reset_mode", "reset_states")
            robots[robot]["env"].reset()
        # perform one step in the simulation to update the scene
        call_service_takeSimStep(ns="eval_sim")
        # Collect new observations after reset for all robots
        for robot in robots:
            # ('robot': {'agent1': obs, 'agent2': obs, ...})
            obs[robot] = robots[robot]["env"].reset()
            not_reseted = False

    # {'robot': [agents]} e.g. {'jackal': [agent1, agent2], 'burger': [agent1]}
    agents = {robot: robots[robot]["agent_dict"]["eval_sim"] for robot in robots}
    # {'robot': [robot_ns]} e.g. {'jackal': [robot1, robot2], 'burger': [robot1]}
    agent_names = {
        robot: [a_name._robot_sim_ns for a_name in agents[robot]] for robot in robots
    }

    # {
    #   'jackal':
    #       'robot1': [reward1, reward2, ...]
    #       'robot2': [reward1, reward2, ...]
    #   'burger':
    #       'robot1': [reward1, reward2, ...]
    #       'robot2': [reward1, reward2, ...]
    # }
    episode_rewards = {
        robot: {agent: [] for agent in agent_names[robot]} for robot in robots
    }
    episode_lengths = []

    # dones -> {'robot': {'robot_ns': False}}
    # e.g. {'jackal': {robot1: False, robot2: False}, 'burger': {'robot1': False}}
    default_dones = {robot: {a: False for a in agent_names[robot]} for robot in robots}
    default_infos = {robot: {a: {} for a in agent_names[robot]} for robot in robots}
    default_actions = {robot: {a: [] for a in agent_names[robot]} for robot in robots}
    # states, actions, episode rewards -> {'robot': {'robot_ns': None}}
    # e.g. {'jackal': {robot1: None, robot2: None}, 'burger': {'robot1': None}}
    default_episode_reward = {
        robot: {a: 0 for a in agent_names[robot]} for robot in robots
    }
    succ_eval = []
    succ_reward = []
    states = {}
    comm = {}
    starts={}
    map_path = rospy.get_param("/world_path")
    map_path = '/'.join(map_path.split('/')[:-1])+"/map.png"
    img = mpimg.imread(map_path)
    img = np.flipud(img)
    plot_traj = rospy.get_param("plot_trjectories", default = False)
    plain = rospy.get_param("crossroad", default=False)
    plot_episodes = random.sample(range(0, n_eval_episodes), min(7,n_eval_episodes))
    plot_episodes.sort()
    tt = []
    crash_ratio = np.zeros((n_eval_episodes))
    comm_ratio = np.zeros((n_eval_episodes))
    bc_ratio = np.zeros((n_eval_episodes))
    res = rospy.get_param("resolution", default = 0.01)
    offx = rospy.get_param("offx", default = 0)
    offy = rospy.get_param("offy", default = 0)
    while len(episode_lengths) < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.

        # Avoid double reset, as VecEnv are reset automatically.
        if not_reseted:
            # Reset the env states for all robots
            for robot in robots:
                rospy.set_param(f"eval_sim/training/{robot}/reset_mode", "reset_states")
                robots[robot]["env"].reset()
            # perform one step in the simulation to update the scene
            call_service_takeSimStep(ns="eval_sim")
            # Collect new observations after reset for all robots
            for robot in robots:
                # ('robot': {'agent1': obs, 'agent2': obs, ...})
                obs[robot] = robots[robot]["env"].reset()
                not_reseted = False
        for robot in robots:
            states[robot] = {}
            comm[robot] = {}
            starts[robot] = {}
            env = robots[robot]["env"]
            for agent in agent_names[robot]:
                states[robot][agent] = np.zeros((env._max_num_moves,2)) 
                comm[robot][agent] = np.zeros((2))
        dones = copy.deepcopy(default_dones)
        infos = copy.deepcopy(default_infos)
        actions = copy.deepcopy(default_actions)
        episode_reward = copy.deepcopy(default_episode_reward)
        episode_length = 0
        crashes = 0
        ts = 0
        while not check_dones(dones):
            ### Get predicted actions and publish those
            for robot in robots:
                # Get predicted actions from each agent
                concat_obs = [obs[robot][agent] for agent in agent_names[robot]]
                pred_actions = robots[robot]["model"].predict(
                    concat_obs,
                    deterministic=deterministic,
                )[0]

                for i, agent in enumerate(agent_names[robot]):
                    actions[robot][agent] = pred_actions[i]

                # Publish actions in the environment
                env = robots[robot]["env"]
                
                env.apply_action(actions[robot])

            ### Make a step in the simulation
            # This moves all agents in the simulation and transfers them into the next state
            call_service_takeSimStep(ns="eval_sim")

            ### Get new obs, rewards, dones, and infos for all robots
            for robot in robots:
                obs[robot], rewards, single_dones, single_infos = robots[robot][
                    "env"
                ].get_states()
                for agent in env.agents:
                    states[robot][agent][episode_length,:] = obs[robot][agent][env.obs_end[agent]-3:env.obs_end[agent]-1]
                    if int(actions[robot][agent][-1]) > 0:
                        comm[robot][agent][0] += 1 
                    elif int(actions[robot][agent][-2]) != 0:
                        comm[robot][agent][1] += 1
                ### Sometimes dones return no entry for an agent
                #   that was done 1 or more steps before
                for agent, done in single_dones.items():
                    dones[robot][agent] = done
                    if dones[robot][agent]:
                        #print(f'{agent} done')
                        ts += episode_length
                        states[robot][agent]=states[robot][agent][:episode_length,:]
                        bc_ratio[len(episode_lengths)] = comm[robot][agent][0]/episode_length
                        comm_ratio[len(episode_lengths)] = comm[robot][agent][1]/episode_length
                        tt.append(episode_length)

                ### Sometimes infos return no entry for an agent
                #   that was done 1 or more steps before
                for agent, info in single_infos.items():
                    
                    for key, value in info.items():
                        infos[robot][agent][key] = value

                # Add up rewards for this episode
                # for (agent, reward) in rewards.items():
                #     # Only add reward if agent is not done
                #     # TODO: Maybe last reward is not added due to if statement
                #     # TODO: Just remove if statement to add all rewards, since all rewards of done agents are 0
                #     if agent in dones[robot].keys() and not dones[robot][agent]:
                #         # episode_reward[robot][agent] += reward
                #         pass
                #     else:
                #         print(f"{agent} is done, not adding reward")

                ### Add up rewards for this episode
                #   Only add reward if agent is not done
                for agent in single_dones.keys():
                    episode_reward[robot][agent] += rewards[agent]
                    crashes += int(infos[robot][agent]["crash"])
                    if episode_length == 0:
                        starts[robot][agent] = infos[robot][agent]["robot_pose"]

                if render:
                    robots[robot]["env"].render()

            ### Document how long this episode was
            episode_length += 1
        
        # Plot robot trajectories of selected episodes
        if plot_traj:# and len(plot_episodes) != 0 and len(episode_lengths) == plot_episodes[0]:
            #j = plot_episodes.pop(0)
            #fig = plt.figure(len(episode_lengths)+1)
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray', origin='lower')
            ax.axis('off') 
            #ax = ax1.twinx()
            # if not plain:
            #     plt.plot([4, 4], [3, 8], linestyle='-', c='black',linewidth=2.5)
            #     for i in range(6):
            #         plt.plot([3, 5], [3+i, 3+i], linestyle='-', c='black',linewidth=2.5)
            col = []
            for robot in robots:              
                for agent in agent_names[robot]:
                    alphas = np.linspace(0, 1, episode_length)
                    x = (states[robot][agent][:,0]-offx)/res
                    y = (states[robot][agent][:,1]-offy)/res
                    th = starts[robot][agent].theta
                    # Plot the points with varying alpha values
                    col.append(np.random.rand(3,))
                    ax.scatter(x, y, color=col[-1], alpha=alphas[:states[robot][agent].shape[0]])

                    # Mark the starting point as square and goal point as an x 
                    ax.scatter(x[0], y[0], marker='s', color=col[-1], s=1.25/res)
                    # Create a rectangle patch with rotation of the robot
                    
                    # Create an arrow pointing in the starting dircetion of the robot
                    arrow = patches.FancyArrowPatch((x[0], y[0]), (x[0] + 0.5/res * np.cos(th),
                                    y[0] + 0.5/res * np.sin(th)),
                                    mutation_scale=15, color='black', arrowstyle='->',transform=ax.transData)
                    ax.add_patch(arrow)
                    # Create an X for the goal
                    
                    #ax.scatter(goal.x*100,goal.y*100, marker='x', color = col, s=125,linewidths=2, label=f'goal of {agent}')
                for i, agent in enumerate(agent_names[robot]):
                    goal = env.goals[agent]
                    ax.scatter((goal.x-offx)/res,(goal.y-offy)/res, marker='x', color = col[i], s=1.25/res,linewidths=2, label=f'goal of {agent}')
            # Add labels and a legend
            # ax.set_xlim([0, 800])
            # ax.set_ylim([0, 1200])
            # ax.autoscale(False)
            ax.axis('off')
            # plt.legend()
            path = rospy.get_param(f"model_path_{robot}", default= "/home/frank/MARL_ws/src/arena-marl/plots")
            #plt.savefig(f"{path[robot]}/{str(j)}.jpg")
            plt.show()

        # TODO: check if this is correct
        done_count = {}
        success_count = {}
        succ_r = {}
        done_reason_count = {robot: np.zeros(5) for robot in robots}
        crash_ratio[len(episode_lengths)] = crashes/ts
        for robot in robots:
            done_count[robot] = np.sum(
                [1 for is_done in dones[robot].values() if is_done]
            )
            succ_r[robot] = [reward for agent, reward in episode_reward[robot].items() if "is_success" in infos[robot][agent] and infos[robot][agent]["is_success"]==1]
            success_count[robot] = np.sum(
                [
                    infos[robot][agent]["is_success"]
                    for agent in infos[robot]
                    if "is_success" in infos[robot][agent]
                    # if [key for key in infos[robot][agent]].count("is_success")
                ]
            )
            succ_reward.append(np.mean(succ_r[robot]))
            succ_eval.append(np.mean([infos[robot][agent]["is_success"] for agent in infos[robot]]))
            ### Count the reasons for termination
            #   For debugging
            #   idx 0-2: done reasons count
            #   idx 3:   normalization count
            #   idx 4:   no reason count
            for agent in infos[robot]:
                if "done_reason" in infos[robot][agent]:
                    done_reason = infos[robot][agent]["done_reason"]
                    done_reason_count[robot][done_reason] += 1
                    done_reason_count[robot][3] += 1
                else:
                    done_reason_count[robot][4] += 1

        if callback is not None:
            callback(locals(), globals())

        ### Collect all episode rewards
        # For each robot append rewards for every of its agents
        # to the list of respective episode rewards
        for robot in robots:
            for agent, reward in episode_reward[robot].items():
                episode_rewards[robot][agent].append(reward)

        ### Document all episode lengths
        episode_lengths.append(episode_length)

        ### Set to reset the environment after each episode
        not_reseted = True

    ### Calculate average rewards
    mean_rewards = {
        robot: {
            agent: np.mean(episode_rewards[robot][agent])
            for agent in agent_names[robot]
        }
        for robot in robots
    }
    print("success rate: ")
    print(np.mean(succ_eval))
    

    print("crash rate: ")
    print(np.mean(crash_ratio), " +/- ", np.std(crash_ratio))
    print(np.median(crash_ratio))

    print("communication rate: ")
    print(np.mean(comm_ratio), " +/- ", np.std(comm_ratio))

    print("broadcast rate: ")
    print(np.mean(bc_ratio), " +/- ", np.std(bc_ratio))

    print("travel time: ")
    print(np.mean(tt), " +/- ", np.std(tt))
    print(np.median(tt))

    for robot in robots:
        print("mean reward: ")
        means = [mean_rewards[robot][agent] for agent in agent_names[robot]]
        med = np.median([el for agent in agent_names[robot] for el in episode_rewards[robot][agent]])
        print(np.mean(means), " +/- ", np.std(means))
        print("median: ", med)
        print("clean mean reward: ")
        print(np.mean(succ_reward), " +/- ", np.std(succ_reward))
    

    ### Calculate standard deviation of rewards
    std_rewards = {
        robot: {
            agent: np.std(episode_rewards[robot][agent]) for agent in agent_names[robot]
        }
        for robot in robots
    }

    if reward_threshold is not None:
        for robot in robots:
            for agent in agent_names[robot]:
                if mean_rewards[robot][agent] < reward_threshold:
                    raise ValueError(
                        "Mean reward for agent {} of robot {} is below threshold! {} < {}".format(
                            agent, robot, mean_rewards[robot][agent], reward_threshold
                        )
                    )

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_rewards, std_rewards


def create_eval_callback(config, train_robots, wandb_logger):
    robots = create_evaluation_setup(config, train_robots)

    return MarlEvalCallback(
        robots=robots,
        wandb_logger=wandb_logger,
        n_eval_episodes=config["periodic_eval"]["n_eval_episodes"],
        eval_freq=config["periodic_eval"]["eval_freq"],
        deterministic=True,
        log_paths={robot: val["paths"]["eval"] for robot, val in train_robots.items()},
        best_model_save_paths={
            robot: val["paths"]["model"] for robot, val in train_robots.items()
        },
        callback_on_eval_end=InitiateNewTrainStage(
            n_envs=config["n_envs"],
            treshhold_type=config["training_curriculum"]["threshold_type"],
            upper_threshold=config["training_curriculum"]["upper_threshold"],
            lower_threshold=config["training_curriculum"]["lower_threshold"],
            task_mode=config["task_mode"],
            model_paths={
                robot: val["paths"]["model"] for robot, val in train_robots.items()
            },
            verbose=1,
        ),
        callback_on_new_best=StopTrainingOnRewardThreshold(
            treshhold_type=config["stop_training"]["threshold_type"],
            threshold=config["stop_training"]["threshold"],
            robots=robots,
            verbose=1,
        ),
    )


def check_dones(dones):
    # Check if all agents for every robot are done
    for robot in dones:
        for agent in dones[robot]:
            if dones[robot][agent]:
                continue
            else:
                return False
    return True


def print_evaluation_results(mean_rewards, std_rewards):
    for robot in mean_rewards:
        print(f"{robot}:")
        for agent in mean_rewards[robot]:
            print(
                f"\t{agent}: {mean_rewards[robot][agent]} +- {std_rewards[robot][agent]}"
            )
    print("")
