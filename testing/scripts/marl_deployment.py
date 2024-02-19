import rospy
from testing.scripts.evaluation import evaluate_policy, print_evaluation_results
from training.tools.argsparser import parse_training_args
from training.tools.train_agent_utils import load_config, create_deployment_setup


def main(args):
    rospy.init_node("marl_deployment")

    # load configuration
    config = load_config(args.config)
    # do NOT change these parameters
    rospy.set_param("debug_mode", config["debug_mode"])
    rospy.set_param("n_moves", config["max_num_moves_per_eps"])
    rospy.set_param("warehouse", True)
    rospy.set_param("choose_goal", True)
    
    #editable parameters
    rospy.set_param("plot_trjectories", False)
    rospy.set_param("slinding_window", False)
    rospy.set_param("window_length", 2)
    rospy.set_param("/curr_stage", 5)
    rospy.set_param("num_ports", 2)
    rospy.set_param("observable_task_goals", 5)
    rospy.set_param("force_communication", False)
    # create dicts for all robot types with all necessary parameters
    robots = create_deployment_setup(config)

    # Evaluate the policy for each robot
    mean_rewards, std_rewards = evaluate_policy(
        robots=robots,
        n_eval_episodes=config["periodic_eval"]["n_eval_episodes"],
        return_episode_rewards=False,
    )

    print_evaluation_results(mean_rewards, std_rewards)


if __name__ == "__main__":
    args, _ = parse_training_args()
    main(args)
