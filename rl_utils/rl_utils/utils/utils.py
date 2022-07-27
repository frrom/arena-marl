import os
import sys
import time
from typing import Callable, List

import rospkg
import rospy
from flatland_msgs.srv import StepWorld, StepWorldRequest
from rl_utils.rl_utils.base_agent_wrapper import BaseDRLAgent
from rl_utils.rl_utils.training_agent_wrapper import TrainingDRLAgent


def call_service_takeSimStep(ns: str, t: float = None):
    """Fast-forwards the simulation.

        Description:
            Simulates the Flatland simulation for a certain amount of seconds.
            
        Args:
            ns (str): Name of the namespace (used for ROS topics).
            t (float, optional): Time in seconds. When ``t`` is None, time is forwarded by ``step_size`` s \
                (ROS parameter). Defaults to None.
        """

    _service_name_step = f"{ns}step_world"
    _sim_step_client = rospy.ServiceProxy(_service_name_step, StepWorld)
    request = StepWorldRequest() if t is None else StepWorldRequest(t)

    try:
        response = _sim_step_client(request)
        rospy.logdebug("step service=", response)
    except rospy.ServiceException as e:
        rospy.logdebug(f"step Service call failed: {e}")


def get_hyperparameter_file(filename: str) -> str:
    """Function which returns the path to the hyperparameter file.

    Args:
        filename (str): Name of the hyperparameter file.

    Returns:
        str: Path to the hyperparameter file.
    """
    return os.path.join(
        rospkg.RosPack().get_path("arena_local_planner_drl"),
        "configs",
        "hyperparameters",
        filename,
    )


def instantiate_train_drl_agents(
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


def instantiate_deploy_drl_agents(
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
) -> List[BaseDRLAgent]:
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
        BaseDRLAgent(
            ns=ns,
            robot_model=robot_model,
            robot_ns=robot_name_prefix + str(i + 1),
            hyperparameter_path=hyperparameter_path,
        )
        for i in range(existing_robots, existing_robots + num_robots)
    ]
