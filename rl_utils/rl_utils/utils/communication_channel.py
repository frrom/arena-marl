import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped
import random

def calc_distance(pose1:Pose2D ,pose2:Pose2D):
    return np.linalg.norm([pose1.x -pose2.x ,pose1.y - pose2.y])

class Channel:
    def __init__(self,
                ns: str,
                agents: dict = {},
                ports: int = 2,
                signal_range: float = 5.0,
                force: bool = False
                ):
        self.signal_range = signal_range
        self.ns = ns
        self.robots = agents
        self.active_robots = list(self.robots.keys())
        self.messages = {}
        for agent in self.active_robots:
            self.messages[agent] = {}
        self.ports = ports
        self.inrange = {}
        self.obs_goals = int(rospy.get_param("/observable_task_goals"))
        self.force = rospy.get_param("force_communication", default = False)

    def add_robot(self,robot:str, mapping):
        self.robots[robot] = mapping
        self.active_robots = list(self.robots.keys())
        self.messages[robot] = {}

    def get_robots_in_range(self, robot:str):
        pos = self.robots[robot].position
        inrange = []
        #self.active_robots.shuffle()
        for key in self.active_robots:
            if calc_distance(pos, self.robots[key].position) < self.signal_range and key != robot:
                inrange.append(int(key[5:]))
        if len(inrange) > self.ports-1:
            inrange[self.ports-1] = len(inrange[self.ports-1:])
            inrange = inrange[:self.ports]
        else:
            for i in range(self.ports - len(inrange)):
                inrange.append(0)
        if len(inrange) != self.ports:
            raise RuntimeError

        
        self.inrange[robot] = inrange
        return inrange

    def send_message_request(self, source, ports):
        if source in self.inrange:
            if int(ports[-1]) > 0 and np.array(self.inrange[source]).all() != 0 and not self.force:
                #print(source + " broadcasting")
                for ins in self.inrange[source]:
                    self.messages["robot"+str(ins)]["broadcast"] = self.observations[source]
                for i in range(len(ports)-1):
                    key = "robot"+str(self.inrange[source][i]) if source in self.inrange else "robot"+str(i)
                    self.messages[source][key] = np.zeros(self.observations[source].shape)
                random.shuffle(self.active_robots)
                # l = list(self.robots.items())
                # random.shuffle(l)
                # self.robots = dict(l)
                return
        
        for i in range(len(ports)-1):
            try:
                key = "robot"+str(self.inrange[source][i])
                if int(ports[i]) > 0 or self.force:
                    self.messages[source][key] = self.observations[key]
                    #print("message from " + key + " to " + source)
                else:
                    self.messages[source][key] = np.zeros(self.observations[key].shape)   
            except:
                self.messages[source]["robot"+str(i)] = np.zeros(self.observations[source].shape)
        return


    def retrieve_messages(self, robot):
        if "broadcast" not in self.messages[robot]:
            self.messages[robot]["broadcast"] = np.zeros(self.observations[robot].size)
        messages = self.messages[robot]
        self.messages[robot] = {}
        if len(messages.keys()) != self.ports:
            #print("keys not matching")
            raise IndexError
        return messages


    def step(self, observations, actions):
        self.observations = observations.copy()
        if len(actions.keys()) != len(self.active_robots):
            self.active_robots = list(actions.keys())
            # for key in actions:
            #     self.active_robots[key] = self.robots[key]
        for key in actions:
            ac = actions[key]
            self.send_message_request(key, ac[-self.ports:])

    def reset(self):
        #self.observations = {}
        self.active_robots = list(self.robots.keys())
        for agent in self.active_robots:
            self.messages[agent] = {}
        self.inrange = {}
        

