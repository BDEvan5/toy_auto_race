from os import name
import numpy as np 
import csv
from matplotlib import pyplot as plt

from toy_auto_race.TD3 import TD3
from toy_auto_race.Utils import LibFunctions as lib
from toy_auto_race.Utils.HistoryStructs import TrainHistory
from toy_auto_race.NavAgents.PurePursuit import PurePursuit
from toy_auto_race.lidar_viz import LidarViz, LidarVizMod



class BaseMod(PurePursuit):
    def __init__(self, agent_name, map_name, sim_conf, pp_conf) -> None:
        super().__init__(map_name, sim_conf, pp_conf)
        self.name = agent_name
        self.n_beams = sim_conf.n_beams
        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer

        # TODO: move to agent history class
        self.mod_history = []
        self.pp_history = []
        self.reward_history = []
        self.critic_history = []

    def transform_obs(self, obs, pp_action):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env
            pp_action: [steer, speed] from pure pursuit controller

        Returns:
            nn_obs: observation vector for neural network
        """
        cur_v = [obs[3]/self.max_v]
        cur_d = [obs[4]/self.max_steer]
        vr_scale = [(pp_action[1])/self.max_v]
        dr_scale = [pp_action[0]/self.max_steer]

        scan = obs[7:-1]

        nn_obs = np.concatenate([cur_v, cur_d, vr_scale, dr_scale, scan])

        return nn_obs

    def modify_references(self, nn_action, d_ref):
        """
        Modifies the reference quantities for the steering.
        Mutliplies the nn_action with the max steering and then sums with the reference

        Args:
            nn_action: action from neural network in range [-1, 1]
            d_ref: steering reference from PP

        Returns:
            d_new: modified steering reference
        """
        d_max = self.max_steer
        d_phi = d_max * nn_action[0] # rad
        d_new = d_ref + d_phi

        return d_new

    def add_history_step(self, nn, pp):
        self.mod_history.append(nn)
        self.pp_history.append(pp)

    def show_vehicle_history(self, wait=False):
        plt.figure(3)
        plt.clf()

        plt.title("Mod Planner Steering actions")
        mod_history = np.array(self.mod_history)
        pp_history = np.array(self.pp_history)

        plt.plot(mod_history)
        plt.plot(pp_history)
        plt.plot(mod_history + pp_history, linewidth=3)
        plt.legend(["NN", "PP", "Action"])

        plt.pause(0.0001)

        plt.figure(5)
        plt.clf()
        plt.title("Critic history Mod planner")
        plt.plot(self.critic_history)

        self.mod_history.clear()
        self.pp_history.clear()
        self.critic_history.clear()

        if wait:
            plt.show()


class ModVehicleTrain(BaseMod):
    def __init__(self, agent_name, map_name, sim_conf, mod_conf=None, load=False, h_size=200):
        """
        Training vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
            load: if the network should be loaded or recreated.
        """
        if mod_conf is None:
            mod_conf = lib.load_conf("mod_conf")

        BaseMod.__init__(self, agent_name, map_name, sim_conf, mod_conf)

        self.path = 'Vehicles/' + agent_name
        state_space = 4 + self.n_beams
        self.agent = TD3(state_space, 1, 1, agent_name)
        # h_size = mod_conf.h
        h_size = h_size
        self.agent.try_load(load, h_size, self.path)

        self.reward_fcn = None
        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None
        self.beta_slope = None

        self.t_his = TrainHistory(agent_name, load)

    def set_reward_fcn(self, r_fcn):
        self.reward_fcn = r_fcn

    def plan_act(self, obs):
        pp_action = super().act(obs)
        nn_obs = self.transform_obs(obs, pp_action)
        self.add_memory_entry(obs, nn_obs)

        self.state = obs
        nn_action = self.agent.act(nn_obs)
        # nn_action = [0]
        self.nn_act = nn_action

        #TODO: move to history method
        self.add_history_step(pp_action[0], nn_action[0]*self.max_steer)
        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.nn_state = nn_obs

        steering_angle = self.modify_references(self.nn_act, pp_action[0])
        speed = 4
        # self.action = np.array([steering_angle, pp_action[1]])
        self.action = np.array([steering_angle, speed])

        return self.action

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.state is not None:
            # reward = self.reward_fcn(self.state, self.action, s_prime, self.nn_act)
            # reward = self.deviation_reward()
            reward = self.slope_reward()

            self.t_his.add_step_data(reward)
            mem_entry = (self.nn_state, self.nn_act, nn_s_prime, reward, False)

            self.agent.replay_buffer.add(mem_entry)

    def deviation_reward(self):
        beta = 0.002
        reward = - abs(self.nn_act[0]) * beta

        return reward
 
        
    def slope_reward(self):
        reward = self.beta_slope * (1- abs(self.nn_act[0])) 

        return reward

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        pp_action = super().act(s_prime)
        nn_s_prime = self.transform_obs(s_prime, pp_action)
        # reward = self.reward_fcn(self.state, self.action, s_prime, self.nn_act)
        reward = s_prime[-1] + self.deviation_reward()

        self.t_his.add_step_data(reward)
        self.t_his.lap_done(False)
        # self.t_his.lap_done(True)
        if len(self.t_his.ep_rewards) % 10 == 0:
            self.t_his.print_update()
            self.agent.save(self.path)
        self.state = None
        mem_entry = (self.nn_state, self.nn_act, nn_s_prime, reward, True)

        self.agent.replay_buffer.add(mem_entry)


class ModVehicleTest(BaseMod):
    def __init__(self, agent_name, map_name, sim_conf, mod_conf=None):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """
        if mod_conf is None:
            mod_conf = lib.load_conf("mod_conf")

        BaseMod.__init__(self, agent_name, map_name, sim_conf, mod_conf)

        self.path = 'Vehicles/' + agent_name
        state_space = 4 + self.n_beams
        self.agent = TD3(state_space, 1, 1, agent_name)
        self.agent.load(directory=self.path)

        print(f"Agent loaded: {agent_name}")

        self.vis = LidarVizMod(10)

    def plan_act(self, obs):
        pp_action = super().act(obs)
        nn_obs = self.transform_obs(obs, pp_action)

        nn_action = self.agent.act(nn_obs, noise=0)
        self.nn_act = nn_action

        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.add_history_step(pp_action[0], nn_action[0]*self.max_steer)
        steering_angle = self.modify_references(self.nn_act, pp_action[0])
        speed = 4
        # action = np.array([steering_angle, pp_action[1]])
        action = np.array([steering_angle, speed])
        # self.vis.add_step(nn_obs[4:], steering_angle/self.max_steer)
        pp = pp_action[0]/self.max_steer
        self.vis.add_step(nn_obs[4:], pp, nn_action)

        return action
