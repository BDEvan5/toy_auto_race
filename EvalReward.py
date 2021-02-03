from Rewards import StdNavReward
import numpy as np
import matplotlib.pyplot as plt
from HistoryStructs import RewardAnalyser, TrainHistory

import timeit
import yaml

from Simulator import ForestSim
from SimMaps import  ForestMap
from ModelsRL import ReplayBufferDQN, ReplayBufferTD3
import LibFunctions as lib

from AgentOptimal import OptimalAgent, TunerCar
from AgentMPC import AgentMPC
from AgentMod import ModVehicleTest, ModVehicleTrain
from RefGen import GenTrainStd, GenTrainStr, GenTest


names = ['columbia', 'levine_blocked', 'mtl', 'porto', 'torino', 'race_track']
name = names[5]
myMap = 'TrackMap1000'
forest_name = 'forest'
bfg = 'BigForest'

def load_config(fname):
    with open('config/' + fname + '.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    return conf_dict


def RunOptimalAgent():
    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    config = load_config("std_config")


    env_map = ForestMap(config)
    env = ForestSim(env_map)

    # agent = OptimalAgent()
    agent = TunerCar(config)

    ra = RewardAnalyser()
    reward = StdNavReward(config)

    done, state, score = False, env.reset(), 0.0
    wpts = agent.init_agent(env_map)
    env.render(wait=False)
    # env.render(True, wpts)
    while not done:
        action = agent.act(state)
        s_p, r, done, _ = env.step(action)
        score += r
        new_r = reward(state, action, s_p, r)
        ra.add_reward(new_r)
        
        state = s_p

        # env.render(True, wpts)
        # env.env_map.render_map(4, True)
        # env.render(False)

    print(f"Score: {score}")
    ra.show_rewards()
    # env.history.show_history(vs=env_map.vs)
    # env.history.show_forces()
    env.render(wait=True)



if __name__ == "__main__":
    RunOptimalAgent()
