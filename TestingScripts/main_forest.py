
import numpy as np
import timeit
import yaml

from toy_auto_race.Utils import LibFunctions as lib
import toy_auto_race.Rewards as r
from toy_auto_race.NavAgents.AgentMod import ModVehicleTest, ModVehicleTrain
from toy_auto_race.NavAgents.PurePursuit import PurePursuit
from toy_auto_race.NavAgents.FollowTheGap import FollowTheGap
from TestingScripts.TrainTest import *

from toy_f110 import ForestSim

map_name = "forest2"


"""
Training Functions to write
"""
def train_nav():
    # train a pure navigation planner
    pass



def train_mod():
    agent_name = "ModForest"
    map_name = "forest"
    # reward = r.RefModReward(0.002)

    env = ForestSim(map_name)
    reward = r.RefCTHReward(env.sim_conf, map_name, 0.004, 0.0004)
    vehicle = ModVehicleTrain(agent_name, map_name, env.sim_conf)
    vehicle.set_reward_fcn(reward)

    train_vehicle(env, vehicle, 100000)

# Obstacles
def test_nav():
    pass 


def test_follow_the_gap():

    env = ForestSim(map_name)
    vehicle = FollowTheGap(env.sim_conf)

    # test_single_vehicle(env, vehicle, True, 10, False)
    test_single_vehicle(env, vehicle, True, 100, add_obs=True)


def test_oracle():
    pass 


def test_mod():
    pass 



def run_all_tests():
    test_nav()
    test_follow_the_gap()
    test_oracle()
    test_mod()


if __name__ == "__main__":

    # train_nav()
    # train_mod()

    # test_nav()
    test_follow_the_gap()
    # test_oracle()
    # test_mod()










