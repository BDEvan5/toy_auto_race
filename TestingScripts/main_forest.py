
from os import wait
from toy_auto_race.NavAgents.Oracle import Oracle
from toy_auto_race.NavAgents.AgentNav import NavTrainVehicle, NavTestVehicle
import numpy as np
import timeit
import yaml

from toy_auto_race.Utils import LibFunctions as lib
import toy_auto_race.Rewards as r
from toy_auto_race.NavAgents.AgentMod import ModVehicleTest, ModVehicleTrain
from toy_auto_race.NavAgents.PurePursuit import PurePursuit
from toy_auto_race.NavAgents.FollowTheGap import FollowTheGap, GapFollower
from TestingScripts.TrainTest import *

from toy_f110 import ForestSim

map_name = "forest2"
nav_name = "Navforest_nr"
mod_name = "ModForest_nr"
eval_name = "BigTest_nr"

"""
Training Functions
"""
def train_nav():
    # agnet_name = "NavForest"

    env = ForestSim(map_name)
    vehicle = NavTrainVehicle(nav_name, env.sim_conf)

    # train_vehicle(env, vehicle, 1000)
    train_vehicle(env, vehicle, 100000)


def train_mod():
    # agent_name = "ModForest_dev"
    env = ForestSim(map_name)

    # reward = r.RefCTHReward(env.sim_conf, map_name, 0.004, 0.0004)
    # reward = r.RefModReward(0.002)

    vehicle = ModVehicleTrain(mod_name, map_name, env.sim_conf)
    # vehicle.set_reward_fcn(reward)

    # train_vehicle(env, vehicle, 1000)
    train_vehicle(env, vehicle, 100000)


"""Test Functions"""
def test_nav():
    # agent_name = "NavForest"

    env = ForestSim(map_name)
    vehicle = NavTestVehicle(nav_name, env.sim_conf)

    test_single_vehicle(env, vehicle, True, 10, wait=True)



def test_follow_the_gap():
    sim_conf = lib.load_conf("fgm_config")
    env = ForestSim(map_name, sim_conf)
    # vehicle = FollowTheGap(env.sim_conf)
    vehicle = GapFollower()

    # test_single_vehicle(env, vehicle, True, 10, False)
    # test_single_vehicle(env, vehicle, True, 100, add_obs=True, vis=True)
    test_single_vehicle(env, vehicle, True, 100, add_obs=True, vis=False)


def test_oracle():
    env = ForestSim(map_name)
    vehicle = Oracle(env.sim_conf)

    test_oracle_vehicle(env, vehicle, True, 100, True, wait=False)


def test_mod():
    # agent_name = "ModForest"

    env = ForestSim(map_name)
    vehicle = ModVehicleTest(mod_name, map_name, env.sim_conf)

    test_single_vehicle(env, vehicle, True, 10, wait=False)



def run_all_tests():
    test_nav()
    test_follow_the_gap()
    test_oracle()
    test_mod()

def big_test():
    env = ForestSim(map_name)
    test = TestVehicles(env.sim_conf, eval_name)

    # agent_name = "NavForest"
    vehicle = NavTestVehicle(nav_name, env.sim_conf)
    test.add_vehicle(vehicle)

    # vehicle = FollowTheGap(env.sim_conf)
    # test.add_vehicle(vehicle)

    vehicle = Oracle(env.sim_conf)
    test.add_vehicle(vehicle)

    # agent_name = "ModForest"
    vehicle = ModVehicleTest(mod_name, map_name, env.sim_conf)
    test.add_vehicle(vehicle)

    # test.run_eval(env, 1, True)
    test.run_eval(env, 1000, True)
    
    

if __name__ == "__main__":

    # train_nav()
    # train_mod()

    # test_nav()
    test_follow_the_gap()
    # test_oracle()
    # test_mod()

    # run_all_tests()
    # big_test()







