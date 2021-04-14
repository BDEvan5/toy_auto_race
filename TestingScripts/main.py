import numpy as np
import timeit
import yaml

from toy_auto_race.Utils import LibFunctions as lib
import toy_auto_race.Rewards as r
from toy_auto_race.NavAgents.AgentMod import ModVehicleTest, ModVehicleTrain
from toy_auto_race.NavAgents.PurePursuit import PurePursuit
from toy_auto_race.NavAgents.FollowTheGap import FollowTheGap
from TestingScripts.TrainTest import *

from toy_f110 import TrackSim, ForestSim


def train_ref_mod():
    agent_name = "RefModTest"
    # map_name = "torino"
    map_name = "porto"
    reward = r.RefModReward(0.002)

    env = TrackSim(map_name)
    vehicle = ModVehicleTrain(agent_name, map_name, env.sim_conf, load=False)
    vehicle.set_reward_fcn(reward)

    train_vehicle(env, vehicle, 1000000)


def train_ref_mod_forest():
    agent_name = "RefModTestF"
    # map_name = "torino"
    map_name = "forest"
    # reward = r.RefModReward(0.002)

    env = ForestSim(map_name)
    reward = r.RefCTHReward(env.sim_conf, map_name, 0.004, 0.0004)
    vehicle = ModVehicleTrain(agent_name, map_name, env.sim_conf)
    vehicle.set_reward_fcn(reward)

    train_vehicle(env, vehicle, 1000000)

"""Tests"""
def test_pp():
    map_name = "porto"
    
    env = TrackSim(map_name)
    vehicle = PurePursuit(map_name, env.sim_conf)

    test_single_vehicle(env, vehicle, True, 100)


def test_gap_follow():
    map_name = "torino"
    # map_name = "porto"
    
    sim_conf = lib.load_conf("fgm_config")
    env = TrackSim(map_name, sim_conf)
    vehicle = FollowTheGap(env.sim_conf)

    test_single_vehicle(env, vehicle, True, 1, add_obs=False, wait=False)


def test_ref_mod():
    agent_name = "RefModTest"
    # map_name = "torino"
    map_name = "porto"

    env = TrackSim(map_name)
    vehicle = ModVehicleTest(agent_name, map_name, env.sim_conf)

    test_single_vehicle(env, vehicle, True, 10)


def time_sim():
    t = timeit.timeit(stmt=test_gap_follow, number=1)
    print(f"Time (1): {t}")

    t = timeit.timeit(stmt=test_gap_follow, number=2)
    print(f"Time (2): {t}")


if __name__ == "__main__":

    # train_ref_mod()
    # train_ref_mod_forest()
    # test_ref_mod()


    # test_pp()
    # test_gap_follow()

    time_sim()

    
