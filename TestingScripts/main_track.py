
from toy_auto_race.NavAgents.Oracle import Oracle
from toy_auto_race.NavAgents.AgentNav import NavTrainVehicle, NavTestVehicle
import numpy as np

from toy_auto_race.Utils import LibFunctions as lib
import toy_auto_race.Rewards as r
from toy_auto_race.NavAgents.AgentMod import ModVehicleTest, ModVehicleTrain
from toy_auto_race.NavAgents.PurePursuit import PurePursuit
from toy_auto_race.NavAgents.FollowTheGap import FollowTheGap, GapFollower
from TestingScripts.TrainTest import *

from toy_f110 import TrackSim

# map_name = "torino"
# map_name = "porto"
map_name = "race_track"
# map_name = "berlin"
run_num = 4
nav_name = "Nav_" + map_name + f"_{run_num}"
mod_name = "Mod_" + map_name + f"_{run_num}"
eval_name = "BigTest_track"

"""
Training Functions
"""

def train_mod():
    env = TrackSim(map_name)

    # vehicle = ModVehicleTrain(mod_name, map_name, env.sim_conf)
    vehicle = ModVehicleTrain(mod_name, map_name, env.sim_conf, load=False, h_size=400)
    vehicle.beta_slope = 0.008

    train_vehicle(env, vehicle, 400000)


"""Test Functions"""


def test_follow_the_gap():
    sim_conf = lib.load_conf("fgm_config")
    env = TrackSim(map_name, sim_conf)
    # vehicle = FollowTheGap(env.sim_conf)
    vehicle = GapFollower()

    test_single_vehicle(env, vehicle, True, 10, False)
    # test_single_vehicle(env, vehicle, True, 100, add_obs=False, vis=True)
    # test_single_vehicle(env, vehicle, True, 100, add_obs=True, vis=False)


def test_oracle():
    env = TrackSim(map_name)
    vehicle = Oracle(env.sim_conf)

    test_oracle_track(env, vehicle, True, 100, add_obs=False, wait=False)


def test_mod():

    env = TrackSim(map_name)
    vehicle = ModVehicleTest(mod_name, map_name, env.sim_conf)

    test_single_vehicle(env, vehicle, True, 100, wait=False, vis=False)
    # test_single_vehicle(env, vehicle, False, 100, wait=False, vis=False)

def big_test():
    env = TrackSim(map_name)
    test = TestVehicles(env.sim_conf, eval_name)

    vehicle = FollowTheGap(env.sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ModVehicleTest(mod_name, map_name, env.sim_conf)
    test.add_vehicle(vehicle)

    # test.run_eval(env, 1, True, add_obs=False)
    # test.run_eval(env, 100, True)
    
    

if __name__ == "__main__":

    train_mod()

    # test_follow_the_gap()
    # test_oracle()
    test_mod()

    # big_test()







