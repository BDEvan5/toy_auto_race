from toy_auto_race.NavAgents.SafetyCar import SafetyCar
from toy_auto_race.NavAgents.Imitation import ImitationTrain
from toy_auto_race.NavAgents.Oracle import Oracle
import numpy as np
import timeit
import yaml

from toy_auto_race.Utils import LibFunctions as lib
import toy_auto_race.Rewards as r
from toy_auto_race.NavAgents.AgentMod import ModVehicleTest, ModVehicleTrain
from toy_auto_race.NavAgents.PurePursuit import PurePursuit
from toy_auto_race.NavAgents.FollowTheGap import TrackFGM
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
    # vehicle = FollowTheGap(env.sim_conf)
    vehicle = GapFollower()

    test_single_vehicle(env, vehicle, True, 1, add_obs=False, wait=False)
    plt.show()

def test_ref_mod():
    agent_name = "RefModTest"
    # map_name = "torino"
    map_name = "porto"

    env = TrackSim(map_name)
    vehicle = ModVehicleTest(agent_name, map_name, env.sim_conf)

    test_single_vehicle(env, vehicle, True, 10)



def generate_initial_data():
    env = ForestSim("forest2")
    oracle_vehicle = Oracle(env.sim_conf)
    imitation_vehicle = ImitationTrain("Pfeiffer", env.sim_conf)

    generat_oracle_data(env, oracle_vehicle, imitation_vehicle, 20000)
    imitation_vehicle.buffer.save_buffer("ImitationData2")


def run_initial_train():
    env = ForestSim("forest2")
    imitation_vehicle = ImitationTrain("Pfeiffer", env.sim_conf)

    imitation_vehicle.buffer.load_data("ImitationData2")
    imitation_vehicle.train(20000)

def run_imitation_training():
    env = ForestSim("forest2")
    oracle_vehicle = Oracle(env.sim_conf)
    imitation_vehicle = ImitationTrain("Pfeiffer", env.sim_conf)

    imitation_vehicle.buffer.load_data("ImitationData1")
    imitation_vehicle.train(5000)

    train_imitation_vehicle(env, oracle_vehicle, imitation_vehicle)

def test_imitation():
    env = ForestSim("forest2")
    imitation_vehicle = ImitationTrain("Pfeiffer", env.sim_conf)
    imitation_vehicle.load()

    test_single_vehicle(env, imitation_vehicle, True, 100)


def test_safety_system():
    sim_conf = lib.load_conf("fgm_config")
    # sim_conf = lib.load_conf("fgm_config")
    env = ForestSim("forest2", sim_conf)
    vehicle = SafetyCar(env.sim_conf)

    # test_oracle_forest(env, vehicle, True, 100, add_obs=True, wait=False)
    test_oracle_track(env, vehicle, True, 100, add_obs=True, wait=False)



if __name__ == "__main__":

    # train_ref_mod()
    # train_ref_mod_forest()
    # test_ref_mod()


    # test_pp()
    # test_gap_follow()


    # generate_initial_data()
    # run_initial_train()
    # run_imitation_training()
    # test_imitation()

    test_safety_system()
