from Rewards import CrossTrackHeadingReward, ModHeadingReward, ModStdTimeReward, OnlineSteering, StdNavReward
import numpy as np
import csv, yaml


import LibFunctions as lib

# from AgentOptimal import OptimalAgent
from AgentMod import ModVehicleTest, ModVehicleTrain
from RefGen import GenTest, GenVehicle

names = ['columbia', 'levine_blocked', 'mtl', 'porto', 'torino', 'race_track']
name = names[5]
myMap = 'TrackMap1000'
forest_name = 'forest'

config_sf = "small_forest"
config_std = "std_config"


from Testing import TestVehicles, TrainVehicle



""" Training sets"""
def train_gen_std():
    load = False

    agent_name = "GenStd_0_02_0"
    config = load_config("std_config")
    vehicle = GenVehicle(config, agent_name, load)
    reward = StdNavReward(config, 0, 0.2, 0)

    TrainVehicle(config, agent_name, vehicle, reward)

def train_gen_cth():
    load = False

    agent_name = "GenCth_1_1_1"
    config = load_config("std_config")
    vehicle = GenVehicle(config, agent_name, load)
    reward = CrossTrackHeadingReward(config, 1, 1, 1)

    TrainVehicle(config, agent_name, vehicle, reward)

def train_gen_steer():
    load = False

    agent_name = "GenSteer_02_02"
    config = load_config("std_config")
    vehicle = GenVehicle(config, agent_name, load)
    reward = OnlineSteering(config, 0.2, 0.2)

    TrainVehicle(config, agent_name, vehicle, reward)

"""Mod training"""
def train_mod_std():
    load = False

    agent_name = "ModStd_04_02_0"
    config = load_config("std_config")
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = ModStdTimeReward(config, 0.4, 0.2, 0)

    TrainVehicle(config, agent_name, vehicle, reward)

def train_mod_time():
    load = False

    agent_name = "ModTime_01_01_1"
    config = load_config("std_config")
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = ModStdTimeReward(config, 0.1, 0.1, 1)

    TrainVehicle(config, agent_name, vehicle, reward)

def train_mod_cth():
    load = False

    agent_name = "ModCth_1_1_1"
    config = load_config("std_config")
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = ModHeadingReward(config, 1, 1, 1)

    TrainVehicle(config, agent_name, vehicle, reward)


def test_compare():
    config = load_config("std_config")
    test = TestVehicles(config, 'RaceComparison_t')

    agent_name = "GenStd_0_02_0"
    vehicle = GenTest(config, agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "GenCth_1_1_1"
    # vehicle = GenTest(config, agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "GenSteer_02_02"
    vehicle = GenTest(config, agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "ModStd_04_02_0"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_01_01_1"
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "ModCth_1_1_1"
    # vehicle = GenTest(config, agent_name)
    # test.add_vehicle(vehicle)



    test.run_eval(10, True)

def test_compare_mod():
    config = lib.load_config(config_std)
    test = TestVehicles(config, 'ModCompare_t1')

    agent_name = "ModStd_test"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test"
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "ModCth_test"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)


    test.run_eval(10, True)

def test_compare_std():
    config = lib.load_config("std_config")
    test = TestVehicles(config, 'RaceComparison_t')

    agent_name = "GenStd_0_02_0"
    agent_name = "GenStd_test"
    vehicle = GenTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "GenCth_1_1_1"
    agent_name = "GenCth_test"
    vehicle = GenTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "GenSteer_02_02"
    agent_name = "GenSteer_test"
    vehicle = GenTest(config, agent_name)
    test.add_vehicle(vehicle)




    test.run_eval(10, True)


def train():
    pass
    # train_gen_std()
    train_gen_steer()
    # train_gen_cth()

    # train_mod_std()
    # train_mod_cth()
    # train_mod_time()


if __name__ == "__main__":
    # train()

    # test_compare()
    test_compare_mod()
# 

