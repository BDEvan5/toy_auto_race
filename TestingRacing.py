import numpy as np
import csv, yaml
from Rewards import CthReward, TimeReward, SteerReward


import LibFunctions as lib
from LibFunctions import load_config

# from AgentOptimal import OptimalAgent
from AgentOptimal import TunerCar
from AgentMod import ModVehicleTest, ModVehicleTrain
from RefGen import GenTest, GenVehicle


config_sf = "small_forest"
config_std = "std_config"


from Testing import TestVehicles, TrainVehicle

config_sf = "small_forest"
config_std = "std_config"
config_med = "med_forest"


""" Training sets"""
def train_gen_time():
    load = False

    agent_name = "GenTime_test"
    config = load_config(config_med)
    vehicle = GenVehicle(config, agent_name, load)
    reward = TimeReward(config, 0.06)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_gen_cth():
    load = False

    agent_name = "GenCth_test"
    config = load_config(config_med)
    vehicle = GenVehicle(config, agent_name, load)
    reward = CthReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_gen_steer():
    load = False

    agent_name = "GenSteer_test"
    config = load_config(config_med)
    vehicle = GenVehicle(config, agent_name, load)
    reward = SteerReward(config, 0.1, 0.1)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

"""Mod training"""
def train_mod_steer():
    load = False

    agent_name = "ModSteer_test"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = SteerReward(config, 0.1, 0.1)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_mod_time():
    load = False

    agent_name = "ModTime_test"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.06)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_mod_cth():
    load = False

    agent_name = "ModCth_test"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = CthReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)


"""Tests """
def test_compare():
    config = load_config(config_med)
    # config = load_config(config_std)
    # test = TestVehicles(config, "test_compare_mod")
    # test = TestVehicles(config, "test_compare_gen")
    test = TestVehicles(config, "test_compare")

    # mod
    agent_name = "ModTime_test"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModCth_test"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_test"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # gen
    agent_name = "GenTime_test"
    vehicle = GenTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "GenCth_test"
    vehicle = GenTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "GenSteer_test"
    vehicle = GenTest(config, agent_name)
    test.add_vehicle(vehicle)

    # PP
    vehicle = TunerCar(config)
    test.add_vehicle(vehicle)

    test.run_eval(100, True)



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

