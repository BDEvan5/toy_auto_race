import numpy as np
import csv, yaml
from Rewards import CthReward, SteerRewardTrack, TimeReward, SteerReward, TimeRewardTrack, TrackDevReward


import LibFunctions as lib
from LibFunctions import load_config

# from AgentOptimal import OptimalAgent
from AgentOptimal import FollowTheGap, TunerCar
from AgentMod import ModVehicleTest, ModVehicleTrain


config_sf = "small_forest"
config_std = "std_config"


from Testing import TestVehicles, TrainVehicle

config_sf = "small_forest"
config_std = "std_config"
config_med = "med_forest"
config_rt = "race_track"


"""Mod training"""
def train_mod_steer():
    agent_name = "ModSteer_test_rt"
    # agent_name = "ModSteer_test_01_01"
    # config = load_config(config_med)
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = SteerRewardTrack(config, 0.1, 0.1)

    TrainVehicle(config, agent_name, vehicle, reward, 4000, 'track')
    # TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_mod_time():
    agent_name = "ModTime_test_rt"
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = TimeRewardTrack(config, 0.12)

    TrainVehicle(config, agent_name, vehicle, reward, 4000, 'track')

def train_mod_cth():
    load = False

    agent_name = "ModCth_test"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = CthReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)


def train_mod_dev():
    agent_name = "ModDev_test_rt"
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name)

    reward = TrackDevReward(config)

    TrainVehicle(config, agent_name, vehicle, reward, 20000, 'track')


"""Tests """

def FullTrain():
    config = load_config(config_med)
    env_name = "raceTrack"
    n_train = 6000

    agent_name = "ModSteer_"  + env_name
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = SteerReward(config, 0.1, 0.1)

    TrainVehicle(config, agent_name, vehicle, reward, n_train)

    agent_name = "ModTime_" + env_name
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = TimeReward(config, 0.12)

    TrainVehicle(config, agent_name, vehicle, reward, n_train)

    agent_name = "ModCth_" + env_name
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = CthReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, n_train)

def FullTest():
    config = load_config(config_med)
    # config = load_config(config_std)

    env_name = "medForest"
    test_name = "compare_" + env_name + "_1"
    test = TestVehicles(config, test_name)

    # mod
    agent_name = "ModTime_" + env_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModCth_" + env_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_" + env_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # PP
    vehicle = TunerCar(config)
    test.add_vehicle(vehicle)

    # FTG
    vehicle = FollowTheGap(config)
    test.add_vehicle(vehicle)

    # test.run_eval(10, True, add_obs=False)
    test.run_eval(10, True, add_obs=True, save=True)

    # test.run_eval(10, True)

"""Time sweep"""
def train_time_sweep():
    load = False
    config = load_config(config_med)

    agent_name = "ModTime_test_04"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)


    agent_name = "ModTime_test_06"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.06)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_08"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.08)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_10"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.1)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_15"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.15)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_18"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.18)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_20"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.20)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_25"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.25)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def test_time_sweep():
    config = load_config(config_med)

    test = TestVehicles(config, "test_time_sweep")

    # mod
    agent_name = "ModTime_test_04"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_06"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_08"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_10"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_15"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_18"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_20"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_25"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    test.run_eval(100, False)

"""Steer sweep"""
def train_steer_sweep():
    load = False
    config = load_config(config_med)

    agent_name = "ModSteer_test_004_004"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = SteerReward(config, 0.04, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)
    agent_name = "ModSteer_test_008_008"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = SteerReward(config, 0.08, 0.08)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    # agent_name = "ModSteer_test_01_01"
    # vehicle = ModVehicleTrain(config, agent_name, load)
    # reward = SteerReward(config, 0.1, 0.1)

    # TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModSteer_test_015_015"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = SteerReward(config, 0.15, 0.15)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModSteer_test_02_02"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = SteerReward(config, 0.2, 0.2)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def test_steer_sweep():
    config = load_config(config_med)

    test = TestVehicles(config, "test_steer_sweep")

    # mod
    agent_name = "ModSteer_test_004_004"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_test_008_008"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_test_01_01"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_test_015_015"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_test_02_02"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)


    test.run_eval(100, False)



"""Smaller tests"""

def test_ftg():
    # config = load_config(config_med)
    config = load_config(config_rt)

    vehicle = TunerCar(config)
    # vehicle = FollowTheGap(config)

    test = TestVehicles(config, "FTG", 'track')
    test.add_vehicle(vehicle)
    test.run_eval(10, True, add_obs=False)
    # testVehicle(config, vehicle, True, 10)

def test_mod():
    config = load_config(config_rt)
    # agent_name = "ModTime_raceTrack"

    # agent_name = "ModSteer_test_rt"
    agent_name = "ModDev_test_rt"
    # agent_name = "ModTime_test_rt"
    # agent_name = "ModTime_medForest"
    vehicle = ModVehicleTest(config, agent_name)
    # vehicle = TunerCar(config)

    test = TestVehicles(config, "Mod_test", 'track')
    test.add_vehicle(vehicle)
    # test.run_eval(10, True, add_obs=False)
    test.run_eval(10, True, add_obs=True)


def train():
    pass

    # train_mod_steer()
    # train_mod_cth()
    # train_mod_time()

    train_mod_dev()

    # train_time_sweep()
    # train_steer_sweep()


if __name__ == "__main__":
    # train()

    # test_compare()
    # test_compare_mod()
    # test_time_sweep()
    # test_steer_sweep()

    # FullTrain()
    # FullTest()


    # test_ftg()
    test_mod()
