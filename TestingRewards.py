import numpy as np
import csv, yaml
import Rewards as r

import LibFunctions as lib
from LibFunctions import load_config
from matplotlib import pyplot as plt

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
    agent_name = "ModSteer_test_rt2"

    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name, load=False)
    reward = TrackSteerReward(config, 0.005, 0.005)

    TrainVehicle(config, agent_name, vehicle, reward, 10000, 'track', show=True)

def train_mod_emp():
    agent_name = "ModEmp_test_rt"

    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name, load=False)
    reward = EmptyR()

    TrainVehicle(config, agent_name, vehicle, reward, 100000, 'track', show=False)
    # TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_mod_time():
    agent_name = "ModTime_test_rt"
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = TrackTimeReward(config, 0.025)

    TrainVehicle(config, agent_name, vehicle, reward, 100000, 'track', show=True)

def train_mod_cth():
    agent_name = "ModCth_test_rt"
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = TrackCthReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, 20000, 'track')


def train_mod_dev():
    agent_name = "ModDev_test_rt"
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name)

    reward = TrackDevReward(config)

    TrainVehicle(config, agent_name, vehicle, reward, 100000, 'track')

def train_mod_std():
    agent_name = "ModStd_test_rt"
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name, load=False)

    reward = TrackStdReward(config)

    TrainVehicle(config, agent_name, vehicle, reward, 100000, 'track')

def train_mod_std2():
    agent_name = "ModStd_test_rt2"
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name, load=False)

    reward = TrackStdReward2(config)

    TrainVehicle(config, agent_name, vehicle, reward, 100000, 'track')

def train_mod_old():
    agent_name = "ModOld_test_rt"
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name)

    reward = TrackOldReward(config)

    TrainVehicle(config, agent_name, vehicle, reward, 20000, 'track')


"""Tests """
def FullTrainRT():
    # config = load_config(config_med)
    config = load_config(config_rt)
    env_name = "porto"
    train_name = "_fin2"
    n_train = 100000
    # n_train = 100

    # 1) no racing reward
    agent_name = "ModEmpty_" + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.EmptyR()
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 2) Original  mod reward
    agent_name = "ModOriginal_" + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.TrackOriginalReward(config, 0.02, 0.02)
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 3) Distance Centerline
    agent_name = "ModCenterDis_" + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.CenterDistanceReward(config, 0.02)
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 4) CTH center
    agent_name = "ModCenterCth_" + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.CenterCTHReward(config, 0.04, 0.004)
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 5) Time
    agent_name = "ModTime_" + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.TrackTimeReward(config, 0.012)
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 6) distance ref
    agent_name = "ModRefDis_" + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.RefDistanceReward(config, 0.02)
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 7) CTH ref
    agent_name = "ModRefCth_" + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.RefCTHReward(config, 0.04, 0.004)
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 8) Steering and Velocity
    agent_name = "ModSteer_"  + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.TrackSteerReward(config, 0.01, 0.01)
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 9) Steering only
    agent_name = "ModStrVel_"  + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.TrackSteerReward(config, 0.0, 0.01)
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # Deviation action
    agent_name = "ModDeviation_" + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.TrackOriginalReward(config, 0.0, 0.02)
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

def PartialTrain():
    # config = load_config(config_med)
    config = load_config(config_rt)
    env_name = "porto"
    train_name = "_fin2"
    n_train = 100000
    # n_train = 100

    # 1) no racing reward
    # agent_name = "ModEmpty_" + env_name + train_name
    # vehicle = ModVehicleTrain(config, agent_name)
    # reward = r.EmptyR()
    # TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 2) Original  mod reward
    # agent_name = "ModOriginal_" + env_name + train_name
    # vehicle = ModVehicleTrain(config, agent_name)
    # reward = r.TrackOriginalReward(config, 0.02, 0.02)
    # TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 3) Distance Centerline
    # agent_name = "ModCenterDis_" + env_name + train_name
    # vehicle = ModVehicleTrain(config, agent_name)
    # reward = r.CenterDistanceReward(config, 0.02)
    # TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 4) CTH center
    # agent_name = "ModCenterCth_" + env_name + train_name
    # vehicle = ModVehicleTrain(config, agent_name)
    # reward = r.CenterCTHReward(config, 0.04, 0.004)
    # TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 5) Time
    agent_name = "ModTime_" + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.TrackTimeReward(config, 0.012)
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 6) distance ref
    # agent_name = "ModRefDis_" + env_name + train_name
    # vehicle = ModVehicleTrain(config, agent_name)
    # reward = r.RefDistanceReward(config, 0.02)
    # TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 7) CTH ref
    # agent_name = "ModRefCth_" + env_name + train_name
    # vehicle = ModVehicleTrain(config, agent_name)
    # reward = r.RefCTHReward(config, 0.04, 0.004)
    # TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 8) Steering and Velocity
    agent_name = "ModStrVel_"  + env_name + train_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.TrackSteerReward(config, 0.01, 0.01)
    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # 9) Steering only
    # agent_name = "ModSteer_"  + env_name + train_name
    # vehicle = ModVehicleTrain(config, agent_name)
    # reward = r.TrackSteerReward(config, 0.0, 0.01)
    # TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # Deviation action
    # agent_name = "ModDeviation_" + env_name + train_name
    # vehicle = ModVehicleTrain(config, agent_name)
    # reward = r.TrackOriginalReward(config, 0.0, 0.02)
    # TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')



def FullTest():
    # config = load_config(config_med)
    # config = load_config(config_std)
    config = load_config(config_rt)

    env_name = "porto"
    train_name = "_fin1"
    # test_name = "compare_" + env_name + train_name
    test_name = "compare_NoObs_" + env_name + train_name
    test = TestVehicles(config, test_name, 'track')

    # 1) no racing reward
    agent_name = "ModEmpty_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 2) Original  mod reward
    agent_name = "ModOriginal_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 3) Distance Centerline
    agent_name = "ModCenterDis_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 4) CTH center
    agent_name = "ModCenterCth_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 5) Time
    agent_name = "ModTime_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 6) distance ref
    agent_name = "ModRefDis_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 7) CTH ref
    agent_name = "ModRefCth_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 8) Steering and Velocity
    agent_name = "ModSteer_"  + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 9) Steering only
    # agent_name = "ModStrVel_"  + env_name + train_name
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    # Deviation action
    agent_name = "ModDeviation_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)



    # PP
    vehicle = TunerCar(config)
    test.add_vehicle(vehicle)

    # FTG
    # vehicle = FollowTheGap(config)
    # test.add_vehicle(vehicle)

    test.run_eval(1, True, add_obs=False, save=True)
    # test.run_eval(10, True, add_obs=True, save=True)
    # test.run_eval(1000, True, add_obs=True, save=False)
    # test.run_eval(1, True, add_obs=True, save=False)

    # test.run_eval(10, True)

def PartialTest():
    # config = load_config(config_med)
    # config = load_config(config_std)
    config = load_config(config_rt)

    env_name = "porto"
    train_name = "_fin2"
    # test_name = "p_compare_" + env_name + train_name
    test_name = "p1_compare_noObs_" + env_name + train_name
    test = TestVehicles(config, test_name, 'track')

    # 1) no racing reward
    # agent_name = "ModEmpty_" + env_name + train_name
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    # 2) Original  mod reward
    # agent_name = "ModOriginal_" + env_name + train_name
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    # 3) Distance Centerline
    # agent_name = "ModCenterDis_" + env_name + train_name
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    # 4) CTH center
    # agent_name = "ModCenterCth_" + env_name + train_name
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    # 5) Time
    # agent_name = "ModTime_" + env_name + train_name
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    # 6) distance ref
    # agent_name = "ModRefDis_" + env_name + train_name
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    # 7) CTH ref
    # agent_name = "ModRefCth_" + env_name + train_name
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    # 8) Steering and Velocity
    agent_name = "ModSteer_"  + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 9) Steering only
    agent_name = "ModStrVel_"  + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # Deviation action
    # agent_name = "ModDeviation_" + env_name + train_name
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)



    # PP
    vehicle = TunerCar(config)
    test.add_vehicle(vehicle)

    # FTG
    # vehicle = FollowTheGap(config)
    # test.add_vehicle(vehicle)

    # test.run_eval(1, True, add_obs=False, save=True, wait=True)
    # test.run_eval(10, True, add_obs=True, save=True)
    test.run_eval(100, True, add_obs=True, save=False, wait=True)
    # test.run_eval(1, True, add_obs=True, save=False)

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

    # vehicle = TunerCar(config)
    vehicle = FollowTheGap(config)

    test = TestVehicles(config, "FTG", 'track')
    test.add_vehicle(vehicle)
    # test.run_eval(10, True, add_obs=False)
    test.run_eval(100, True, add_obs=True)
    # testVehicle(config, vehicle, True, 10)

def test_mod():
    config = load_config(config_rt)
    test = TestVehicles(config, "Mod_test", 'track')
    # agent_name = "ModTime_raceTrack"

    # agent_name = "ModTime_test_rt"
    # agent_name = "ModSteer_test_rt"
    # agent_name = "ModCth_test"

    env_name = "porto"
    train_name = "_fin2"
    agent_name = "ModTime_" + env_name + train_name

    # agent_name = "ModDev_test_rt"
    # agent_name = "ModOld_test_rt"
    # agent_name = "ModStd_test_rt"
    
    # agent_name = "ModEmp_test_rt"
    # agent_name = "ModSteer_test_rt2"

    # agent_name = "ModTime_medForest"
    # agent_name = "ModDev_raceTrack"
    # agent_name = "ModSteer_test_om"
    vehicle = ModVehicleTest(config, agent_name)
    # vehicle = TunerCar(config)


    test.add_vehicle(vehicle)

    # agent_name = "ModStd_test_rt2"
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    # test.run_eval(10, True, add_obs=False)
    # test.run_eval(100, True, add_obs=True, wait=True)
    # test.run_eval(1, show=True, add_obs=False, wait=True)
    # test.run_eval(10, show=True, add_obs=True, wait=True)
    test.run_eval(100, True, add_obs=True, wait=True)
    # test.run_eval(1, True, add_obs=False)
    # plt.show()

def train_test():
    config = load_config(config_rt)

    agent_name = "ModSteer_test_rt2"
    reward = TrackSteerReward(config, 0.005, 0.005)
    


    vehicle = ModVehicleTrain(config, agent_name, load=False)
    TrainVehicle(config, agent_name, vehicle, reward, 100000, 'track', show=True)

    test = TestVehicles(config, "Mod_test", 'track')
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)


    test.run_eval(1, True, add_obs=False)
    test.run_eval(100, True, add_obs=True, wait=False)



def train():
    pass

    # train_mod_steer()
    # train_mod_cth()
    # train_mod_time()

    train_mod_emp()
    # train_mod_dev()
    # train_mod_std()
    # train_mod_std2()
    # train_mod_old()

    # train_time_sweep()
    # train_steer_sweep()


if __name__ == "__main__":
    # train()

    # test_compare()
    # test_compare_mod()
    # test_time_sweep()
    # test_steer_sweep()

    # FullTrainRT()
    # FullTest()

    # PartialTrain()
    PartialTest()

    # train_test()
    # test_mod()
    # test_ftg()
