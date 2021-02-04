import numpy as np
import matplotlib.pyplot as plt
from HistoryStructs import RewardAnalyser, TrainHistory

import timeit
import yaml

from Simulator import ForestSim
from SimMaps import  ForestMap
from ModelsRL import ReplayBufferDQN, ReplayBufferTD3
import LibFunctions as lib
from LibFunctions import load_config
from Rewards import *

from AgentOptimal import OptimalAgent, TunerCar
from AgentMPC import AgentMPC
from AgentMod import ModVehicleTest, ModVehicleTrain
from RefGen import GenVehicle, GenTest

names = ['columbia', 'levine_blocked', 'mtl', 'porto', 'torino', 'race_track']
name = names[5]
myMap = 'TrackMap1000'
forest_name = 'forest'
bfg = 'BigForest'

config_sf = "small_forest"
config_std = "std_config"




def RunOptimalAgent():
    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    config = lib.load_config("std_config")


    env_map = ForestMap(config)
    env = ForestSim(env_map)

    # agent = OptimalAgent()
    agent = TunerCar(config)
    ra = RewardAnalyser()

    done, state, score = False, env.reset(), 0.0
    wpts = agent.init_agent(env_map)
    env.render(wait=False)
    # env.render(True, wpts)
    while not done:
        action = agent.act(state)
        s_p, r, done, _ = env.step(action)
        score += r

        state = s_p


        # env.render(True, wpts)
        # env.env_map.render_map(4, True)
        # env.render(False)

    print(f"Score: {score}")
    ra.show_rewards()
    env.history.show_history(vs=env_map.vs)
    env.history.show_forces()
    env.render(wait=True)

def RunMpcAgent():
    env_map = ForestMap(forest_name)
    env = ForestSim(env_map)

    agent = AgentMPC()

    done, state, score = False, env.reset(), 0.0
    wpts = agent.init_agent(env_map)
    # env.render(wait=True)
    # env.render(True, wpts)
    while not done:
        action, pts, t, cwpts = agent.act(state)
        s_p, r, done, _ = env.step(action, dt=t)
        score += r
        state = s_p

        # env.render(True, wpts)
        # env.env_map.render_map(4, True)
        env.render(True, pts1=pts, pts2=cwpts)

    print(f"Score: {score}")
    # env.show_history()
    env.render(wait=True)



"""Train"""
def TrainVehicle(config, agent_name, vehicle, reward, steps=20000):
    path = 'Vehicles/' + agent_name
    buffer = ReplayBufferTD3()

    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    env_map = ForestMap(config)
    env = ForestSim(env_map)

    t_his = TrainHistory(agent_name)
    print_n = 500

    done = False
    state, wpts, vs = env.reset()
    vehicle.init_agent(env_map)
    reward.init_reward(wpts, vs)

    for n in range(steps):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a)

        new_r = reward(state, a, s_prime, r)
        vehicle.add_memory_entry(new_r, done, s_prime, buffer)
        t_his.add_step_data(new_r)

        state = s_prime
        vehicle.agent.train(buffer, 2)
        
        # env.render(False)

        if n % print_n == 0 and n > 0:
            t_his.print_update()
            vehicle.agent.save(directory=path)
        
        if done:
            t_his.lap_done(True)
            vehicle.show_vehicle_history()
            env.render(wait=False, save=False)

            vehicle.reset_lap()
            state, wpts, vs = env.reset()
            reward.init_reward(wpts, vs)


    vehicle.agent.save(directory=path)
    t_his.save_csv_data()

    print(f"Finished Training: {agent_name}")

    return t_his.rewards

"""General test function"""
def testVehicle(vehicle, show=False, obs=True):
    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    env_map = ForestMap(forest_name)
    env = ForestSim(env_map)

    crashes = 0
    completes = 0
    lap_times = []

    wpts = vehicle.init_agent(env_map)
    done, state, score = False, env.reset(), 0.0
    for i in range(100): # 10 laps
        print(f"Running lap: {i}")
        # if obs:
        #     env_map.reset_map()
        while not done:
            a = vehicle.act(state)
            s_p, r, done, _ = env.step(a)
            state = s_p
            # env.render(False, vehicle.scan_sim)
        print(f"Lap time updates: {env.steps}")
        if show:
            # vehicle.show_vehicle_history()
            env.render(wait=False)
            # env.render(wait=True)

        if r == -1:
            crashes += 1
        else:
            completes += 1
            lap_times.append(env.steps)
        state = env.reset()
        
        # env.reset_lap()
        env.reset()
        vehicle.reset_lap()
        done = False

    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {(completes / (completes + crashes) * 100):.2f} %")
    print(f"Lap times: {lap_times} --> Avg: {np.mean(lap_times)}")


""" Training sets"""
def train_gen_std():
    load = False

    agent_name = "GenStd_test"
    config = load_config(config_sf)
    vehicle = GenVehicle(config, agent_name, load)
    # reward = StdNavReward(config, -0.02, 0.2, 0)
    reward = StdNavReward(config, 0, 0.2, 0)

    TrainVehicle(config, agent_name, vehicle, reward)

def train_gen_cth():
    load = False

    agent_name = "GenCth_test"
    config = load_config(config_sf)
    vehicle = GenVehicle(config, agent_name, load)
    reward = CrossTrackHeadingReward(config, 0.5, 1, 0.1)

    TrainVehicle(config, agent_name, vehicle, reward)

def train_gen_steer():
    load = False

    agent_name = "GenSteer_02_02_02"
    config = load_config("std_config")
    vehicle = GenVehicle(config, agent_name, load)
    reward = OnlineSteering(config, 0.2, 0.2, 0.2)

    TrainVehicle(config, agent_name, vehicle, reward)

"""Mod training"""
def train_mod_std():
    load = False

    agent_name = "ModStd_test"
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



"""Total functions"""
def test_GenCth():
    agent_name = "GenCth_test"
    config = load_config("std_config")
    vehicle = GenTest(config, agent_name)

    testVehicle(vehicle, True)


def testOptimal():
    agent = OptimalAgent()

    testVehicle(agent, obs=False, show=True)



# Development functions

def test_mapping():
    env_map = ForestMap(forest_name)
    env = ForestSim(env_map)

    for i in range(100):
        env.reset()
        env_map.get_optimal_path()
        env.render(wait=False)
        env_map.get_velocity()
        env.render(wait=True)
        

def timing():
    # t = timeit.timeit(stmt=RunModAgent, number=1)
    # print(f"Time: {t}")
    
    t = timeit.timeit(stmt=testOptimal, number=1)
    print(f"Time: {t}")


if __name__ == "__main__":

    train_gen_std()
    # train_gen_steer()
    # train_gen_cth()

    # train_mod_std()
    # train_mod_cth()
    # train_mod_time()

    # timing()

    # RunMpcAgent()
    # test_mapping()





    
