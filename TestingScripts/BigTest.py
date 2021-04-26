
from toy_auto_race.NavAgents.Oracle import Oracle
from toy_auto_race.NavAgents.AgentNav import NavTrainVehicle, NavTestVehicle
import numpy as np

from toy_auto_race.Utils import LibFunctions as lib
import toy_auto_race.Rewards as r
from toy_auto_race.NavAgents.AgentMod import ModVehicleTest, ModVehicleTrain
from toy_auto_race.NavAgents.FollowTheGap import ForestFGM, TrackFGM
from TestingScripts.TrainTest import *

from toy_f110 import ForestSim, TrackSim

from toy_auto_race.NavAgents.Oracle import Oracle
from toy_auto_race.NavAgents.AgentNav import NavTrainVehicle, NavTestVehicle
import numpy as np


map_name_f = "forest2"
nav_name_f = "Navforest_1"
mod_name_f = "ModForest_1"
# mod_name = "ModForest_nr6"
# nav_name = "Navforest_nr5"
repeat_name = "RepeatTest_1"
eval_name_f= "BigTest1"

"""
Training Functions
"""
def train_nav_f():
    env = ForestSim(map_name_f)
    vehicle = NavTrainVehicle(nav_name_f, env.sim_conf, h_size=500)

    # train_vehicle(env, vehicle, 100)
    train_vehicle(env, vehicle, 200000)


def train_mod_f():
    env = ForestSim(map_name_f)

    vehicle = ModVehicleTrain(mod_name_f, map_name_f, env.sim_conf, load=False, h_size=500)
    # train_vehicle(env, vehicle, 100)
    train_vehicle(env, vehicle, 200000)


def train_repeatability():
    env = ForestSim(map_name_f)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"

        vehicle = ModVehicleTrain(train_name, map_name_f, env.sim_conf, load=False)

        # train_vehicle(env, vehicle, 100)
        train_vehicle(env, vehicle, 200000)




def big_test_f():
    env = ForestSim(map_name_f)
    test = TestVehicles(env.sim_conf, eval_name_f)

    vehicle = NavTestVehicle(nav_name_f, env.sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ForestFGM()
    test.add_vehicle(vehicle)

    vehicle = Oracle(env.sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ModVehicleTest(mod_name_f, map_name_f, env.sim_conf)
    test.add_vehicle(vehicle)

    # test.run_eval(env, 1, True)
    test.run_eval(env, 1000, True, wait=False)



def test_repeat():
    env = ForestSim(map_name_f)
    test = TestVehicles(env.sim_conf, repeat_name)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"
        vehicle = ModVehicleTest(train_name, map_name_f, env.sim_conf)
        test.add_vehicle(vehicle)

    # test.run_eval(env, 1000, False)
    test.run_eval(env, 1000, False)




# map_name = "torino"
# map_name = "porto"
map_name = "race_track"
# map_name = "berlin"
run_num = 1
nav_name = "Nav_" + map_name + f"_{run_num}"
mod_name = "Mod_" + map_name + f"_{run_num}"
eval_name = "TrackEval_1"

"""
Training Functions
"""

def train_mod():
    env = TrackSim(map_name)

    # vehicle = ModVehicleTrain(mod_name, map_name, env.sim_conf)
    vehicle = ModVehicleTrain(mod_name, map_name, env.sim_conf, load=False, h_size=500)

    train_vehicle(env, vehicle, 500000)
    # train_vehicle(env, vehicle, 100)

def train_nav():
    env = TrackSim(map_name)

    # vehicle = ModVehicleTrain(mod_name, map_name, env.sim_conf)
    vehicle = NavTrainVehicle(mod_name, env.sim_conf, h_size=500)

    # train_vehicle(env, vehicle, 100)
    train_vehicle(env, vehicle, 500000)


def big_test():
    sim_conf = lib.load_conf("race_config")
    env = TrackSim(map_name, sim_conf)
    test = TestVehicles(sim_conf, eval_name)

    vehicle = TrackFGM()
    test.add_vehicle(vehicle)

    vehicle = ModVehicleTest(mod_name, map_name, sim_conf)
    test.add_vehicle(vehicle)

    # test.run_eval(env, 1, True, add_obs=False)
    test.run_eval(env, 100, True)
    


if __name__ == "__main__":
    
    # train_mod_f()
    # train_nav_f()

    # train_repeatability()
    
    # train_mod()
    # train_nav()

    # big_test_f()
    # test_repeat()
    big_test()






