
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


map_name_forest = "forest2"
train_test_n = 3
nav_name_forest = f"Navforest_{train_test_n}"
mod_name_forest = f"ModForest_{train_test_n}"

repeat_name = f"RepeatTest_{train_test_n}"
eval_name_f= f"BigTest{train_test_n}"

map_name_track = "race_track"
run_num = 3
nav_name_track = "Nav_" + map_name_track + f"_{run_num}"
mod_name_track = "Mod_" + map_name_track + f"_{run_num}"
eval_name_track = "TrackEval_1"



"""
Training Functions
"""
def train_nav_forest():
    env = ForestSim(map_name_forest)
    vehicle = NavTrainVehicle(nav_name_forest, env.sim_conf, h_size=200)

    # train_vehicle(env, vehicle, 100)
    train_vehicle(env, vehicle, 200000)


def train_mod_forest():
    env = ForestSim(map_name_forest)

    vehicle = ModVehicleTrain(mod_name_forest, map_name_forest, env.sim_conf, load=False, h_size=200)
    # train_vehicle(env, vehicle, 100)
    train_vehicle(env, vehicle, 200000)

def train_repeatability():
    env = ForestSim(map_name_forest)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"

        vehicle = ModVehicleTrain(train_name, map_name_forest, env.sim_conf, load=False)

        # train_vehicle(env, vehicle, 100)
        train_vehicle(env, vehicle, 200000)


def run_comparison_forest():
    env = ForestSim(map_name_forest)
    test = TestVehicles(env.sim_conf, eval_name_f)

    vehicle = NavTestVehicle(nav_name_forest, env.sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ForestFGM()
    test.add_vehicle(vehicle)

    vehicle = Oracle(env.sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ModVehicleTest(mod_name_forest, map_name_forest, env.sim_conf)
    # vehicle = ModVehicleTest("ModForest_nr6", map_name_forest, env.sim_conf)
    test.add_vehicle(vehicle)

    # test.run_eval(env, 1, True)
    test.run_eval(env, 100, False, wait=False)



def test_repeat():
    env = ForestSim(map_name_forest)
    test = TestVehicles(env.sim_conf, repeat_name)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"
        vehicle = ModVehicleTest(train_name, map_name_forest, env.sim_conf)
        test.add_vehicle(vehicle)

    # test.run_eval(env, 1000, False)
    test.run_eval(env, 100, False)





"""
Training Functions
"""

def train_mod_track():
    env = TrackSim(map_name_track)

    # vehicle = ModVehicleTrain(mod_name, map_name, env.sim_conf)
    vehicle = ModVehicleTrain(mod_name_track, map_name_track, env.sim_conf, load=False, h_size=200)

    train_vehicle(env, vehicle, 200000)
    # train_vehicle(env, vehicle, 100)

def train_nav_track():
    env = TrackSim(map_name_track)

    # vehicle = ModVehicleTrain(mod_name, map_name, env.sim_conf)
    vehicle = NavTrainVehicle(nav_name_track, env.sim_conf, h_size=200)

    # train_vehicle(env, vehicle, 100)
    train_vehicle(env, vehicle, 200000)


def big_test_track():
    sim_conf = lib.load_conf("race_config")
    env = TrackSim(map_name_track, sim_conf)
    test = TestVehicles(sim_conf, eval_name_track)

    vehicle = TrackFGM()
    test.add_vehicle(vehicle)

    vehicle = ModVehicleTest(mod_name_track, map_name_track, sim_conf)
    test.add_vehicle(vehicle)

    # vehicle = NavTestVehicle(mod_name_track, map_name_track, sim_conf)
    # test.add_vehicle(vehicle)

    # test.run_eval(env, 1, True, add_obs=False)
    test.run_eval(env, 100, False)
    


if __name__ == "__main__":
    
    # train_mod_forest()
    # train_nav_forest()

    
    # train_mod_track()
    # train_nav_track()

    # run_comparison_forest()
    # big_test_track()

    # train_repeatability()
    test_repeat()





