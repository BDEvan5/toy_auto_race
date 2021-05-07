


from toy_auto_race.Utils import LibFunctions as lib
from toy_auto_race.NavAgents.AgentMod import ModVehicleTest, ModVehicleTrain
from TestingScripts.TrainTest import *

from toy_f110 import ForestSim, TrackSim

import numpy as np


map_name_f = "forest2"
map_name_track = "race_track"
train_n = 1
mod_name_f = f"ModForest_{train_n}"
mod_name_track = f"Mod_{map_name_track}_{train_n}"




def train_mod_f():
    sim_conf = lib.load_conf("std_config")
    env = ForestSim(map_name_f, sim_conf)

    vehicle = ModVehicleTrain(mod_name_f, map_name_f, sim_conf, load=False, h_size=200)
    train_vehicle(env, vehicle, 1000)
    # train_vehicle(env, vehicle, 200000)

# doesnt' work
def train_mod_track():
    sim_conf = lib.load_conf("std_config")
    env = TrackSim(map_name_track, sim_conf)

    # vehicle = ModVehicleTrain(mod_name, map_name, env.sim_conf)
    vehicle = ModVehicleTrain(mod_name_track, map_name_track, sim_conf, load=False, h_size=500)

    train_vehicle(env, vehicle, 500000)

def test_mod_forest():
    sim_conf = lib.load_conf("std_config")
    env = ForestSim(map_name_f, sim_conf)
    vehicle = ModVehicleTest(mod_name_f, map_name_f, sim_conf)

    test_single_vehicle(env, vehicle, True, 100, wait=False)

def test_mod_track():
    sim_conf = lib.load_conf("race_config")
    env = TrackSim(map_name_track, sim_conf)
    vehicle = ModVehicleTest(mod_name_track, map_name_track, sim_conf)

    test_single_vehicle(env, vehicle, False, 100, wait=True, vis=False, add_obs=False)


if __name__ == "__main__":
    
    train_mod_f()
    train_mod_track()
    test_mod_track()

    test_mod_forest()








