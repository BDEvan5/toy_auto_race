
from toy_auto_race.NavAgents.AgentNav import NavTrainVehicle, NavTestVehicle

from toy_auto_race.Utils import LibFunctions as lib
from TestingScripts.TrainTest import *

from toy_f110 import ForestSim, TrackSim


map_name_forest = "forest2"
map_name_track = "race_track"
train_n = 1
nav_name_f = f"NavForest_{train_n}"
nav_name_track = f"Nav_{map_name_track}_{train_n}"

"""
Training Functions
"""
def train_nav_forest():
    sim_conf = lib.load_conf("std_config")
    env = ForestSim(map_name_forest, sim_conf)
    vehicle = NavTrainVehicle(nav_name_f, sim_conf, h_size=200)

    train_vehicle(env, vehicle, 1000)
    # train_vehicle(env, vehicle, 200000)

def train_nav_track():
    env = TrackSim(map_name_track)

    vehicle = NavTrainVehicle(nav_name_track, map_name_track, env.sim_conf, load=False, h_size=500)

    train_vehicle(env, vehicle, 500000)

def test_nav_forest():
    sim_conf = lib.load_conf("std_config")
    env = ForestSim(map_name_forest, sim_conf)
    vehicle = NavTestVehicle(nav_name_f, sim_conf)

    test_single_vehicle(env, vehicle, True, 100, wait=False)

def test_nav_track():
    sim_conf = lib.load_conf("race_config")
    env = TrackSim(map_name_track, sim_conf)
    vehicle = NavTestVehicle(nav_name_track, sim_conf)

    test_single_vehicle(env, vehicle, False, 100, wait=True, vis=False, add_obs=False)


if __name__ == "__main__":
    
    # train_nav_forest()

    test_nav_forest()



