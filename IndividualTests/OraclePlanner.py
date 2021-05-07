
from toy_auto_race.NavAgents.Oracle import Oracle

from TestingScripts.TrainTest import *

from toy_f110 import TrackSim, ForestSim

map_name = "race_track"
run_num = 1


def run_oracle_forest():
    sim_conf = lib.load_conf("std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = Oracle(sim_conf)

    test_oracle_forest(env, vehicle, True, 100, True, wait=False)
    # test_oracle_forest(env, vehicle, True, 1, False, wait=False)

def run_oracle_track():
    sim_conf = lib.load_conf("std_config")
    env = TrackSim(map_name, sim_conf)
    vehicle = Oracle(sim_conf)

    test_oracle_track(env, vehicle, True, 100, add_obs=False, wait=False)


if __name__ == "__main__":
    run_oracle_forest()
    run_oracle_track()

