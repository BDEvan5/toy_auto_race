
from toy_auto_race.Utils import LibFunctions as lib 
from toy_auto_race.NavAgents.FollowTheGap import ForestFGM, TrackFGM

from toy_f110 import ForestSim, TrackSim
from TrainTest import test_single_vehicle

map_name_forest = "forest2"
map_name_track = "race_track"



def run_follow_the_gap_forest():
    sim_conf = lib.load_conf("fgm_config")
    env = ForestSim(map_name_forest, sim_conf)
    vehicle = ForestFGM()

    # test_single_vehicle(env, vehicle, True, 10, False, vis=True)
    test_single_vehicle(env, vehicle, True, 100, add_obs=True, vis=False)


def run_follow_the_gap_track():
    sim_conf = lib.load_conf("fgm_config")
    env = TrackSim(map_name_track, sim_conf)
    vehicle = TrackFGM()

    # test_single_vehicle(env, vehicle, True, 10, False)
    test_single_vehicle(env, vehicle, True, 100, add_obs=True, vis=False)
    # test_single_vehicle(env, vehicle, True, 100, add_obs=True, vis=True, wait=False)
    # test_single_vehicle(env, vehicle, True, 100, add_obs=True, vis=False)


if __name__ =="__main__":
    run_follow_the_gap_forest()
    run_follow_the_gap_track



